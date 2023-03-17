import mindspore
from mindspore import Tensor, ops, nn, dataset
from util.network import Generator, Discriminator
from conf import conf
import os
from util.loss import L_content, L_motion, L_structure, L_tv, lsgan_loss, lsgan_loss_g, lsgan_loss_d
from util.vgg import vgg19
from util.guided_image_filter import guided_filter
from util.utility import gray_ish
from typing import Tuple
from util.dataset import MyDatasetLoader
from matplotlib import pyplot as plt
from mindspore.train import Model

class WithLossCellG(nn.Cell):
    def __init__(self,net_surface,net_texture,net_generator,VGG):
        super().__init__()
        self.net_surface=net_surface
        self.net_texture=net_texture
        self.net_generator=net_generator
        self.VGG=VGG

    def generator_forward(self,ori: Tensor, last_ori: Tensor = None, last_pred: Tensor = None) -> Tuple[Tensor]:
        
        pred = self.net_generator(ori)

        ori_vgg = self.VGG(ori)
        pred_vgg = self.VGG(pred)
        pred_gray = gray_ish(pred)
        pred_guided = guided_filter(pred, pred, 8, 0.05)

        surface_loss = conf["W_surface"] * \
            lsgan_loss_g(self.net_surface(pred_guided))
        texture_loss = conf["W_texture"] * \
            lsgan_loss_g(self.net_texture(pred_gray))
        structure_loss = conf["W_structure"]*L_structure(pred, pred_vgg, self.VGG)
        content_loss = conf["W_content"] * \
            L_content(pred, ori, pred_vgg, ori_vgg, self.VGG)
        tv_loss = conf["W_tv"]*L_tv(pred)

        motion_loss = 0
        if last_ori is not None and last_pred is not None:
            motion_loss = conf["W_motion"] * \
                L_motion(last_pred, pred, last_ori, ori, pred_vgg, ori_vgg, self.VGG)

        loss = surface_loss+texture_loss+structure_loss+content_loss+tv_loss+motion_loss

        return loss, pred
    
    def construct(self, x):
        loss,out=self.generator_forward(x)
        return loss, pred

class WithLossCellS(nn.Cell):
    def __init__(self,net_surface,net_generator,VGG):
        super().__init__()
        self.net_surface=net_surface
        self.net_generator=net_generator
        self.VGG=VGG

    def surface_discriminator_forward(self,real: Tensor, fake: Tensor) -> Tuple[Tensor]:
        real_guided = guided_filter(real, real, 8, 0.05)
        fake_guided = guided_filter(fake, fake, 8, 0.05)
        
        fake_guided = ops.stop_gradient(fake_guided)

        surface_loss = lsgan_loss_d(self.net_surface(
            real_guided), self.net_surface(fake_guided))

        return surface_loss    
    
    def construct(self,cartoon_data,real_data):
        # generated=self.net_generator(real_data)

        loss=self.surface_discriminator_forward(cartoon_data,generated)
        return loss

class WithLossCellT(nn.Cell):
    def __init__(self,net_texture,net_generator,VGG):
        super().__init__()
        self.net_texture=net_texture
        self.net_generator=net_generator
        self.VGG=VGG
    
    def texture_discriminator_forward(self,real: Tensor, fake: Tensor) -> Tuple[Tensor]:
        real_gray = gray_ish(real)
        fake_gray = gray_ish(fake)
        
        fake_gray = ops.stop_gradient(fake_gray)

        texture_loss = lsgan_loss_d(self.net_texture(
            real_gray), self.net_texture(fake_gray))

        return texture_loss
    
    def construct(self,cartoon_data,real_data):
        # generated=self.net_generator(real_data)

        loss=self.texture_discriminator_forward(cartoon_data,generated)
        return loss

class GAN(nn.Cell):
    def __init__(self,netG,netS,netT):
        super().__init__()
        self.netG=netG
        self.netS=netS
        self.netT=netT

    def construct(self,cartoon,real):
        
        output_G, pred=self.netG(real).view(-1)
        g_loss=output_G.mean()
        
        output_S=self.netS(cartoon,pred).view(-1)
        output_T=self.netT(cartoon,pred).view(-1)
        s_loss=output_S.mean()
        t_loss=output_T.mean()
        
        return s_loss,t_loss,g_loss

def train():

    
    generator = Generator()
    surface_discriminator = Discriminator(3)
    texture_discriminator = Discriminator(1)

    VGG = vgg19(True)

    if conf["continue_training"]:
        if os.path.exists("./model/generator.ckpt"):
            mindspore.load_checkpoint("./model/generator.ckpt", generator)
        if os.path.exists("./model/surface_discriminator.ckpt"):
            mindspore.load_checkpoint("./model/surface_discriminator.ckpt", surface_discriminator)
        if os.path.exists("./model/surface_discriminator.ckpt"):
            mindspore.load_checkpoint("./model/texture_discriminator.ckpt", texture_discriminator)

    learning_rate = conf["lr"]
    optimizer_generator = nn.Adam(
        generator.trainable_params(), learning_rate=learning_rate)
    optimizer_texture_generator = nn.Adam(
        surface_discriminator.trainable_params(), learning_rate=learning_rate)
    optimizer_surface_generator = nn.Adam(
        texture_discriminator.trainable_params(), learning_rate=learning_rate)
    optimizer_generator.update_parameters_name('optim_g.')
    optimizer_surface_generator.update_parameters_name('optim_s_d.')
    optimizer_texture_generator.update_parameters_name('optim_t_d.')

    # grad_generator_fn = ops.value_and_grad(
    #     generator_forward, None, optimizer_generator.parameters, has_aux=True)
    # grad_surface_discriminator_fn = ops.value_and_grad(
    #     surface_discriminator_forward, None, optimizer_surface_generator.parameters
    # )
    # grad_texture_discriminator_fn = ops.value_and_grad(
    #     texture_discriminator_forward, None, optimizer_texture_generator.parameters
    # )

    real_train_loader = MyDatasetLoader(
        conf["real_train_dataset"])
    real_train = dataset.GeneratorDataset(
        real_train_loader, ["real_imgs"], shuffle=True)

    real_test_loader = MyDatasetLoader(
        conf["real_test_dataset"])
    real_test = dataset.GeneratorDataset(
        real_test_loader, ["real_imgs"], shuffle=True)

    cartoon_train_loader = MyDatasetLoader(
        conf["cartoon_train_dataset"])
    cartoon_train = dataset.GeneratorDataset(
        cartoon_train_loader, ["cartoon_imgs"], shuffle=True)

    cartoon_test_loader = MyDatasetLoader(
        conf["cartoon_test_dataset"])
    cartoon_test = dataset.GeneratorDataset(
        cartoon_test_loader, ["cartoon_imgs"], shuffle=True)

    real_train=real_train.batch(conf["batch"])
    real_test=real_test.batch(conf["batch"])
    cartoon_train=cartoon_train.batch(conf["batch"])
    cartoon_test=cartoon_test.batch(conf["batch"])

    netG=WithLossCellG(surface_discriminator,texture_discriminator,generator,VGG)
    netS=WithLossCellS(surface_discriminator,generator,VGG)
    netT=WithLossCellT(texture_discriminator,generator,VGG)

    train_step_G=nn.TrainOneStepCell(netG,optimizer_generator)
    train_step_S=nn.TrainOneStepCell(netS,optimizer_surface_generator)
    train_step_T=nn.TrainOneStepCell(netT,optimizer_texture_generator)

    gan_net=GAN(train_step_G,train_step_S,train_step_T)

    print("Starting Training Loop...")
    print(len(real_train_loader),len(real_test_loader))

    G_losses = []
    D_losses = []

    total = max(len(real_train_loader), len(real_test_loader))

    for epoch in range(conf["epoch"]):
        generator.set_train()
        surface_discriminator.set_train()
        texture_discriminator.set_train()
        gan_net.set_train()

        for i, (real, cartoon) in enumerate(zip(real_train.create_tuple_iterator(), cartoon_train.create_tuple_iterator())):

            real=real[0]
            cartoon=cartoon[0]

            s_loss,t_loss,g_loss=gan_net(cartoon,real)
            d_loss=s_loss+t_loss
            
            if i % 100 == 0 or i == total-1:
                print("[%d/%d][%d/%d] Loss_D:%7.4f Loss_G:%7.4f" % (
                    epoch+1, conf["epoch"], (i+1)*conf["batch"], total, d_loss.asnumpy(), g_loss.asnumpy()))
                if i == 2000:
                    break;

            D_losses.append(d_loss.asnumpy())
            G_losses.append(g_loss.asnumpy())
        
        print("Saving...")

        mindspore.save_checkpoint(generator, "./model/generator.ckpt")
        mindspore.save_checkpoint(surface_discriminator,
                                  "./model/surface_discriminator.ckpt")
        mindspore.save_checkpoint(texture_discriminator,
                                  "./model/texture_discriminator.ckpt")
        
        print("Saved...")
        
    plt.figure(figsize=(10, 5))
    plt.title("Losses")
    plt.plot(G_losses, label="G", color="blue")
    plt.plot(D_losses, label="D", color="orange")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()