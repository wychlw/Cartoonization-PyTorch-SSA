import torch
from dataset import *
from util.network import Generator, Discriminator
from util.loss import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.vgg import VGG


def train(generator: Generator, surface_disc: Discriminator, texture_disc: Discriminator, device: str = "CPU"):

    if conf["continue_training"] and "generator.pth" in os.listdir("model"):
        gen_checkpoint = torch.load("model/generator.pth")
        surface_checkpoint = torch.load("model/surface.pth")
        texture_checkpoint = torch.load("model/texture.pth")
        generator.load_state_dict(gen_checkpoint)
        surface_disc.load_state_dict(surface_checkpoint)
        texture_disc.load_state_dict(texture_checkpoint)

    structure_loss = L_structure()
    content_loss = L_content()
    tv_loss = L_tv()
    surface_loss = L_surface(surface_disc)
    texture_loss = L_texture(texture_disc)

    cartoon_train_data = Cartoon_train_dataset()
    real_train_data = Real_train_dataset()
    cartoon_train_dl = DataLoader(cartoon_train_data, batch_size=conf["batch"])
    real_train_dl = DataLoader(real_train_data, batch_size=conf["batch"])

    generator.train()
    surface_disc.train()
    texture_disc.train()

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=conf["lr"])
    surface_disc_optimizer = torch.optim.Adam(
        surface_disc.parameters(), lr=conf["lr"])
    texture_disc_optimizer = torch.optim.Adam(
        texture_disc.parameters(), lr=conf["lr"])

    total = max(len(cartoon_train_dl), len(real_train_dl))

    for epoch in range(conf["epoch"]):
        pbar = tqdm(total=total)
        for batch, (cartoon, real) in enumerate(zip(cartoon_train_dl, real_train_dl)):
            cartoon, real = cartoon[0].to(device), real[0].to(device)

            pred = generator(real)
            pred_vgg = VGG(pred)
            real_vgg = VGG(real)

            # generator.train()
            # surface_disc.eval()
            # texture_disc.eval()

            surface_g, surface_d = surface_loss(pred, real)
            texture_g, texture_d = texture_loss(pred, real)

            generate_loss = structure_loss(pred, pred_vgg)*conf["W_structure"] + \
                content_loss(pred, real, pred_vgg, real_vgg)*conf["W_content"] + \
                tv_loss(pred)*conf["W_tv"] + \
                surface_g*conf["W_surface"] + \
                texture_g*conf["W_texture"]

            # generator_optimizer.zero_grad()
            generate_loss.backward()
            # generator_optimizer.step()

            # surface_disc_optimizer.zero_grad()
            surface_d.backward()
            # surface_disc_optimizer.step()

            # texture_disc_optimizer.zero_grad()
            texture_d.backward()
            # texture_disc_optimizer.step()

            # if batch % conf["grad_batch"] == 0:
            generator_optimizer.step()
            generator_optimizer.zero_grad()
            surface_disc_optimizer.step()
            surface_disc_optimizer.zero_grad()
            texture_disc_optimizer.step()
            texture_disc_optimizer.zero_grad()

            pbar.update(real.shape[0])

            if batch % 10 == 0:
                loss_g = generate_loss.item()
                loss_d = surface_d.item()+texture_d.item()
                print(f"Loss_G: {loss_g:>7f}  Loss_D: {loss_d:>7f}")

                torch.save(generator.state_dict(), "model/generator.pth")
                torch.save(surface_disc.state_dict(), "model/surface.pth")
                torch.save(texture_disc.state_dict(), "model/texture.pth")

        torch.save(generator.state_dict(), "model/generator.pth")
        torch.save(surface_disc.state_dict(), "model/surface.pth")
        torch.save(texture_disc.state_dict(), "model/texture.pth")
