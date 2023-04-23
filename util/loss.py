from conf import conf
import torch
from torch import nn
from torch import Tensor
from util.feature_extractor import feature_extractor
from util.vgg import VGG
from util.guided_image_filtering import guided_filter
from util.effecient_segmentation_new import effecient_segmentation
from util.network import Discriminator


def gray_ish(t: Tensor) -> Tensor:
    N, C, H, W = t.shape
    ret = t[:, 0]*0.299+t[:, 1]*0.587+t[:, 2]*0.114
    return ret.reshape(N, 1, H, W)


class L_structure(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, pred: Tensor, pred_vgg: Tensor = None):
        if pred_vgg is None:
            pred_vgg = VGG(pred)
        fill_color = effecient_segmentation(pred)
        fill_f = VGG(fill_color)
        loss = self.l1loss(pred_vgg, fill_f)
        return loss


class L_content(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, pred: Tensor, ori: Tensor, pred_vgg: Tensor = None, ori_vgg: Tensor = None):
        N, C, H, W = pred.shape
        if pred_vgg is None:
            pred_vgg = VGG(pred)
        if ori_vgg is None:
            ori_vgg = VGG(ori)

        loss = self.l1loss(pred_vgg, ori_vgg)/(C*H*W)
        return loss


class L_tv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor):
        N, C, H, W = pred.shape
        dx = pred[:, :, 1:, :]-pred[:, :, :-1, :]
        dy = pred[:, :, :, 1:]-pred[:, :, :, :-1]

        loss = (torch.mean(dx)+torch.mean(dy))/(C*H*W)

        return loss


class L_motion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def C(self, t1: Tensor, t2: Tensor) -> Tensor:
        N, C, H, W = t1.shape
        t1 = t1.reshape(N, C, H*W)
        t2 = t2.reshape(N, C, H*W)
        t1 = t1/torch.norm(t1, dim=1, keepdim=True)
        t2 = t2/torch.norm(t2, dim=1, keepdim=True)
        return torch.bmm(t1.permute(0, 2, 1), t2)

    def forward(self, pred1: Tensor, pred2: Tensor, ori1: Tensor, ori2: Tensor):
        pred1_vgg = feature_extractor(pred1)
        pred2_vgg = feature_extractor(pred2)
        ori1_vgg = feature_extractor(ori1)
        ori2_vgg = feature_extractor(ori2)

        pred_map = self.C(pred1_vgg, pred2_vgg)
        ori_map = self.C(ori1_vgg, ori2_vgg)

        loss = self.cosine_loss(pred_map, ori_map, torch.ones(
            pred_map.shape[0]).to(pred1.device))

        return loss


class Gan_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def gan_loss_g(self, fake):
        return -torch.mean(torch.log(fake))

    def gan_loss_d(self, real, fake):
        fake = fake.detach()
        return -torch.mean(torch.log(real)+torch.log(1-fake))

    def forward(self, real, fake):
        loss_g = self.gan_loss_g(fake)
        loss_d = self.gan_loss_d(real, fake)
        return loss_g, loss_d


class Lsgan_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def gan_loss_g(self, fake):
        return torch.mean((fake-1)**2)

    def gan_loss_d(self, real, fake):
        fake = fake.detach()
        return torch.mean((real-1)**2)+torch.mean(fake**2)

    def forward(self, real, fake):
        loss_g = self.gan_loss_g(fake)
        loss_d = self.gan_loss_d(real, fake)
        return loss_g, loss_d


class L_surface(nn.Module):
    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = discriminator

        self.loss = Lsgan_loss()

    def forward(self, fake: Tensor, real: Tensor = None) -> Tensor:
        fake = guided_filter(fake, fake, 8, 0.05)
        if real is None:
            return self.loss.gan_loss_g(self.discriminator(fake))
        else:
            real = guided_filter(real, real, 8, 0.05)
            return self.loss(self.discriminator(real), self.discriminator(fake))


class L_texture(nn.Module):
    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = discriminator

        self.loss = Lsgan_loss()

    def color_shift(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        r_w = (0.199-0.399)*torch.rand(1, dtype=x.dtype, device=x.device)+0.399
        g_w = (0.487-0.687)*torch.rand(1, dtype=x.dtype, device=x.device)+0.687
        b_w = (0.114-0.314)*torch.rand(1, dtype=x.dtype, device=x.device)+0.314

        out = r_w*x[:, 0, :, :]+g_w*x[:, 1, :, :]+b_w*x[:, 2, :, :]

        return out.reshape(N, 1, H, W)

    def forward(self, fake: Tensor, real: Tensor = None) -> Tensor:
        fake = self.color_shift(fake)
        if real is None:
            return self.loss.gan_loss_g(self.discriminator(fake))
        else:
            real = self.color_shift(real)
            return self.loss(self.discriminator(real), self.discriminator(fake))
