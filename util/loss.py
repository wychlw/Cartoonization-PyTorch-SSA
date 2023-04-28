from conf import conf
<<<<<<< HEAD
import mindspore
import numpy as np
from mindspore import nn
from mindspore import Tensor, ops
from util.vgg import vgg19
from util.effecient_segmentation_new import effecient_segmentation
from typing import Tuple

def L_motion(pred1: Tensor, pred2: Tensor, ori1: Tensor, ori2: Tensor, pred_vgg: Tensor = None, ori_vgg: Tensor = None, VGG=vgg19(True)) -> Tensor:

    def C(t1: Tensor, t2: Tensor) -> Tensor:
        N, C, H, W = t1.shape
        t1 = t1.reshape(N, C, H*W)
        t2 = t2.reshape(N, C, H*W)

        res = t1.transpose((0, 2, 1))@t2

        return res
    
    
    if pred_vgg is None:
        pred2_feature = VGG(pred2)
    else:
        pred2_feature = pred_vgg
    if ori_vgg is None:
        ori2_feature = VGG(ori2)
    else:
        ori2_feature = ori_vgg

    pred1_feature = VGG(ori1)
    ori1_feature = VGG(ori1)

    pred_map = C(pred1_feature, pred2_feature)
    ori_map = C(ori1_feature, ori2_feature)

    cosine_loss = nn.CosineEmbeddingLoss()
    loss = cosine_loss(pred_map, ori_map, ops.ones(pred_map.shape[0]))

    return loss


def L_structure(pred_seq: Tensor, pred_vgg: Tensor = None, VGG = vgg19(True)) -> Tensor:
    if pred_vgg is None:
        pred_vgg = VGG(pred_seq)

    fill_color = effecient_segmentation(pred_seq)
    fill_vgg = VGG(fill_color)

    l1loss = nn.L1Loss()
    loss = l1loss(pred_vgg, fill_vgg)

    return loss


def L_content(pred_seq: Tensor, ori_seq: Tensor, pred_vgg: Tensor = None, ori_vgg: Tensor = None, VGG = vgg19(True)) -> Tensor:

    N, C, H, W = pred_seq.shape

    if pred_vgg is None:
        pred_vgg = VGG(pred_seq)
    if ori_vgg is None:
        ori_vgg = VGG(ori_seq)

    l1loss = nn.L1Loss()
    loss = l1loss(pred_vgg, ori_vgg)/(C*H*W)

    return loss

class Loss_tv(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()
        
    def construct(self, pred_seq):

        N, C, H, W = pred_seq.shape

        dx = pred_seq[:, :, 1:, :]-pred_seq[:, :, :-1, :]
        dy = pred_seq[:, :, :, 1:]-pred_seq[:, :, :, :-1]

        mean = ops.reduce_mean(dx**2)+ops.reduce_mean(dy**2)
        loss = mean/(C*H*W)
        return loss

def L_tv(pred_seq: Tensor) -> Tensor:

    N, C, H, W = pred_seq.shape

    dx = pred_seq[:, :, 1:, :]-pred_seq[:, :, :-1, :]
    dy = pred_seq[:, :, :, 1:]-pred_seq[:, :, :, :-1]

    mean = ops.reduce_mean(dx**2)+ops.reduce_mean(dy**2)
    loss = mean/(C*H*W)

    return loss

class Lsgan_loss_g(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        
    def construct(self, fake_img):
        
        return self.mean(fake_img)
        # return ops.mean((fake_img-1)**2)
    
class Lsgan_loss_d(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        
    def construct(self, real_img, fake_img):

        return 0.5*(self.mean((real_img-1)**2)+self.mean(fake_img**2))
        # return 0.5*(ops.mean((real_img-1)**2)+ops.mean(fake_img**2))

def lsgan_loss_g(fake_img: Tensor) -> Tensor:
    return ops.mean((fake_img-1)**2)

def lsgan_loss_d(real_img: Tensor, fake_img: Tensor) -> Tensor:
    return 0.5*(ops.mean((real_img-1)**2)+ops.mean(fake_img**2))

def lsgan_loss(real: Tensor, fake: Tensor) -> Tuple[Tensor, Tensor]:

    def g(fake_img: Tensor) -> Tensor:
        return ops.mean((fake_img-1)**2)

    def d(real_img: Tensor, fake_img: Tensor) -> Tensor:
        return 0.5*(ops.mean((real_img-1)**2)+ops.mean(fake_img**2))

    return d(real, fake), g(fake)

class Gan_loss_g(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        self.log=ops.Log()
        
    def construct(self, fake_img):

        return -self.mean(self.log(fake_img))
        # return -ops.mean(ops.log(fake_img))
    
class Gan_loss_d(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        self.log=ops.Log()
        
    def construct(self, real_img, fake_img):

        return -self.mean(self.log(real_img)+self.log(1-fake_img))
        # return -ops.mean(ops.log(real_img)+ops.log(1-fake_img))

def gan_loss_g(fake_img: Tensor) -> Tensor:
    return -ops.mean(ops.log(fake_img))

def gan_loss_d(real_img: Tensor, fake_img: Tensor) -> Tensor:
    return -ops.mean(ops.log(real_img)+ops.log(1-fake_img))

def gan_loss(real: Tensor, fake: Tensor) -> Tuple[Tensor, Tensor]:

    def g(fake_img: Tensor) -> Tensor:
        return -ops.mean(ops.log(fake))

    def d(real_img: Tensor, fake_img: Tensor) -> Tensor:
        return -ops.mean(ops.log(real)+ops.log(1-fake))

    return d(real, fake), g(fake)
=======
import torch
from torch import nn
from torch import Tensor
from util.feature_extractor import feature_extractor
from util.vgg import VGG
from util.guided_image_filtering import guided_filter
from util.effecient_segmentation_new import effecient_segmentation
from util.network import Discriminator

import torchvision.utils as vutils


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
        # ps = torch.concat([pred, fill_color]).clone(
        # ).detach().to(torch.device("cpu"))
        # vutils.save_image(ps, "ls.jpg")
        N, C, H, W = pred_vgg.shape
        loss = self.l1loss(pred_vgg, fill_f)*255/(C*H*W)
        return loss


class L_content(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, pred: Tensor, ori: Tensor, pred_vgg: Tensor = None, ori_vgg: Tensor = None):

        if pred_vgg is None:
            pred_vgg = VGG(pred)
        if ori_vgg is None:
            ori_vgg = VGG(ori)

        N, C, H, W = pred_vgg.shape
        loss = self.l1loss(pred_vgg, ori_vgg)*255/(C*H*W)
        return loss


class L_tv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor):
        N, C, H, W = pred.shape
        dx = pred[:, :, 1:, :]-pred[:, :, :-1, :]
        dy = pred[:, :, :, 1:]-pred[:, :, :, :-1]

        loss = torch.abs((torch.mean(dx)+torch.mean(dy))/(C*H*W))

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
        # fake = fake.detach()
        return -torch.mean(torch.log(real)+torch.log(1-fake))

    def forward(self, real, fake):
        real = torch.sigmoid(real)
        fake = torch.sigmoid(fake)
        loss_g = self.gan_loss_g(fake)
        loss_d = self.gan_loss_d(real, fake)
        return loss_g, loss_d


class Lsgan_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def gan_loss_g(self, fake):
        return torch.mean((fake-1)**2)

    def gan_loss_d(self, real, fake):
        # fake = fake.detach()
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

        return out.reshape(N, 1, H, W)/(r_w+g_w+b_w)

    def forward(self, fake: Tensor, real: Tensor = None) -> Tensor:
        fake = self.color_shift(fake)
        if real is None:
            return self.loss.gan_loss_g(self.discriminator(fake))
        else:
            real = self.color_shift(real)
            return self.loss(self.discriminator(real), self.discriminator(fake))
>>>>>>> pytorch
