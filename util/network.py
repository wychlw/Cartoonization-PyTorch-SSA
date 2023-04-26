import torch
from torch import nn
from torch import Tensor
from torch.nn.utils import spectral_norm
from conf import conf


def nothing(t):
    return t


sn = None
if conf["sn"]:
    sn = spectral_norm
else:
    sn = nothing


class EncodingLayer(nn.Module):
    def __init__(self, in_channle: int, out_channel: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channle, out_channels=in_channle,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channle, out_channels=out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

        )

    def forward(self, x: Tensor) -> Tensor:
        ret = self.net(x)
        return ret


class ResidualBlock(nn.Module):
    def __init__(self, channel: int, K: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel,
                      kernel_size=K, stride=1, padding=K//2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=channel, out_channels=channel,
                      kernel_size=K, stride=1, padding=K//2),
        )
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.network(x)
        ret = out+x
        return self.leaky_relu(ret)


class DecodingLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, R: int = 3):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=R, stride=1, padding=R//2)
        self.r = R

        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.resize_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.soft_max = nn.Softmax(dim=1)

    def tramsform(self, now: Tensor, before: Tensor) -> Tensor:
        N, C, H, W = before.shape

        f = before.permute(0, 2, 3, 1).reshape(N*H*W, 1, C)
        g = self.unfold(now).permute(0, 2, 1).reshape(N*H*W, C, self.r**2)
        z = torch.matmul(f, g).squeeze()

        alpha = self.soft_max(z)

        r = torch.matmul(g, alpha.unsqueeze(2)).squeeze()
        out = r.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return out

    def forward(self, now: Tensor, before: Tensor) -> Tensor:
        N, C, H, W = now.shape

        o1 = self.conv1(now)
        o2 = self.resize_bilinear(o1)
        trans = self.tramsform(o2, before)
        o3 = self.conv2(before+trans)
        out = self.leaky_relu(o3)

        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding=3),
            nn.LeakyReLU(0.2, True)
        )

        self.e1 = EncodingLayer(32, 64)
        self.e2 = EncodingLayer(64, 128)

        self.res = nn.Sequential(
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3)
        )

        self.d1 = DecodingLayer(128, 64)
        self.d2 = DecodingLayer(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 3, 7, stride=1, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        o0 = self.proc(x)     # 256 256 32
        o1 = self.e1(o0)      # 128 128 64
        o2 = self.e2(o1)      # 64 64 128
        o3 = self.res(o2)     # 64 64 128
        o4 = self.d1(o3, o1)   # 128 128 64
        o5 = self.d2(o4, o0)   # 256 256 32
        out = self.out(o5)    # 256 256 3
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()

        self.net = nn.Sequential(
            sn(nn.Conv2d(in_channel, 32, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True),

            sn(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True),

            sn(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True)
        )

        self.proc = sn(nn.Conv2d(128, 1, 1, padding=0))

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        out = self.proc(x)
        return out

# class Network(nn.Module):   # reduce the network 4 times so can run on my computer
#     def __init__(self):
#         super().__init__()
#         self.e1 = EncodingLayer(3, 16, 3)
#         self.e2 = EncodingLayer(16, 32, 3)
#         self.r1 = ResidualBlock(32, 3)
#         self.r2 = ResidualBlock(32, 3)
#         self.r3 = ResidualBlock(32, 3)
#         self.r4 = ResidualBlock(32, 3)
#         self.d1 = DecodingLayer(32, 32, 3, 3)
#         self.d2 = DecodingLayer(32, 16, 3, 3)

#         self.out = nn.Sequential(
#             sn(nn.Conv2d(16, 16, 3, 1, 1)),
#             nn.BatchNorm2d(16),
#              nn.LeakyReLU(0.2,True),
#             sn(nn.Conv2d(16, 16, 3, 1, 1)),
#             nn.BatchNorm2d(16),
#              nn.LeakyReLU(0.2,True),
#             sn(nn.Conv2d(16, 3, 3, 1, 1)),
#             nn.ReLU(),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         # %
#         of1, o1 = self.e1(x)   # 64*256*256 64*128*128
#         # %
#         of2, o2 = self.e2(o1)  # 128*128*128 128*64*64
#         # %
#         ro1 = self.r1(o2)     # 128*64*64
#         # %
#         ro2 = self.r2(ro1)    # 128*64*64
#         # %
#         ro3 = self.r3(ro2)    # 128*64*64
#         # %
#         ro4 = self.r4(ro3)    # 128*64*64
#         # %
#         u1 = self.d1(ro4, of2)  # 128*128*128
#         # %
#         u2 = self.d2(u1, of1)  # 64*256*256
#         # %
#         o = self.out(u2)
#         # %

#         return o
