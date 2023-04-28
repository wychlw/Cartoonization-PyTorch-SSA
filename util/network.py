<<<<<<< HEAD
import mindspore
from mindspore import nn, Tensor, ops
from mindspore.ops.operations import image_ops
from conf import conf
from util.spectral_norm import Conv2d_Spetral_Norm

if conf["sn"]:
    conv3x3=Conv2d_Spetral_Norm
else:
    conv3x3 = nn.Conv2d

# def conv3x3(in_channel, out_channel, kernel_size=3, stride=1, padding=1):
#     if conf["sn"]:
#         return Conv2d_Spetral_Norm(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, pad_mode="pad")
#     else:
#         return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, pad_mode="pad")


class ResidualBlock(nn.Cell):

    def __init__(self, channel):
        super().__init__()

        self.net = nn.SequentialCell(
            nn.Conv2d(channel, channel, 3, stride=1,
                      padding=1, pad_mode="pad"),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, pad_mode="pad")
        )

    def construct(self, x: Tensor) -> Tensor:
        out = x
        out = self.net(out)

        out = out+x
        out = nn.LeakyReLU()(out)
        return out


class EncodingLayer(nn.Cell):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()

        self.net = nn.SequentialCell(
            nn.Conv2d(in_channel, in_channel, 3, stride=2,
                      padding=1, pad_mode="pad"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, out_channel, 3,
                      stride=1, padding=1, pad_mode="pad"),
            nn.LeakyReLU()
        )

    def construct(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecodingLayer(nn.Cell):

    def transform(self, before: Tensor, now: Tensor, R: int = 2) -> Tensor:
        N, C, H, W = now.shape

        f = ops.transpose(before, (0, 2, 3, 1))  # N H W C
        f = ops.reshape(f, (N*H*W, 1, C))          # N*H*W

        g_ext = ops.transpose(now, (0, 2, 3, 1)) # N H W C
        g_ext = self.extract(g_ext) # N H W C*R
        g_ext = ops.reshape(g_ext, (N*H*W, C, R*R))

        f_ext = ops.reshape(f, (N*H*W, 1, C))
        z = ops.matmul(f_ext, g_ext)
        z = ops.reshape(z, (N*H*W, R*R))
        # z = ops.squeeze(z)
        # alpha = ops.softmax(z, 1)
        alpha=self.soft_max(z)
        alpha_ext = ops.reshape(alpha, (N*H*W, R*R, 1))
        r_ext = ops.matmul(g_ext, alpha_ext)
        r = ops.reshape(r_ext, (N, H, W, C))    # N H W C
        r = ops.transpose(r, (0, 3, 1, 2))      # N C H W

        return before

    def __init__(self, in_channel: int, out_channel: int, K1=3, K2=7):
        super().__init__()
        
        self.resize_bilinear = nn.ResizeBilinear()

        self.conv1 = nn.Conv2d(in_channel, in_channel,
                               K1, stride=1, padding=K1//2, pad_mode="pad")
        self.conv2 = nn.Conv2d(in_channel, out_channel,
                               K2, stride=1, padding=K2//2, pad_mode="pad")
        self.leaky_relu = nn.LeakyReLU()

        # The ExtrctImagePatches currently has issue when using larger than [1,1,2,2]
        
        self.extract = ops.operations._inner_ops.ExtractImagePatches([1, 1, 2, 2], [1, 1, 1, 1], [
                          1, 1, 1, 1], "same")
        
        self.soft_max=nn.Softmax(1)

    def construct(self, now: Tensor, before: Tensor) -> Tensor:
        N, C, H, W = now.shape

        o1 = self.resize_bilinear(now, (H*2, W*2))
        trans = self.transform(before, o1)
        # trans = o1
        out = self.conv1(before+trans)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
=======
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
>>>>>>> pytorch

        return out


<<<<<<< HEAD
class Generator(nn.Cell):

    def __init__(self):
        super().__init__()

        self.proc = nn.SequentialCell(
            nn.Conv2d(3, 32, 7, stride=1, padding=3, pad_mode="pad"),
            nn.LeakyReLU()
=======
class DecodingLayer2(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, R: int = 3):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=R, stride=1, padding=R//2)
        self.r = R

        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.resize_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel*2, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
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
        o3 = self.conv2(torch.cat((before, trans),dim=1))
        out = self.leaky_relu(o3)

        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding=3),
            nn.LeakyReLU(0.2, True)
>>>>>>> pytorch
        )

        self.e1 = EncodingLayer(32, 64)
        self.e2 = EncodingLayer(64, 128)

<<<<<<< HEAD
        self.res = nn.SequentialCell(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.afres = nn.SequentialCell(
            nn.Conv2d(128, 64, 3, stride=1, padding=1, pad_mode="pad"),
            nn.LeakyReLU()
        )

        self.d1 = DecodingLayer(64, 32)
        self.d2 = DecodingLayer(32, 3, 3, 7)

        self.out = nn.Tanh()

    def construct(self, input: Tensor) -> Tensor:
        o0 = self.proc(input)  # 256, 256, 32

        o1 = self.e1(o0)  # 128, 128, 64

        o2 = self.e2(o1)  # 64, 64, 128

        o3 = self.res(o2)  # 64, 64, 128

        o4 = self.afres(o3)  # 64, 64, 64

        o5 = self.d1(o4, o1)  # 128, 128, 32

        o6 = self.d2(o5, o0)  # 256, 256, 3

        return self.out(o6)


class Discriminator(nn.Cell):

    def __init__(self, in_channel=3):
        super().__init__()

        self.net = nn.SequentialCell(
           conv3x3(in_channel, 32, 3, stride=2, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),
           conv3x3(32, 32, 3, stride=1, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),

           conv3x3(32, 64, 3, stride=2, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),
           conv3x3(64, 64, 3, stride=1, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),

           conv3x3(64, 128, 3, stride=2, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),
           conv3x3(128, 128, 3, stride=1, padding=1, pad_mode="pad"),
           nn.LeakyReLU()
        )

        self.snc = None
        
        self.mean=ops.ReduceMean()
        self.den = nn.Dense(128, 1)

    def construct(self, input: Tensor) -> Tensor:
        
        a = self.net(input)

        b = self.mean(a, (2, 3))
        
        c = self.den(b)

        return c
=======
        self.res = nn.Sequential(
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3)
        )

        self.d1 = DecodingLayer2(128, 64)
        self.d2 = DecodingLayer2(64, 32)

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
>>>>>>> pytorch
