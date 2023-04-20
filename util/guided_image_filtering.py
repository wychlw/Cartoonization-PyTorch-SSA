import torch
from torch import Tensor


def box_filter(src: Tensor, r: int) -> Tensor:
    N, C, H, W = src.shape
    weight = 1/((2*r+1)**2)
    kernel = torch.ones((C, 1, 2*r+1, 2*r+1), device=src.device)*weight
    out = torch.nn.functional.conv2d(src, kernel, padding=r, groups=C)
    return out


def guided_filter(src: Tensor, guided: Tensor, r, eps=0.01) -> Tensor:
    N, C, H, W = src.shape
    norm = box_filter(torch.ones(
        (1, 1, H, W), dtype=src.dtype, device=src.device), r)

    meax_src = box_filter(src, r)/norm
    mean_guided = box_filter(guided, r)/norm
    conv = box_filter(src*guided, r)/norm-meax_src*mean_guided
    var = box_filter(src*src, r)/norm-meax_src*meax_src

    a = conv/(var+eps)
    b = mean_guided-a*meax_src

    mean_a = box_filter(a, r)/norm
    mean_b = box_filter(b, r)/norm

    out = mean_a*src+mean_b
    return out