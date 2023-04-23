import torch
from torch import Tensor
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg19 = models.vgg19(pretrained=True)
vgg19.to(device)
vgg19.eval()

# (x-mean)/std
# x*std+mean

vgg_mean = torch.tensor([0.584, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)


def VGG(x: Tensor) -> Tensor:
    x = ((x*0.5+0.5)-vgg_mean)/vgg_std
    res = vgg19.features[:26](x)
    return res
