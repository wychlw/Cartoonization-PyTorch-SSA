import torch
from torch import Tensor
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.to(device)
vgg16.eval()

def VGG(x:Tensor)->Tensor:
    res=vgg16(x)
    return res