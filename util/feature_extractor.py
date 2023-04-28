import torch
from torch import Tensor
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.to(device)
resnet50.eval()

def feature_extractor(x:Tensor)->Tensor:
    res=None

    for name,module in resnet50._modules.items():
        x=module(x)
        if name == "layer4":
            res=x
            break
    res=res.reshape(res.shape[:2]+(res.shape[2]*res.shape[3],))
    return res
