import torch
from torch import Tensor
from skimage import segmentation, color
from conf import conf
from joblib import Parallel, delayed

def __proc(img:Tensor)->Tensor:
    img = img.permute(1, 2, 0).detach().numpy()
    label = segmentation.felzenszwalb(img,1,0.8,100)
    gen = color.label2rgb(label, img,kind="avg")
    gen = Tensor(gen)
    gen = gen.permute(2, 0, 1)
    return gen

def effecient_segmentation(x: Tensor, *args, **kwargs) -> Tensor:
    num_job = x.shape[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(__proc)(img) for img in x)
    return torch.stack(batch_out)

