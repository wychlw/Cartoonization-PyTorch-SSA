import torch
from torch import Tensor
from skimage import segmentation, color, graph
from conf import conf
from joblib import Parallel, delayed


device = "cuda" if torch.cuda.is_available() else "cpu"


def __proc(img: Tensor) -> Tensor:
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    label = segmentation.felzenszwalb(img, 1, 0.8, 60)
    # label = segmentation.slic(
    #     img, compactness=30, n_segments=400, start_label=1)
    # rag = graph.rag_mean_color(img, label)
    # label2 = graph.cut_normalized(label, rag)
    # label2 = graph.cut_threshold(label, rag, 30)
    gen = color.label2rgb(label, img, kind="avg")
    gen = Tensor(gen)
    gen = gen.permute(2, 0, 1)
    return gen


def effecient_segmentation(x: Tensor, *args, **kwargs) -> Tensor:
    num_job = x.shape[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(__proc)(img) for img in x)
    res = torch.stack(batch_out)
    return res.to(device)
