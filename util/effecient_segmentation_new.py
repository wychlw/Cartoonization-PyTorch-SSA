from mindspore import Tensor, ops
from conf import conf
from skimage import segmentation, color

from joblib import Parallel, delayed
def effecient_segmentation(x:Tensor,*args,**kwargs)->Tensor:
    
    def proc(img:Tensor)->Tensor:
        
        img=img.transpose(1,2,0)
        img=img.asnumpy()
        seg_img=segmentation.felzenszwalb(img,1,0.8,200)
        out_img=color.label2rgb(seg_img,img,kind="avg")
        out_img=Tensor(out_img)
        out_img=out_img.transpose(2,0,1)
        return out_img

    imgs = [x[i,:,:,:] for i in range(x.shape[0])]

    # batch_out = Parallel(n_jobs=x.shape[0])(delayed(proc)(img) for img in imgs)

    batch_out = [proc(img) for img in imgs]

    return ops.stack(batch_out)