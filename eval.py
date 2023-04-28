import numpy as np
from conf import conf
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from util.network import *
from dataset import *
from tqdm import tqdm


def saveimg(arr, type_name="no_type", start=0):
    arr = arr.clone().detach().to(torch.device("cpu"))
    for i in arr:
        vutils.save_image(i, "result/"+type_name+"/"+str(start)+".jpg")
        start += 1
    return start


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

generator = Generator().to(device)
gen_checkpoint = torch.load("model/generator.pth")
generator.load_state_dict(gen_checkpoint)
generator.eval()

real_face = Real_face_dataset(False)
real_scenery = Real_scenery_dataset(False)

real_face_loader = DataLoader(
    real_face, batch_size=conf["batch"], shuffle=True)
real_scenery_loader = DataLoader(
    real_scenery, batch_size=conf["batch"], shuffle=True)

faidx = 0
scidx = 0

pbar = tqdm(total=100//conf["batch"]+1)

for i, (real_face, real_scenery) in enumerate(zip(real_face_loader, real_scenery_loader)):
    if i*conf["batch"] > 100:
        break
    real_face = real_face[0].to(device)
    real_scenery = real_scenery[0].to(device)
    fake_face = generator(real_face)
    fake_scenery = generator(real_scenery)
    faidx = saveimg(fake_face, "face", faidx)
    scidx = saveimg(fake_scenery, "scenery", scidx)
    pbar.update(1)
