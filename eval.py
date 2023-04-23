import numpy as np
from conf import conf
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from util.network import *
from dataset import *


def saveimg(t, sn):
    st = t.clone().detach().to(torch.device("cpu"))
    vutils.save_image(st, sn+".jpg")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

generator = Generator().to(device)
gen_checkpoint = torch.load("model/generator.pth")
generator.load_state_dict(gen_checkpoint)
generator.eval()

real_test_data = Real_test_dataset()
real_test_dl = DataLoader(real_test_data, batch_size=conf["batch"])

real = None
for i in real_test_dl:
    real = i[0].to(device)
    break
pred = generator(real)

pred = pred.cpu()
real = real.cpu()


saveimg(real[0], "rt")
saveimg(pred[0], "pt")

print(pred[0])

# npr = pred.detach()
# npi = npr[0].permute(1, 2, 0).numpy()
# print("dsfhiuhc")
# print(real.shape)
# plt.imshow(real[0].permute(1, 2, 0).numpy())
