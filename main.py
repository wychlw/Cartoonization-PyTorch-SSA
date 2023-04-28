
import torch

from dataset import *
from train import *
from conf import conf

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

generator = Generator().to(device)
surface_disc = Discriminator().to(device)
texture_disc = Discriminator(1).to(device)

print(generator)
print(surface_disc)
print(texture_disc)

train(generator, surface_disc, texture_disc, device)

print("Done!")
