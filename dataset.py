import numpy as np
from conf import conf
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class Real_train_dataset(Dataset):
    real_dataset_path = [
        "data/train_photo/",
        "data/test/real/",
        "data/face_photo/",
        "data/scenery_photo/",

    ]

    img = []
    idx_map: np.ndarray

    def __init__(self):
        super().__init__()
        for i in self.real_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

    def __getitem__(self, index):
        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)


class Real_test_dataset(Dataset):
    real_dataset_path = [
        "data/test/test_photo256/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self):
        super().__init__()
        for i in self.real_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

    def __getitem__(self, index):
        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)


class Cartoon_train_dataset(Dataset):

    cartoon_dataset_path = [
        "data/Hayao/style/",
        "data/Paprika/style/",
        "data/Shinkai/style/",
        "data/SummerWar/style/",
        "data/Paprika/style/",
        "data/face_cartoon/kyoto_face/",
        "data/face_cartoon/pa_face/",
        "data/scenery_cartoon/hayao/",
        "data/scenery_cartoon/hosoda/",
        "data/scenery_cartoon/shinkai/",
        "data/scenery_cartoon/hayao/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self):
        super().__init__()

        for i in self.cartoon_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

    def __getitem__(self, index):
        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)


class Cartoon_test_dataset(Dataset):

    cartoon_dataset_path = [
        "data/spirit_away/",
        "data/test/label_map/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self):
        super().__init__()

        for i in self.cartoon_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

    def __getitem__(self, index):
        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)
