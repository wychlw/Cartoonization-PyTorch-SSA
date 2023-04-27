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
        "data/test/test_photo256/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self, preload=True):
        super().__init__()
        for i in self.real_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

        self.preload = preload
        if self.preload:
            to_tensor = transforms.ToTensor()
            resize = transforms.Resize([256, 256], antialias=True)
            self.preimg = [resize(to_tensor(Image.open(i[0])))
                           for i in self.img]

    def __getitem__(self, index):

        if self.preload:
            pic = self.preimg[self.idx_map[index]]
            return pic, 0

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
        "data/spirit_away/",
        "data/test/label_map/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self, preload=True):
        super().__init__()

        for i in self.cartoon_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 1] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

        self.preload = preload
        if self.preload:
            to_tensor = transforms.ToTensor()
            resize = transforms.Resize([256, 256], antialias=True)
            self.preimg = [resize(to_tensor(Image.open(i[0])))
                           for i in self.img]

    def __getitem__(self, index):

        if self.preload:
            pic = self.preimg[self.idx_map[index]]
            return pic, 1

        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)


class Real_scenery_dataset(Dataset):
    real_dataset_path = [
        "data/scenery_photo/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self, preload=True):
        super().__init__()
        for i in self.real_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

        self.preload = preload
        if self.preload:
            to_tensor = transforms.ToTensor()
            resize = transforms.Resize([256, 256], antialias=True)
            self.preimg = [resize(to_tensor(Image.open(i[0])))
                           for i in self.img]

    def __getitem__(self, index):

        if self.preload:
            pic = self.preimg[self.idx_map[index]]
            return pic, 0

        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)


class Real_face_dataset(Dataset):
    real_dataset_path = [
        "data/face_photo/",
    ]

    img = []
    idx_map: np.ndarray

    def __init__(self, preload=True):
        super().__init__()
        for i in self.real_dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            with_labels = [[m, 0] for m in paths]
            self.img.extend(with_labels)
        self.idx_map = np.arange(len(self.img))
        np.random.shuffle(self.idx_map)

        self.preload = preload
        if self.preload:
            to_tensor = transforms.ToTensor()
            resize = transforms.Resize([256, 256], antialias=True)
            self.preimg = [resize(to_tensor(Image.open(i[0])))
                           for i in self.img]

    def __getitem__(self, index):

        if self.preload:
            pic = self.preimg[self.idx_map[index]]
            return pic, 0

        path, label = self.img[self.idx_map[index]]
        pic = Image.open(path)
        pic = transforms.ToTensor()(pic)
        pic = transforms.Resize([256, 256], antialias=True)(pic)
        return pic, label

    def __len__(self):
        return len(self.img)
