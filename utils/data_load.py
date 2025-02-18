import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        gt = int(data[1])
        item = (data[0], gt)
        images.append(item)
    return images





class OfficeImage(data.Dataset):
    def __init__(self, root, label, split="train", transform=None):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.split = split
        self.imgs = imgs
        self.transform = transform
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert("RGB")

        img = img.resize((256, 256), Image.BILINEAR)

        if self.split == "train":
            w, h = img.size
            tw, th = (227, 227)
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)
            img = img.crop((x1, y1, x1 + tw, y1 + th))
        if self.split == "test":
            img = img.crop((15, 15, 242, 242))

        img = np.array(img, dtype=np.float32)
        img = img[:, :, ::-1]
        img = img - self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.imgs)


def transform_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def transform_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


class OfficeHomeImage(data.Dataset):
    def __init__(self, root, label, split="train", transform=None):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.split = split
        self.imgs = imgs
        self.transform = transform
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.transform_train=transform_train()
        self.transform_test=transform_test()


    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert("RGB")
        if self.split == "train":
            img = self.transform_train(img)
        if self.split == "test":
            img = self.transform_test(img)

        return img, target

    def __len__(self):
        return len(self.imgs)