import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from mydataset import MyDataset
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

os.system('rm -r data_aug/')
os.mkdir('data_aug')

traindir = 'data_small'
valdir = 'data_small'




#----------------------------------wq--------------------------------------------------------------------------------


class AddGaussNoise(object):
    def __init__(self, ns):
        self.NS = ns

    def __call__(self, image):
        assert self.NS <= 1
        image_temp = np.array(image, dtype=np.float32)
        noise = np.random.randint(int(-255 * self.NS), int(255 * self.NS),
                                  size=image_temp.shape)
        ny_image = np.add(image_temp, noise)
        ny_image = (ny_image >= 255) * 255 + \
                   (ny_image <= 0) * 0 + \
                   ((ny_image > 0) & (ny_image < 255)) * ny_image
        ny_image = ny_image.astype(np.uint8)
        return ny_image




class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))
        img = img.resize(self.size, Image.BILINEAR)
        return img


normalize = transforms.Normalize(mean=[0.58071129, 0.52168848, 0.46118198],std=[0.29525993, 0.3020832, 0.32327064])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        AddGaussNoise(ns=1),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08),
        # transforms.RandomHorizontalFlip(),
        # # # transforms.RandomAffine(degrees=45, shear=30, fillcolor=0, resample=PIL.Image.BILINEAR),
        # transforms.RandomRotation(degrees=45),
        # transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=3),
        # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        # normalize,
    ]))

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

#---------------------------------wq---------------------------------------------------------------------------------





for j in range(5):
    for i, (images, target) in enumerate(train_loader):
        images = images[0].numpy().transpose((1, 2, 0)) * 255
        images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./data_aug/{:3d}_{:3d}.jpg'.format(i, j), images)

