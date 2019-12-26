import cv2
from torch.utils.data import Dataset
import numpy as np
import os
import torch
try:
    import moxing as mox
except:
    pass
else:
    os.system('pip install albumentations')
from albumentations import (IAAPerspective, RandomBrightnessContrast, HueSaturationValue, ChannelDropout, ISONoise,
                            GaussNoise, ShiftScaleRotate, ElasticTransform, Normalize, RandomResizedCrop, Resize, Compose, OneOf, HorizontalFlip)


class MyDataset(Dataset):
    def __init__(self, dir, mode='train'):

        self.height = 224
        self.width = 224
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # self.mean = (0.46118198, 0.52168848, 0.58071129)
        # self.std = (0.32327064, 0.3020832, 0.29525993)
        self.img_lists = []
        self.mode = mode
        class_dirs = os.listdir(dir)
        for class_dir in class_dirs:
            img_names = os.listdir(os.path.join(dir, class_dir))
            for img_name in img_names:
                img_item = {}
                img_item['path'] = os.path.join(dir, class_dir, img_name)
                img_item['label'] = int(class_dir)
                self.img_lists.append(img_item)

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img_item = self.img_lists[idx]
        img = cv2.imread(img_item['path'], 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = img_item['label']

        if self.mode == 'train':
            aug = Compose([
                # OneOf([
                # RandomBrightnessContrast(brightness_limit=0.3, p=0.5),
                # HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5)], p=1),
                # # ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
                # ISONoise(color_shift=(0.01, 0.05), intensity=(0.0, 0.0), always_apply=False, p=0.5),
                # GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                Normalize(mean=self.mean, std=self.std),
                HorizontalFlip(p=0.5),
                # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                RandomResizedCrop(self.height, self.width, scale=(0.08, 1.0), ratio=(0.75, 1.333))])
        else:
            aug = Compose([Normalize(mean=self.mean, std=self.std),
                           Resize(self.height, self.width, interpolation=1, always_apply=False, p=1)])

        # aug = FancyPCA(alpha=0.1)
        # aug = RandomBrightnessContrast(brightness_limit=0.5, p=1)
        # aug = HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1)
        # aug = ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=1)
        # aug = ISONoise(color_shift=(0.01, 0.05), intensity=(0.0, 0.0), always_apply=False, p=1)
        # aug = GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5)
        # aug = ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=45, p=1)
        # aug = RandomResizedCrop(224, 224, scale=(0.5, 1.5), ratio=(0.75, 1.33), p=1)

        image = aug(image=img)['image']

        # image_name = img_item['path'][img_item['path'].rfind('/')+1:]
        # img_raw_dir = './data_aug'
        # cv2.imwrite(os.path.join(img_raw_dir, image_name), img)
        # image_name = image_name[:-4] + '_aug.jpg'
        # cv2.imwrite(os.path.join(img_raw_dir, image_name), image)

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.copy()).float()
        label = torch.tensor(label).long()

        sample = {'img': image, 'label': label}

        return sample
