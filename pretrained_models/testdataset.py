import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from mydataset import MyDataset
import torch.nn as nn
import numpy as np


# 定义时的接口与之前的ImageFolder一样
traindir = '/data2/yujin/cadene_cnn/data/train_val/train'
mydataset = MyDataset(traindir, mode='train')
myloader = torch.utils.data.DataLoader(mydataset, batch_size=4, shuffle=True)


# related to mix_up
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# related to mix_up
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


TOTAL_EPOCHS = 100
MIXUP_EPOCH = 50


criterion = nn.CrossEntropyLoss()

for epoch in range(TOTAL_EPOCHS):

    if epoch > MIXUP_EPOCH:
        mix_up_flag = True
    else:
        mix_up_flag = False
    for i, batch in enumerate(myloader):  # Mydataset的接口 is different from ImageFolder
        images = batch['img']  # Mydataset的接口 is different from ImageFolder
        labels = batch['label']  # Mydataset的接口 is different from ImageFolder

        if mix_up_flag:
            inputs, targets_a, targets_b, lam = mixup_data(images.cuda(), labels.cuda(), alpha=1.0)  # related to mix_up
            outputs = model(inputs)
            optimizer.zero_grad()

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # related to mix_up
            loss.backward()
            optimizer.step()



