import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
import nibabel as nib
from tqdm import tqdm
import json

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def conv3x3(inplanes, planes, ksize=3):
    return nn.Sequential(
            nn.Conv3d(inplanes, planes, ksize, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))

def repeat(x, n=3):
    # nc333
    b, c, h, w, t = x.shape
    x = x.unsqueeze(5).unsqueeze(4).unsqueeze(3)
    x = x.repeat(1, 1, 1, n, 1, n, 1, n)
    return x.view(b, c, n*h, n*w, n*t)

class DeepMedic(nn.Module):
    def __init__(self, c=1, n1=30, n2=40, n3=50, m=150, up=True):
        super(DeepMedic, self).__init__()
        n = 2*n3
        self.branch1 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.branch2 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=3,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                conv3x3(n, m, 1),
                conv3x3(m, m, 1),
                nn.Conv3d(m, 3, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x1, x2 = inputs
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x2 = self.up3(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x

#
# x1 = torch.rand(8, 1, 25, 25, 25)
# x2 = torch.rand(8, 1, 19, 19, 19)
#
# x1, x2 = x1.cuda(), x2.cuda()
# model = DeepMedic().cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer.zero_grad()
#
#
# x = model((x1, x2))
# print(x.size())
#
# logits = torch.softmax(x, dim=1)
# preds = torch.argmax(logits, dim=1)
#
# criterion = torch.nn.MSELoss()
# loss = criterion(logits.reshape(-1)[:10].float(), preds.reshape(-1)[:10].float())**2
#
# loss.backward()
#
# print(loss.item())
#
