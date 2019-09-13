import os 
import random 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import time
import cv2
from PIL import Image
import numpy as np
from unet import UNet


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up11 = nn.ConvTranspose2d(512, 256, 3, stride=2)
        self.up12 = nn.ConvTranspose2d(256, 128, 3, stride=2)
        self.up13 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.out1 = outconv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up11(x5)
        x = self.up12(x)
        x = self.up13(x)
        x = self.out1(x)
        out = torch.tanh(x)
        return out

class Chartdata(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

    def __len__(self):
        return len(os.listdir(self.img_path))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_path)[idx]
        input_image_path = os.path.join(self.img_path, img_name)
        input_image = torch.tensor(np.array(cv2.imread(input_image_path))).float().permute(2,0,1)

        return input_image

model = UNet(3,3)
print(model) 

batch_size = 20 
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr = lr)





