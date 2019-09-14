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
from unet.unet_parts import *
from util import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up11 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up12 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up13 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up14 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out1 = outconv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up11(x5)
        x = self.up12(x)
        x = self.up13(x)
        x = self.up14(x)
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = nn.DataParallel(UNet(3,3)).to(device)
print(model) 

batch_size = 20 
lr = 0.0001
epoch = 20

optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.MSELoss()

train_data = Chartdata(img_path = "../../data/MIX/train_data/")
dataloader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True, num_workers=12)

global_step = 0
start_time = time.time()

min_loss = 100

idx = len(dataloader)
print_idx = int(epoch *idx * 1./50)

for ep in range(epoch):
    model.train()

    if ep == int(epoch //3):
        lr = lr/10
        optimizer = optim.Adam(model.parameters(), lr = lr)
    if ep == int(epoch*2//3):
        lr = lr/10
        optimizer = optim.Adam(model.parameters(), lr = lr)

    for idi, train_batch in enumerate(dataloader):
        train_images = train_batch.to(device)

        optimizer.zero_grad()

        output_images = model(train_images)

        loss = criterion(output_images, train_images*1./255)
        
        loss.backward
        optimizer.step()
        print("Epoch:[{}]===Step:[{}/{}]===Time:[{:.2f}]===Learning Rate:{}===Regression_Loss:[{:.4f}]".format(ep, idi, idx, time.time()-start_time, lr, loss.item()))
        
        if (global_step%print_idx) == 0:
            index = 0
            nroll = int(batch_size**0.5)
            new_im = Image.new('RGB', (5120,5120))
            for i in range(0,5121-5120//nroll,5120//nroll):
                try:
                    for j in range(0, 5121-5120//nroll,5120//nroll):
                        im = Image.fromarray(image_norm(output_images[index].permute(1,2,0).squeeze(2).detach().cpu().clone().numpy()).astype("uint8"))
                        im.thumbnail((512,512))
                        new_im.paste(im, (i,j))
                        print(index)
                        index += 1
                except:
                    break
            if os.path.exists("./train_samples/") == False:
                os.mkdir("./train_samples/")
                
            new_im.save("./train_samples/{}.png".format(global_step))

        global_step += 1
    
    if loss.item() < min_loss:
        torch.save(model.module.state_dict(), "./weight/model.pt")





