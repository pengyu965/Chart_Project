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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
class Operator:
    def __init__(self, netG, netD = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netG = netG.to(self.device)
        self.netD = netD

    def trainer(self, data_path, batch_size, lr, epoch,):
        self.batch_size = batch_size
        self.lr = lr 
        self.epoch = epoch 
        self.optimizer = optim.Adam(self.netG.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        self.data = Chartdata(data_path = data_path)
        self.dataloader = DataLoader(dataset = self.data, batch_size = self.batch_size, shuffle = True, num_workers = 28)

        global_step = 0
        start_time = time.time()

        idx = len(self.dataloader)

        print_idx = int(self.epoch * idx*1. /100)

        for ep in range(self.epoch):

            if ep == int(self.epoch //3):
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.netG.parameters(), lr = self.lr)
            if ep == int(self.epoch*2//3):
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.netG.parameters(), lr = self.lr)

            for idi, train_batch in enumerate(self.dataloader):
                train_images = train_batch[0].to(self.device)
                train_gt = train_batch[1].to(self.device)

                self.optimizer.zero_grad()
                fake_images = self.netG(train_images)

                loss = self.criterion(fake_images*1. , train_gt*1./255)
                loss.backward()
                self.optimizer.step()


                print("Epoch:[{}]===Step:[{}/{}]===Time:[{:.2f}]===Learning Rate:{}\nTrain_loss:[{:.4f}]]".format(ep, idi, idx, time.time()-start_time, self.lr, loss.item()))

                if (global_step%print_idx) == 0:
                    index = 0
                    nroll = int(self.batch_size**0.5)
                    new_im = Image.new('RGB', (12800,9600))
                    for i in range(0,12800-12800//nroll,12800//nroll):
                        try:
                            for j in range(0, 9600-9600//nroll,9600//nroll):
                                im = Image.fromarray(image_norm(fake_images[index].permute(1,2,0).detach().cpu().clone().numpy()).astype("uint8"))
                                im.thumbnail((1280,960))
                                new_im.paste(im, (i,j))
                                print(index)
                                index += 1
                        except:
                            break
                    
                    new_im.save("./samples/{}.png".format(global_step))

                global_step += 1
                
            try:
                torch.save(self.netG.state_dict(), "./weight/{}.pt".format(ep))
            except:
                pass

    def predictor(self, image_path):
        self.netG.eval()

        if os.path.isdir(image_path):
            print("image_path is a directory")
            idi = 1
            for image in os.listdir(image_path):
                img_tensor = torch.tensor(
                    cv2.imread(os.path.join(image_path, image))
                ).permute(2,0,1).float().unsqueeze(0).to(self.device)
                generated_img_tensor = self.netG(img_tensor)
                generated_img = Image.fromarray(
                    image_norm(
                        generated_img_tensor[0].permute(1,2,0).detach().cpu().clone().numpy()
                    ).astype("uint8")
                )
                generated_img.save("./predict_result/{}.png".format(image))
                print(idi)
                idi += 1
        elif os.path.isfile(image_path):
            print("image_path is a image file")
            img_tensor = torch.tensor(
                    cv2.imread(image_path)
                ).permute(2,0,1).float().unsqueeze(0).to(self.device)
            generated_img_tensor = self.netG(img_tensor)
            generated_img = Image.fromarray(
                image_norm(
                    generated_img_tensor[0].permute(1,2,0).detach().cpu().clone().numpy()
                ).astype("uint8")
            )
            generated_img.save("./predict_result/{}.png".format(image))


def image_norm(arr):
    if len(arr.shape) > 2:
        (x, y, _) = arr.shape
    else:
        (x, y) = arr.shape
    max_v = np.max(arr)
    min_v = np.min(arr)
    new_arr = np.zeros_like(arr)

    for i in range(x):
        for j in range(y):
            new_arr[i,j,:] = (arr[i,j,:] - min_v)*255.0/(max_v-min_v)
    
    return new_arr


class Chartdata(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path+"/bb/"))
    
    def __getitem__(self, idx):
        input_images_path = os.path.join(self.data_path, "images")
        gt_images_path = os.path.join(self.data_path, "bb")

        gt_images_name = os.listdir(gt_images_path)[idx]
        input_images_name = gt_images_name

        gt_images = torch.tensor(cv2.imread(os.path.join(gt_images_path, gt_images_name))).float().permute(2,0,1)
        input_images = torch.tensor(cv2.imread(os.path.join(input_images_path,input_images_name))).float().permute(2,0,1)

        return (input_images, gt_images)

