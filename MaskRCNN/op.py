import os 
import random 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import time
import cv2
from PIL import Image
import numpy as np
from utils0 import *
import utils
from torch import autograd
import sys


class Operator:
    def __init__(self, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = nn.DataParallel(model).to(self.device)

    def trainer(self, img_path, gt_path, batch_size, lr, epoch, writer = False):
        self.batch_size = batch_size
        self.lr = lr 
        self.epoch = epoch 
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.SGD(params, lr=self.lr,
                                momentum=0.9, weight_decay=0.0005)
        # self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.writer = writer

        self.train_data = ChartDataset(img_folder = img_path+"/train/", gt_folder = gt_path)
        self.sampler = RandomSampler(self.train_data, True, 3000)

        self.dataloader = DataLoader(dataset = self.train_data, batch_size = self.batch_size, shuffle = False, num_workers = 4, collate_fn=utils.collate_fn, sampler=self.sampler)

        global_step = 0
        start_time = time.time()

        idx = len(self.dataloader)
        print_idx = int(self.epoch * idx*1. /50)
        val_min_loss = 10

        for ep in range(self.epoch):
            self.model.train()


            for idi, (images, targets) in enumerate(self.dataloader):

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # vis_images = images[0].detach().cpu().permute(1,2,0) *255
                # vis_images = transforms.ToPILImage()(images[0].cpu())
                # vis_targets = targets[0]["labels"].cpu()
                # print(vis_targets)
                
                # print(vis_images, vis_targets.shape)

                # cv2.imshow("image", np.array(vis_images))
                # cv2.waitKey()

                # for i in range(vis_targets.shape[0]):
                #     cv2.imshow("masks", np.array(vis_targets[i]*255).astype("uint8"))
                #     cv2.waitKey()


                # print(images[1].shape)
                # print(targets[1]["masks"].shape)
                

                loss_dict = self.model(images, targets)
                # print(loss_dict)

                losses = sum(loss for loss in loss_dict.values())

                print("===Epoch:{}/{}===Step:{}/{}===Time:[{:.2f}]\n===Loss:{:.4f}".format(ep, epoch, idi, idx, time.time()-start_time, losses.item()))

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                # Visulization
                # =============================================
                if (global_step%print_idx) == 0:
                    self.model.eval()

                    new_im = Image.new("RGB", (520*2,520*len(images)))

                    print("Visualizing.....")

                    for i in range(len(images)):
                        vis_images = images[i]
                        r_dic = self.model(vis_images.unsqueeze(0))
                        
                        vis_map, masked_map = vis(vis_images.cpu(), r_dic)

                        im = Image.fromarray(vis_map)
                        masked_im = Image.fromarray(masked_map)
                        new_im.paste(im, (0, i*520))
                        new_im.paste(masked_im, (520, i*520))
                        print(i)

                    if os.path.exists("./train_samples/") == False:
                            os.mkdir("./train_samples/")
                            
                    new_im.save("./train_samples/{}.png".format(global_step))

                    self.model.train()

                # =============================================


                global_step += 1
            
            with torch.no_grad():
                val_loss = self.validator(img_path + "/val", gt_path, global_step)

            if val_loss < val_min_loss:
                val_min_loss = val_loss
                if os.path.exists("./weight/") == False:
                    os.mkdir("./weight/")
                torch.save(self.model.module.state_dict(), "./weight/model.pt")

    def validator(self, val_img_path, val_gt_path, global_step):
        # self.model.eval()
        val_bsize = 2 

        val_data = ChartDataset(img_folder = val_img_path, gt_folder = val_gt_path)
        val_dataloader = DataLoader(dataset=val_data, batch_size=val_bsize,num_workers=4,collate_fn=utils.collate_fn)
        val_idx = len(val_dataloader)

        val_total_loss = 0

        for idi, (val_images, val_targets) in enumerate(val_dataloader):
            val_images = list(val_image.to(self.device) for val_image in val_images)
            val_targets = [{k: v.to(self.device) for k, v in t.items()} for t in val_targets]
            val_loss_dict = self.model(val_images, val_targets)

            val_losses = sum(val_loss for val_loss in val_loss_dict.values())
            val_total_loss += val_losses.item()
        
        
        print("***"*3, "\nValidation Loss:{:.4f}".format(val_total_loss*1./val_idx), "\n"+"***"*3)
        return val_total_loss*1. / val_idx

class ChartDataset:
    def __init__(self, img_folder, gt_folder):
        self.img_folder = img_folder
        self.gt_folder = gt_folder 
        self.transformer = transforms.Compose([
            transforms.ToTensor()
        ]) 

    def __len__(self):
        return len(os.listdir(self.img_folder))
    
    def __getitem__(self, idx):
        img_name = os.listdir(self.img_folder)[idx]
        img_path = os.path.join(self.img_folder, img_name)

        gt_name = img_name[:-4]+".json"
        gt_path = os.path.join(self.gt_folder, gt_name)


        input_image = Image.open(img_path).convert("RGB")
        input_image = np.array(input_image)

        bb, label, mask = bb_label_mask(gt_path, input_image)
        bb = torch.as_tensor(bb, dtype=torch.float32) 
        label = torch.as_tensor(label, dtype=torch.int64)
        mask = torch.as_tensor(mask, dtype = torch.uint8)

        area = (bb[:, 3] - bb[:, 1]) * (bb[:, 2] - bb[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(bb),), dtype=torch.int64)

        target = {}
        target["boxes"] = bb
        target["labels"] = label
        target["masks"] = mask
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # print(target["masks"].max())

        input_image = self.transformer(input_image)
        # input_image = torch.as_tensor(input_image, dtype=torch.float32).permute(2,0,1)

        return (input_image, target)


    
