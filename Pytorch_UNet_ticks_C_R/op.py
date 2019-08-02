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
from util import *

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
        self.netG = nn.DataParallel(netG).to(self.device)
        self.netD = netD

    def trainer(self, img_path, gt_path, batch_size, lr, epoch):
        self.batch_size = batch_size
        self.lr = lr 
        self.epoch = epoch 
        self.optimizer = optim.Adam(self.netG.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_r = Vector_Regression_Loss()

        self.train_data = Chartdata(img_path = img_path+"/train/", gt_path = gt_path)

        
        # indices = torch.randperm(len(self.data)).tolist()
        # self.train_data = torch.utils.data.Subset(self.data, indices[:-int(len(self.data)*2./10)])
        # self.val_data = torch.utils.data.Subset(self.data, indices[-int(len(self.data)*2./10):-int(len(self.data)*1./10)])
        # self.test_data = torch.utils.data.Subset(self.data, indices[-int(len(self.data)*1./10):])

        self.dataloader = DataLoader(dataset = self.train_data, batch_size = self.batch_size, shuffle = True, num_workers = 28)

        global_step = 0
        start_time = time.time()

        idx = len(self.dataloader)

        print_idx = int(self.epoch * idx*1. /50)

        val_min_loss = 10


        for ep in range(self.epoch):

            self.netG.train()

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

                # print(np.max(fake_images.detach().cpu().clone().numpy()),np.min(fake_images.detach().cpu().clone().numpy()))
                # print(np.max(train_gt.detach().cpu().clone().numpy()), np.min(train_gt.detach().cpu().clone().numpy()))
                # loss = self.criterion(fake_images*1. , train_gt*1./255)
                # print(fake_images.shape, train_gt.shape)
                loss_c = self.criterion(fake_images[:,:6,:,:], train_gt[:,:,:,0].long())
                loss_r = self.criterion_r(fake_images, train_gt)
                loss = loss_c + loss_r
                loss.backward()
                self.optimizer.step()


                print("Epoch:[{}]===Step:[{}/{}]===Time:[{:.2f}]===Learning Rate:{}\nTrain_loss:[{:.4f}]]".format(ep, idi, idx, time.time()-start_time, self.lr, loss.item()))

                if (global_step%print_idx) == 0:
                    index = 0
                    nroll = int(self.batch_size**0.5)
                    new_im = Image.new('RGB', (5120,5120))
                    for i in range(0,5121-5120//nroll,5120//nroll):
                        try:
                            for j in range(0, 5121-5120//nroll,5120//nroll):
                                # im = Image.fromarray(image_norm(fake_images[index].permute(1,2,0).squeeze(2).detach().cpu().clone().numpy()).astype("uint8"))
                                im = Image.fromarray(out_vis(fake_images[index].permute(1,2,0).detach().cpu().clone().numpy()).astype("uint8"))
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

            ## Validation
            with torch.no_grad():
                val_loss = self.validator(img_path+"/val/", gt_path, global_step)

            if val_loss < val_min_loss:
                val_min_loss = val_loss 
                if os.path.exists("./weight/") == False:
                    os.mkdir("./weight/")
                torch.save(self.netG.module.state_dict(), "./weight/model.pt")

        torch.save(self.netG.module.state_dict(), "./weight/final.pt")
        
        ## Testing
        with torch.no_grad():
            self.validator(img_path+"/test/", gt_path, global_step)


    def validator(self, val_img_path, val_gt_path, global_step):
        self.netG.eval()
        self.criterion = nn.CrossEntropyLoss()
        val_bsize = 16

        val_data = Chartdata(img_path = val_img_path, gt_path = val_gt_path)
        val_dataloader = DataLoader(dataset = val_data, batch_size = val_bsize, shuffle = True, num_workers = 28)
        val_idx = len(val_dataloader)

        val_total_loss = 0

        for idi, val_batch in enumerate(val_dataloader):
            val_images = val_batch[0].to(self.device)
            val_gt = val_batch[1].to(self.device)

            fake_val_images = self.netG(val_images)

            valloss_c = self.criterion(fake_val_images[:,:6,:,:], val_gt[:,:,:,0].long())
            valloss_r = self.criterion_r(fake_val_images, val_gt)
            valloss = valloss_c.item() + valloss_r.item()
            val_total_loss += valloss

        index = 0
        nroll = int(val_bsize**0.5)
        new_im = Image.new('RGB', (5120,5120))
        for i in range(0,5121-5120//nroll,5120//nroll):
            try:
                for j in range(0, 5121-5120//nroll,5120//nroll):
                    # im = Image.fromarray(image_norm(fake_images[index].permute(1,2,0).squeeze(2).detach().cpu().clone().numpy()).astype("uint8"))
                    im = Image.fromarray(out_vis(fake_val_images[index].permute(1,2,0).detach().cpu().clone().numpy()).astype("uint8"))
                    im.thumbnail((512,512))
                    new_im.paste(im, (i,j))
                    index += 1
            except:
                break
        if os.path.exists("./val_samples/") == False:
            os.mkdir("./val_samples/")
            
        new_im.save("./val_samples/{}.png".format(global_step))
        
        print("***"*3, "\nValidation Loss:{:.4f}".format(val_total_loss*1./val_idx), "\n"+"***"*3)
        return val_total_loss*1. / val_idx


    def predictor(self, image_path, visualize=True):
        self.netG.eval()

        if os.path.isdir(image_path):
            print("image_path is a directory")
            idi = 1
            for image in os.listdir(image_path):
                img_tensor = torch.tensor(
                    # cv2.imread(os.path.join(image_path, image)),
                    cv2.resize(
                        cv2.imread(os.path.join(image_path, image)),
                        (512,512),
                        interpolation = cv2.INTER_AREA
                    )
                ).float().permute(2,0,1).unsqueeze(0).to(self.device)
                generated_img_tensor = self.netG(img_tensor)
                if visualize == True:
                    generated_img = Image.fromarray(
                        out_vis(generated_img_tensor[0].permute(1,2,0).detach().cpu().clone().numpy()).astype(np.uint8)
                        # image_norm(
                        #     generated_img_tensor[0].permute(1,2,0).squeeze(2).detach().cpu().clone().numpy()
                        # ).astype("uint8")
                    )
                    generated_img.thumbnail((512,512))
                    generated_img.save("./predict_result/{}".format(image))
                else:
                    generated_arr = channel_binarization(
                        generated_img_tensor[0].permute(1,2,0).detach().cpu().clone().numpy()
                    )
                    np.save(
                        "./predict_result/{}".format(image[:-4]), 
                        generated_arr
                        )
                print(idi)
                idi += 1
        elif os.path.isfile(image_path):
            image_name,_ = os.path.splitext(os.path.split(image_path)[1])
            print("image_path is a image file")
            img_tensor = torch.tensor(
                # cv2.imread(image_path)
                cv2.resize(
                    cv2.imread(image_path),
                    (512,512),
                    interpolation = cv2.INTER_AREA
                )
                ).permute(2,0,1).float().unsqueeze(0).to(self.device)
            generated_img_tensor = self.netG(img_tensor)
            generated_img = Image.fromarray(
                out_vis(
                    generated_img_tensor[0].permute(1,2,0).detach().cpu().clone().numpy()
                ).astype("uint8")
            )
            generated_img.save("./predict_result/{}.png".format(image_name))


class Chartdata(Dataset):
    def __init__(self, img_path, gt_path):
        self.img_path = img_path
        self.gt_path = gt_path
    def __len__(self):
        return len(os.listdir(self.img_path))
    
    def __getitem__(self, idx):
        img_name = os.listdir(self.img_path)[idx]
        input_images_path = os.path.join(self.img_path, img_name)
        ## This line map code should be modified to a good manner in future
        # line_map_path = os.path.join("../../data/SUMIT/rs_linemap_sampled/", img_name)
        ### 
        gt_npy_path = os.path.join(self.gt_path, img_name[:-3]+"npy")
        input_images = torch.tensor(np.array(cv2.imread(input_images_path))).float().permute(2,0,1)

        # line_maps = torch.tensor(cv2.imread(line_map_path)[:,:,0]).float().unsqueeze(0)
        # input_images = torch.cat((input_images, line_maps), dim=0)

        # print(np.array(cv2.imread(gt_images_path)).shape)
        gt_images = torch.tensor(np.load(gt_npy_path))
        

        return (input_images, gt_images)



def Vector_Regression_Loss(result, gt):
    b, c, h, w = result.shape
    overlap_area = 0
    total_area = 0
    loss = 0
    for k in range(b):
        for i in range(h):
            for j in range(w):
                _class = torch.argmax(result[k,:6,i,j])
                if _class == 2:
                    if gt[k,i,j,0] == 2:
                        overlap_area += 1
                        gt_vector = [gt[k,i,j,3]-i, gt[k,i,j,4]-j]
                        rs_vector = result[k,6:,i,j]


                        loss += torch.sqrt((rs_vector[0]-gt_vector[0])**2 + (rs_vector[1]-gt_vector[1])**2)

                elif _class == 4:
                    if gt[k,i,j,0] == 4:
                        overlap_area += 1
                        gt_vector = [gt[k,i,j,3]-i, gt[k,i,j,4]-j]
                        rs_vector = result[k,6:,i,j]
                        
                        loss += torch.sqrt((rs_vector[0]-gt_vector[0])**2 + (rs_vector[1]-gt_vector[1])**2)
                else:
                    loss += torch.sum(result[k,6:,i,j]-result[k,6:,i,j])
                    

    
    loss /= overlap_area
    return loss.float()


