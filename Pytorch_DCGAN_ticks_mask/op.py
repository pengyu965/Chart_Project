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
        self.netG = nn.DataParallel(netG).to(self.device)
        self.netD = netD

    def trainer(self, img_path, gt_path, batch_size, lr, epoch):
        self.batch_size = batch_size
        self.lr = lr 
        self.epoch = epoch 
        self.optimizer = optim.Adam(self.netG.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.train_data = Chartdata(img_path = img_path+"/train/", gt_path = gt_path)
        self.val_data = Chartdata(img_path = img_path+"/val/", gt_path = gt_path)
        self.test_data = Chartdata(img_path = img_path+"/test/", gt_path = gt_path)
        
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
                loss = self.criterion(fake_images, train_gt)
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
                val_loss = self.validator(self.val_data)

            if val_loss < val_min_loss:
                val_min_loss = val_loss 
                if os.path.exists("./weight/") == False:
                    os.mkdir("./weight/")
                torch.save(self.netG.state_dict(), "./weight/model.pt")
        
        ## Testing
        self.validator(self.test_data)


    def validator(self, val_data):

        self.netG.eval()
        val_dataloader = DataLoader(dataset = val_data, batch_size = 16, shuffle = True, num_workers = 28)
        val_idx = len(val_dataloader)

        val_total_loss = 0

        for idi, val_batch in enumerate(val_dataloader):
            val_images = val_batch[0].to(self.device)
            val_gt = val_batch[1].to(self.device)

            fake_val_images = self.netG(val_images)

            valloss = self.criterion(fake_val_images*1., val_gt*1./255)
            val_total_loss += valloss.item()
        
        print("***"*3, "\nValidation Loss:{:.4f}".format(val_total_loss*1./val_idx), "\n"+"***"*3)
        return val_total_loss*1. / val_idx


    def predictor(self, image_path):
        self.netG.eval()

        if os.path.isdir(image_path):
            print("image_path is a directory")
            idi = 1
            for image in os.listdir(image_path):
                img_tensor = torch.tensor(
                    cv2.imread(os.path.join(image_path, image)),
                    # cv2.resize(
                    #     cv2.imread(os.path.join(image_path, image)),
                    #     (512,512)
                    # )
                ).permute(2,0,1).float().unsqueeze(0).to(self.device)
                generated_img_tensor = self.netG(img_tensor)
                generated_img = Image.fromarray(
                    out_vis(generated_img_tensor.permute(1,2,0).squeeze(2).detach().cpu().clone().numpy())
                    # image_norm(
                    #     generated_img_tensor[0].permute(1,2,0).squeeze(2).detach().cpu().clone().numpy()
                    # ).astype("uint8")
                )
                generated_img.save("./predict_result/{}".format(image))
                print(idi)
                idi += 1
        elif os.path.isfile(image_path):
            image_name,_ = os.path.splitext(os.path.split(image_path)[1])
            print("image_path is a image file")
            img_tensor = torch.tensor(
                cv2.imread(image_path)
                # cv2.resize(
                #     cv2.imread(image_path),
                #     (512,512)
                # )
                ).permute(2,0,1).float().unsqueeze(0).to(self.device)
            generated_img_tensor = self.netG(img_tensor)
            generated_img = Image.fromarray(
                image_norm(
                    generated_img_tensor[0].permute(1,2,0).squeeze(2).detach().cpu().clone().numpy()
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
        gt_npy_path = os.path.join(self.gt_path, img_name[:-3]+"npy")
        input_images = torch.tensor(np.array(cv2.imread(input_images_path))).float().permute(2,0,1)
        # print(np.array(cv2.imread(gt_images_path)).shape)
        gt_images = torch.tensor(np.load(gt_npy_path)).long()
        

        return (input_images, gt_images)


def image_norm(arr):
    if len(arr.shape) > 2:
        (x, y, _) = arr.shape
    else:
        (x, y) = arr.shape
    max_v = np.max(arr)
    min_v = np.min(arr)
    new_arr = np.zeros_like(arr)

    if len(arr.shape) > 2:
        for i in range(x):
            for j in range(y):
                new_arr[i,j,:] = (arr[i,j,:] - min_v)*255.0/(max_v-min_v)
    else:
        for i in range(x):
            for j in range(y):
                new_arr[i,j] = (arr[i,j] - min_v)*255.0/(max_v-min_v)
        
    
    return new_arr

def out_vis(arr):
    color_lib = [
        (255,255,0),
        (255,0,255),
        (0,255,255),
        (135,206,250),
        (255,192,203),
        (0,0,0),
        (191,62,255),
        (255,215,0),
        (255,128,0),
        (100,149,237),
        (0,255,255),
        (202,255,112),
        (255,165,0),
        (250,128,114)
    ]

    x, y, z = arr.shape
    new_arr = np.zeros((x,y,3))

    for i in range(x):
        for j in range(y):
            pixel = arr[i,j,:]
            idi = np.argmax(pixel)
            new_arr[i,j,:] = np.array(color_lib[idi])

    return new_arr
                


