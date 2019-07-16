import torch
import torch.nn as nn 
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader


import time 
import json 
import sys

import numpy as np
import time
import random
import nltk
from PIL import Image

class Operator:
    def __init__(self, model, batch_size, lr, epoch, class_num):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device_2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        
        self.model = model
        # self.model = torch.nn.DataParallel(model.to(self.device))
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.class_num = class_num
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # self.train_data = Chartdata(input_path = self.input_dir+"/sumit_train.json")
        # self.train_dataloader = DataLoader(dataset = self.train_data, batch_size = self.batch_size, shuffle = True, num_workers = 28)
        # self.val_data = Chartdata(input_path = self.input_dir+"/sumit_val.json")
        # self.val_dataloader = DataLoader(dataset = self.val_data, batch_size = self.batch_size, shuffle = False, num_workers = 28)
        

        self.writer = SummaryWriter()

    def trainer(self, train_data, val_data = None):

        self.train_dataloader = DataLoader(dataset = train_data, batch_size = self.batch_size, shuffle = True, num_workers = 28)

        global_step = 0
        start_time = time.time()

        best_val_acc = 0
        best_global_step = 0

        idx = len(self.train_dataloader)

        for ep in range(self.epoch):

            if ep == int(self.epoch //3):
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
            if ep == int(self.epoch*2//3):
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

            for idi, train_batch in enumerate(self.train_dataloader):


                images = train_batch["images"].to(self.device)
                labels = train_batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                output_logits = self.model(images)

                try:
                    if self.model.aux_logits:
                        _loss = [] 
                        for i in range(len(output_logits)):
                            _loss.append(self.criterion(output_logits[i], labels))
                        loss = sum(_loss)
                        preds = torch.argmax(output_logits[0], dim = -1)
                    else:
                        loss = self.criterion(output_logits, labels)
                        preds = torch.argmax(output_logits, dim = -1)
                except:
                    loss = self.criterion(output_logits, labels)
                    preds = torch.argmax(output_logits, dim = -1)

                acc = torch.sum(preds == labels).float()/self.batch_size

                loss.backward()
                self.optimizer.step()
                print("Epoch:[{}]===Step:[{}/{}]===Time:[{:.2f}]===Learning Rate:{}\nTrain_loss:[{:.4f}], Train_acc[{:.4f}]".format(ep, idi, idx, time.time()-start_time, self.lr, loss.item(), acc))

                self.writer.add_scalar("training_accuracy", acc, global_step)
                self.writer.add_scalar("training_loss", loss.item(), global_step)

                if val_data:
                    if global_step % 500 == 0:
                        cur_val_acc = self.validator(val_data = val_data, global_step = global_step)
                        if cur_val_acc >= best_val_acc:
                            best_val_acc = cur_val_acc
                            best_global_step = global_step
                            best_model_state = self.model.state_dict()

                        self.model.train()
                else: 
                    best_model_state = self.model.state_dict()



                global_step += 1
        
        torch.save(best_model_state,"./weight/GoogLeNet/weight.pt")
        print(best_global_step, global_step, best_val_acc)


    
    def validator(self, val_data, global_step = None, s_writer = True, verbose = False):
        self.val_dataloader = DataLoader(dataset = val_data, batch_size = self.batch_size, shuffle = True, num_workers = 28)

        val_acc = 0
        val_loss = 0

        val_idx = len(self.val_dataloader)

        self.model.eval()
        for val_idi, val_batch in enumerate(self.val_dataloader):
            val_images = val_batch["images"].to(self.device)
            val_labels = val_batch["labels"].to(self.device)


            val_output_logits = self.model(val_images)
            # print(val_output_logits)
            # print(val_labels)

            # try:
            #     if self.model.aux_logits:
            #         print(self.model.aux_logits)
            #         loss = self.criterion(val_output_logits[0], val_labels)
            #         preds = torch.argmax(val_output_logits[0], dim = -1)
            # except:
            #     print("except")
            loss = self.criterion(val_output_logits, val_labels)
            preds = torch.argmax(val_output_logits, dim = -1)

            cur_acc = torch.sum(preds == val_labels).float()/self.batch_size
            cur_loss = loss.item()

            val_acc += cur_acc
            val_loss += cur_loss 
            
            if verbose:
                print("Step:[{}/{}]===Validation Loss:[{:.4f}]===Validation Acc:[{:.4f}]".format(val_idi, val_idx, cur_loss, cur_acc ))
        
        if s_writer:
            self.writer.add_scalar("val_accuracy", val_acc/val_idx, global_step)
            self.writer.add_scalar("val_loss", val_loss/val_idx, global_step)


        print("\n===\nAvg Val Loss: {:.4f}, Avg Val Acc: {:.4f}\n===\n".format(val_loss/val_idx, val_acc.item()/val_idx))
        return val_acc.item()/val_idx


class Chartdata(Dataset):
    def __init__(self, input_path):
        self.data = json.load(open(input_path, 'r'))
        # self.label_dic = {
        #     "Vertical box":0,
        #     "Donut":1,
        #     "Stacked vertical bar":2,
        #     "Scatter":3,
        #     "Horizontal box":4,
        #     "Line":5,
        #     "Grouped vertical bar":6,
        #     "Pie":7,
        #     "Grouped horizontal bar":8,
        #     "Stacked horizontal bar":9
        #     }
        self.label_dic = {
            "Line":0,
            "Vertical bar":1,
            "Scatter":2,
            "Vertical box":3,
            "Horizontal bar":4,
            "Pie":5,
            "Horizontal box":6
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        images_path = self.data[idx]["chart"]

        images = torch.tensor(
            np.array(Image.open(images_path).convert("RGB").resize((64,64), Image.ANTIALIAS))
            ).float().permute(2,0,1)
        
        labels = torch.tensor(self.label_dic[self.data[idx]["type"]])

        samples = {"images":images, "labels":labels}
        
        return samples
