import os

import json
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import random
import nltk
from PIL import Image

class chartdata(Dataset):
    def __init__(self, input_path):
        self.data = json.load(open(input_path, 'r'))
        self.label_dic = {
            "Vertical box":0,
            "Donut":1,
            "Stacked vertical bar":2,
            "Scatter":3,
            "Horizontal box":4,
            "Line":5,
            "Grouped vertical bar":6,
            "Pie":7,
            "Grouped horizontal bar":8,
            "Stacked horizontal bar":9
            }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        images_path = self.data[idx]["chart"]
        images = torch.tensor(
            np.array(Image.open(images_path).convert("RGB").resize((256,256), Image.ANTIALIAS))
            ).float().permute(2,0,1)
        
        
        labels = torch.tensor(self.label_dic[self.data[idx]["type"]])
        samples = {"images":images, "labels":labels}
        
        return samples

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# chart_dataset = chartdata(input_path = "./data/sumit_train.json")
dataloader = DataLoader(dataset = trainset, batch_size = 30, shuffle = True, num_workers = 4)
# print(chart_dataset[5])
for idi, (inputs, targets) in enumerate(dataloader):
    print(idi)
    print("inputs", inputs)
    print("targets", targets)

