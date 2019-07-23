from unet import UNet
import torch 
import json 
import os 
import time 
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true',
                    help='Train the model')
parser.add_argument('--eval', action='store_true',
                    help='Evaluate the model')
parser.add_argument('--predict', action='store_true',
                    help='Get prediction result')
parser.add_argument('--finetune', action='store_true',
                    help='Fine tuning the model')
# parser.add_argument('--load', type=int, default=99,
                    # help='Epoch id of pre-trained model')
parser.add_argument("--data", type= str, help='input_data')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate')
parser.add_argument('--epoch', type=int, default = 10, 
                    help = "Max Epoch")
parser.add_argument('--bsize', type=int, default=60,
                    help='Batch size')
parser.add_argument('--keep_prob', type=float, default=0.4,
                    help='Keep probability for dropout')
parser.add_argument('--class_num', type=int, default = 10, 
                    help='class number')
# parser.add_argument('--maxepoch', type=int, default=100,
#                     help='Max number of epochs for training')

# parser.add_argument('--im_name', type=str, default='.png',
#                     help='Part of image name')

cfg = parser.parse_args()


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = UNet(n_channels = 4, n_classes = 1)