import os 
import json
import cv2
import numpy as np 
import shutil
from tqdm import tqdm
# import torch 
# from torch.utils.data import Dataset, DataLoader

data_path = "../data/SUMIT/images/"
json_gt_path = "../data/SUMIT/json_gt/"

color_lib = [
    (255,255,0),
    (255,0,255),
    (0,255,255),
    (135,206,250),
    (255,192,203),
    (191,62,255),
    (255,215,0),
    (255,128,0),
    (100,149,237),
    (0,255,255),
    (202,255,112),
    (255,165,0),
    (250,128,114)
]

for gt_file in tqdm(os.listdir(json_gt_path)):
    gt_file_path = os.path.join(json_gt_path, gt_file)
    gt_dic = 
