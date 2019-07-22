import os 
import shutil
import random 
from tqdm import tqdm

data_path = "../data/SUMIT/"
img_path = data_path+"/rs_images/"
gt_path = data_path+"/rs_json_gt/"
sampled_img_path = data_path + "/rs_images_sampled/"
sampled_gt_path = data_path + "/rs_json_gt_sampled/"

samples = random.sample(os.listdir(gt_path),8000)

for name in tqdm(samples):
    shutil.copy2(gt_path+name, sampled_gt_path+name)
    shutil.copy2(img_path+name[:-4]+"png", sampled_img_path+name[:-4]+"png")

