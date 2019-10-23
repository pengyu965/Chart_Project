import os 
import shutil 
import multiprocessing
from tqdm import tqdm 

# This script is used for build a new directory under PMC/ which named as tasks345_data

images_path = "../data/PMC/images/tasks345/"

json_gt_path = "../data/PMC/annotations/tasks345/output_JSON/"

tasks345_data_path = "../data/PMC/tasks345_data/"

tasks345_images_path = "../data/PMC/tasks345_data/images/"

tasks345_json_gt_path = "../data/PMC/tasks345_data/json_gt/"

if not os.path.exists(tasks345_data_path):
    os.mkdir(tasks345_data_path)
if not os.path.exists(tasks345_images_path):
    os.mkdir(tasks345_images_path)
if not os.path.exists(tasks345_json_gt_path):
    os.mkdir(tasks345_json_gt_path)

file_list = []

for file in os.listdir(images_path):
    file_list.append(file[:-4])

def mover(file_name):
    shutil.copy2(images_path+file_name+".jpg", tasks345_images_path)
    shutil.copy2(json_gt_path+file_name+".json", tasks345_json_gt_path)

pool = multiprocessing.Pool()

for i in tqdm(pool.imap(mover, file_list), total = len(file_list)):
    pass