import os 
import json 
import random
import shutil
import multiprocessing
from tqdm import tqdm
import threading 

sumit_image_path = "../data/SUMIT/rs_padded_images_sampled/"
# sumit_image_path = "../data/SUMIT/rs_images_sampled/"

sumit_train_image_path = "../data/SUMIT/train/"
sumit_val_image_path = "../data/SUMIT/val/"
# sumit_test_image_path = "../data/SUMIT/test/"

if not os.path.exists(sumit_train_image_path):
    os.mkdir(sumit_train_image_path)
if not os.path.exists(sumit_val_image_path):
    os.mkdir(sumit_val_image_path)
# if not os.path.exists(sumit_test_image_path):
#     os.mkdir(sumit_test_image_path)

total_len = len(os.listdir(sumit_image_path))
total_list = random.sample(os.listdir(sumit_image_path), total_len)

train_list = total_list[:-int(total_len*1./10)]
val_list = total_list[-int(total_len*1./10):]
# test_list = total_list[-int(total_len*1./10):]

for img_name in tqdm(train_list):
    shutil.move(sumit_image_path+img_name, sumit_train_image_path)
for img_name in tqdm(val_list):
    shutil.move(sumit_image_path+img_name, sumit_val_image_path)
# for img_name in tqdm(test_list):
#     shutil.move(sumit_image_path+img_name, sumit_test_image_path)

# threading.Thread(target=loop1).start()
# threading.Thread(target=loop2).start()
# threading.Thread(target=loop3).start()


shutil.move(sumit_train_image_path, sumit_image_path+"train/")
shutil.move(sumit_val_image_path, sumit_image_path+"val/")
# shutil.move(sumit_test_image_path, sumit_image_path+"test/")


