import os 
import json 

gt_path = "../data/SUMIT/rs_json_gt_sampled/"

img_path = "../data/SUMIT/rs_images_sampled/"

gt_list = os.listdir(gt_path)
img_list = os.listdir(img_path)
print(len(gt_list), len(img_list))

for file in gt_list:
    img = file[:-4]+"png"
    if img not in img_list:
        print(img)

for img in img_list:
    gt = img[:-3]+"json"
    if gt not in gt_list:
        print(gt)