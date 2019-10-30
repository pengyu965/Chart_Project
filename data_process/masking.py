r'''
This file is masking the tick label randomly.
'''

import os 
import cv2 
import json 
import sys 
import random 
import multiprocessing 
from tqdm import tqdm 
from matplotlib import pyplot as plt
from verify_gt_bb import verify

images_path = "../data/SUMIT/images_sampled/"
gts_path = "../data/SUMIT/json_gt_sampled/"

masked_images_path = "../data/SUMIT/masked_images_sampled/"
masked_gts_path = "../data/SUMIT/masked_json_gt_sampled/"

if not os.path.exists(masked_images_path):
    os.mkdir(masked_images_path)
if not os.path.exists(masked_gts_path):
    os.mkdir(masked_gts_path)

def img_gt_reader(f_name):
    img_path = images_path + f_name + ".png"
    gt_path = gts_path + f_name + ".json" 
    img = cv2.imread(img_path)
    gt = json.load(open(gt_path, 'r'))
    return img, gt

def masking(img, gt, prob = 0.25):
    needed_dic = gt 

    text_id_role = {}

    for text_role in needed_dic["input"]["task3_output"]["text_roles"]:
        text_id_role[text_role["id"]] = text_role["role"]

    for text_bb in needed_dic["input"]["task2_output"]["text_blocks"]:
        text_bb_id = text_bb["id"]
        if text_id_role[text_bb_id] == "tick_label" and random.uniform(0,1) <= prob:
            x0 = text_bb["bb"]["x0"]
            y0 = text_bb["bb"]["y0"]
            x1 = x0 + text_bb["bb"]["width"]
            y1 = y0 + text_bb["bb"]["height"]
            cv2.rectangle(img, (x0,y0), (x1,y1), (255,255,255), -1)
            needed_dic["input"]["task2_output"]["text_blocks"].remove(text_bb)

    return img, needed_dic



def main(f_name):
    img, gt = img_gt_reader(f_name)

    masked_img, masked_gt = masking(img, gt, prob = 0.4)

    cv2.imwrite(masked_images_path+f_name+".png", masked_img)
    with open(masked_gts_path+f_name+".json", 'w') as f:
        f.write(json.dumps(masked_gt, indent=4))


if __name__ == "__main__":
    # main("3")
    # verify(masked_images_path, masked_gts_path, "3")

    with open("../data/SUMIT/sample_list.json",'r') as f:
        sample_list = json.load(f)

    pool = multiprocessing.Pool()
    for i in tqdm(pool.imap(main, sample_list), total = len(sample_list)):
        pass