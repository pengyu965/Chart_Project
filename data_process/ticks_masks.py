import os 
import json 
import cv2
import numpy as np
from tqdm import tqdm 
import multiprocessing

gt_path = "../data/SUMIT/rs_json_gt_sampled/"
img_path = "../data/SUMIT/rs_images_sampled/"

ticks_mask_path = "../data/SUMIT/rs_ticks_masks_sampled/"

if os.path.exists(ticks_mask_path) == False:
    os.mkdir(ticks_mask_path)

def masks_gen(gt_json):
# for gt_json in os.listdir(gt_path):
    gt_file = json.load(open(os.path.join(gt_path, gt_json),'r'))
    img_arr = np.zeros((512,512)).astype(np.uint8)
    img = img_arr


    for axis in gt_file["input"]["task4_output"]["axes"]:
        for item in gt_file["input"]["task4_output"]["axes"][axis]:
            x = item["tick_pt"]["x"]
            y = item["tick_pt"]["y"]
            cv2.circle(img,(x,y), 5, (255,255,255), -1)
            # print([x,y])

    # cv2.imshow("masks", img)
    # cv2.waitKey(0)

    cv2.imwrite(ticks_mask_path+gt_json[:-4]+"png", img)


# masks_gen("3.json")
pool = multiprocessing.Pool()
for i in tqdm(pool.imap(masks_gen, os.listdir(gt_path)), total = len(os.listdir(gt_path))):
    pass
    