import os 
import json 
import cv2
import numpy as np
from tqdm import tqdm 
import multiprocessing

gt_path = "../data/SUMIT/rs_json_gt_sampled/"
img_path = "../data/SUMIT/rs_images_sampled/"

ticks_mask_path = "../data/SUMIT/rs_masks_sampled/"

if os.path.exists(ticks_mask_path) == False:
    os.mkdir(ticks_mask_path)

def masks_gen(gt_json):
# for gt_json in os.listdir(gt_path):
    gt_file = json.load(open(os.path.join(gt_path, gt_json),'r'))

    background_mask = np.ones((512,512)).astype(np.uint8)*255
    chart_title_mask = np.zeros((512,512)).astype(np.uint8)
    lengend_label_mask = np.zeros((512,512)).astype(np.uint8)
    ticks_labels_mask = np.zeros((512,512)).astype(np.uint8)
    axis_titles_mask = np.zeros((512,512)).astype(np.uint8)
    axis_labels_mask = np.zeros((512,512)).astype(np.uint8)
    ticks_mask = np.zeros((512,512)).astype(np.uint8)
    
    

    # ticks_mask
    for axis in gt_file["input"]["task4_output"]["axes"]:
        for item in gt_file["input"]["task4_output"]["axes"][axis]:
            x = item["tick_pt"]["x"]
            y = item["tick_pt"]["y"]
            cv2.circle(ticks_mask,(x,y), 5, (255,255,255), -1)
            cv2.circle(background_mask,(x,y), 5, (0,0,0), -1)
    
    # ticks_labels_mask
    for t_bb in gt_file["input"]["task2_output"]["text_blocks"]:
        t_id = t_bb["id"]
        x0 = t_bb["bb"]["x0"]
        y0 = t_bb["bb"]["y0"]
        x1 = x0 + t_bb["bb"]["width"]
        y1 = y0 + t_bb["bb"]["height"]
        for t_role in gt_file["input"]["task3_output"]["text_roles"]:
            if t_id == t_role["id"]:
                t_type = t_role["role"]
                break
        if t_type == "chart_title":
            cv2.rectangle(chart_title_mask, (x0,y0), (x1,y1), (255,255,255), -1)
            cv2.rectangle(background_mask,(x0,y0), (x1,y1), (0,0,0), -1)
        elif t_type == "axis_title":
            cv2.rectangle(axis_titles_mask, (x0,y0), (x1,y1), (255,255,255), -1)
            cv2.rectangle(background_mask,(x0,y0), (x1,y1), (0,0,0), -1)
        elif t_type == "tick_label":
            cv2.rectangle(ticks_labels_mask, (x0,y0), (x1,y1), (255,255,255), -1)
            cv2.rectangle(background_mask,(x0,y0), (x1,y1), (0,0,0), -1)
        elif t_type == "legend_label":
            cv2.rectangle(lengend_label_mask, (x0,y0), (x1,y1), (255,255,255), -1)
            cv2.rectangle(background_mask,(x0,y0), (x1,y1), (0,0,0), -1)
    
    # cv2.imshow("chart_title", chart_title_mask)
    # cv2.waitKey(0)
    # cv2.imshow("axis_title", axis_titles_mask)
    # cv2.waitKey(0)
    # cv2.imshow("tick_label", ticks_labels_mask)
    # cv2.waitKey(0)
    # cv2.imshow("legend_label", lengend_label_mask)
    # cv2.waitKey(0)
    # cv2.imshow("ticks", ticks_mask)
    # cv2.waitKey(0)
    # cv2.imshow("background", background_mask)
    # cv2.waitKey(0)
    
    final_arr = np.concatenate(
        (
            np.expand_dims(chart_title_mask, axis = 2),
            np.expand_dims(axis_titles_mask, axis = 2),
            np.expand_dims(ticks_labels_mask, axis = 2),
            np.expand_dims(lengend_label_mask, axis = 2),
            np.expand_dims(ticks_mask, axis = 2),
            np.expand_dims(background_mask,axis = 2),
        ),
        axis = 2
        )
    np.save(ticks_mask_path+gt_json[:-5], final_arr)
        
            
            # print([x,y])

    # cv2.imshow("masks", img)
    # cv2.waitKey(0)

    # cv2.imwrite(ticks_mask_path+gt_json[:-4]+"png", img)

# for file in os.listdir(gt_path):
#     masks_gen(file)

pool = multiprocessing.Pool()
for i in tqdm(pool.imap(masks_gen, os.listdir(gt_path)), total = len(os.listdir(gt_path))):
    pass
    