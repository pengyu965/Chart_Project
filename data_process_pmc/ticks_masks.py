r"""
Generate the ground truth masks 
The classes from top channel to the bottom channel [0:5] are:
(
    [0] Chart Title
    [1] Axis Title
    [2] Ticks Label
    [3] Lengend_Label
    [4] Ticks_Masks
    [5] Background
    [6] Ticks Label ID
    [7] Ticks Marks ID
    [8] vector Center
)
And the final output would be Three channels
(
    [0] The masks Class layer [value of each pixel from 0-5], representing the pixel class, one channel
    [X] Ticks Label ID Layer, one channel, [X] represents we don't have this layer for now
    [X] Ticks ID Layer, one channel, [X] represents we don't have this layer for now
    [2] vector Layer, two channel
)
"""

import os 
import json 
import cv2
import numpy as np
from tqdm import tqdm 
import multiprocessing
import sys 


gt_path = "../data/PMC/tasks345_data/rs_json_gt_new/"
img_path = "../data/PMC/tasks345_data/rs_images/"

ticks_mask_path = "../data/PMC/tasks345_data/rs_masks/"

# gt_path = "../data/SUMIT/rs_json_gt_sampled/"
# img_path = "../data/SUMIT/rs_images_sampled/"

# ticks_mask_path = "../data/SUMIT/rs_masks_sampled/"

if os.path.exists(ticks_mask_path) == False:
    os.mkdir(ticks_mask_path)

def masks_gen(gt_json):
    img_size = (512,512)
# for gt_json in os.listdir(gt_path):
    gt_file = json.load(open(os.path.join(gt_path, gt_json),'r'))

    chart_title_mask = np.zeros(img_size).astype(np.uint8)
    axis_titles_mask = np.zeros(img_size).astype(np.uint8)
    ticks_labels_mask = np.zeros(img_size).astype(np.uint8)
    ## Legend_titles is not used
    legend_titles_mask = np.zeros(img_size).astype(np.uint8)
    ##
    lengend_label_mask = np.zeros(img_size).astype(np.uint8)
    ticks_mask = np.zeros(img_size).astype(np.uint8)
    background_mask = np.ones(img_size).astype(np.uint8)*255

    # ticks_labels_ID_mask = np.full(img_size, np.nan)
    # ticks_ID_mask = np.full(img_size, np.nan)

    vector_center_masks = np.zeros(img_size+(2,))
    vector_masks = np.zeros(img_size+(2,))

    # print(gt_json)    
    
    tick_id_center = {}
    # ticks_mask
    for axis in gt_file["input"]["task4_output"]["axes"]:
        for item in gt_file["input"]["task4_output"]["axes"][axis]:
            x = item["tick_pt"]["x"]
            y = item["tick_pt"]["y"]
            t_id = item["id"]
            cv2.circle(ticks_mask,(x,y), 5, (255,255,255), -1)
            cv2.circle(background_mask,(x,y), 5, (0,0,0), -1)
            # Ticks ID Layer
            # cv2.circle(ticks_ID_mask, (x,y), 5, (t_id), -1)
            # Ticks area center point
            cv2.circle(vector_center_masks, (x,y), 5, (y,x), -1)
            tick_id_center[str(t_id)] = (y,x)

    
    # All kinds of labels' mask
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
            cv2.rectangle(background_mask, (x0,y0), (x1,y1), (0,0,0), -1)
            # Tick Label ID layer
            # cv2.rectangle(ticks_labels_ID_mask, (x0,y0), (x1,y1), (t_id), -1)
            # Tick Label area ---> tick point center. Need to fullfill the assumption that 
            # all ticks labels would must have an associated ticks marks (though not all ticks marks have associated labels)
            # Then all t_id would be one key in tick_id_center.
            cv2.rectangle(vector_center_masks, (x0,y0), (x1,y1), tick_id_center[str(t_id)],-1)

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
    class_arr = np.zeros((512,512))
    x, y, z = final_arr.shape
    for i in range(x):
        for j in range(y):
            idi = np.argmax(final_arr[i,j,:])
            class_arr[i,j] = idi

    # Process the vector layer
    a = np.zeros((512,512,3)).astype(np.uint8)
    
    for i in range(x):
        for j in range(y):
            # The array and image axis are fliped, (x,y) ---> (y,x)
            if vector_center_masks[i,j,0] != 0 or vector_center_masks[i,j,1] != 0:

                # cv2.circle(a, (j,i), 5,(255,255,255), -1)

                vector = vector_center_masks[i,j,:] - np.array([i,j])
                vector_masks[i,j,:] = vector

                # ## Normalization
                # ## Need to process the center, where vector = 0,0
                # if vector[0] == 0 and vector[1] == 0:
                #     vector_masks[j,i,:] = vector
                # else:
                #     vector_masks[j,i,:] = vector/((vector[0]**2+vector[1]**2)**0.5)

    #             cv2.circle(a, (int(j+vector[1]),int(i+vector[0])), 0, (255,255,255), -1)
    
    # cv2.imshow("example", a)
    # cv2.waitKey(0)

    class_arr = np.expand_dims(class_arr, axis = 2)
    # ticks_labels_ID_mask = np.expand_dims(ticks_labels_ID_mask, axis = 2)
    # ticks_ID_mask = np.expand_dims(ticks_ID_mask, axis = 2)
    final_masks = np.concatenate((class_arr, vector_masks), axis = 2)
    np.save(ticks_mask_path+gt_json[:-5], final_masks)
    # print(final_masks.shape)
        


# for file in os.listdir(gt_path):
#     masks_gen(file)

# masks_gen("162996.json")

pool = multiprocessing.Pool()
for i in tqdm(pool.imap(masks_gen, os.listdir(gt_path)), total = len(os.listdir(gt_path))):
    pass
    