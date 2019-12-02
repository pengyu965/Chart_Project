import os 
import cv2 
import json 
import numpy as np
import torch
from torchvision import transforms

textrole_label = {
    "axis_title" : 1,
    "tick_label" : 2,
    "legend_label":3,
    "tick_mark" :  4,
    "chart_title": 5
} 

def bb_label_mask(j_path, img):
    j_file = json.load(open(j_path, 'r'))
    h, w, c = img.shape
    threshold = 3

    bb_list = []
    label_list = []
    mask_list = []

    id_role = {}

    for item in j_file["input"]["task3_output"]["text_roles"]:
        id_role[item["id"]] = item["role"]


    for item in j_file["input"]["task2_output"]["text_blocks"]:
        x0 = item["bb"]["x0"]
        y0 = item["bb"]["y0"]
        x1 = x0 + item["bb"]["width"]
        y1 = y0 + item["bb"]["height"]

        if x1 < x0:
            x_inter = x0 
            x0 = x1 
            x1 = x_inter 
        if y1 < y0:
            y_inter = y0 
            y0 = y1 
            y1 = y_inter

        text_id = item["id"]


        text_role = id_role[text_id]

        if text_role not in textrole_label.keys():
            continue

        label_list.append(textrole_label[text_role])
        bb_list.append([x0,y0,x1,y1]) 

        mask = np.zeros((h,w), dtype = np.uint8)
        cv2.rectangle(mask, (x0,y0), (x1,y1),(1),-1)
        mask_list.append(list(mask))

    for axis in j_file["input"]["task4_output"]["axes"]:
        for item in j_file["input"]["task4_output"]["axes"][axis]:
            x0 = item["tick_pt"]["x"]-5
            y0 = item["tick_pt"]["y"]-5
            x1 = x0+10
            y1 = y0+10

            bb_list.append([x0,y0,x1,y1])
            label_list.append(textrole_label["tick_mark"])
            # label_list.append(1)

            mask = np.zeros((h,w), dtype = np.uint8)
            cv2.rectangle(mask, (x0,y0), (x1,y1),(1),-1)
            mask_list.append(list(mask))
            

    
    return bb_list, label_list, mask_list

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask >= 0.5,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def vis(img, r_dic):
    color_lib = [
        (255,255,255),
        (255,0,255),
        (0,255,255),
        (135,206,250),
        (255,192,203),
        (255,255,0),
        (191,62,255),
        (255,215,0),
        (255,128,0),
        (100,149,237),
        (0,255,255),
        (202,255,112),
        (255,165,0),
        (250,128,114)
    ]
    class_id = {
        0: "background",
        1: "axis_title",
        2: "tick_label",
        3: "legend_label",
        4: "tick_mark",
        5: "chart_title"
    }
    img = transforms.ToPILImage()(img)
    img = np.array(img)
    r_dic = r_dic[0]

    boxes = r_dic["boxes"].detach().cpu().numpy()
    labels = r_dic["labels"].detach().cpu().numpy()
    scores = r_dic["scores"].detach().cpu().numpy()
    masks = r_dic["masks"].detach().cpu()

    masked_img = np.copy(img)

    for i in range(boxes.shape[0]):
        if scores[i] > -0.1:
            label = labels[i]
            color = color_lib[label]
            class_txt = class_id[label]
            cv2.rectangle(img, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), color, 1)
            cv2.putText(img, str(round(scores[i],2))+","+class_txt, (boxes[i][0],boxes[i][1]), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1, cv2.LINE_AA)
            mask = masks[i][0]
            masked_img = apply_mask(masked_img, mask, color)

    return img, masked_img
    




# bb_label_mask("../../data/SUMIT/rs_json_gt_sampled/3.json", cv2.imread("../../data/SUMIT/rs_images_sampled/train/3.png"))
        


    