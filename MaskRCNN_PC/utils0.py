import os 
import cv2 
import json 
import numpy as np
import torch
from torchvision import transforms

role_label = {
    "axis_area" : 1,
    "plot_area":2,
    "legend_area" : 3,
    "chart_title": 4,
    "axis_title": 5
} 

def min_max_point(bb_list):
    x_min = bb_list[0][0]
    y_min = bb_list[0][1]
    x_max = bb_list[0][0]
    y_max = bb_list[0][1]

    for item in bb_list:
        if item[0] < x_min:
            x_min = item[0]
        if item[1] < y_min:
            y_min = item[1]
        if item[0] > x_max:
            x_max = item[0]
        if item[1] > y_max:
            y_max = item[1]

    return [x_min, y_min, x_max, y_max]

def bb_label_mask(j_path, img):
    j_file = json.load(open(j_path, 'r'))
    h, w, c = img.shape

    bb_list = []
    label_list = []
    mask_list = []

    for item in j_file:
        try:
            label_id = role_label[item[0]]
        except KeyError:
            print(j_path)
        label_list.append(label_id)

        bb = item[1]
        bb_list.append(bb)

        mask = np.zeros((h,w), dtype = np.uint8)
        cv2.rectangle(mask, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])),(1),-1)
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
        1: "axis_area",
        2: "plot_area",
        3: "legend_area",
        4: "chart_title",
        5: "axis_title"
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
        if scores[i] > 0.3:
            label = labels[i]
            color = color_lib[label]
            class_txt = class_id[label]
            cv2.rectangle(img, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), color, 1)
            cv2.putText(img, str(round(scores[i],2))+","+class_txt, (boxes[i][0],boxes[i][1]), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1, cv2.LINE_AA)
            mask = masks[i][0]
            masked_img = apply_mask(masked_img, mask, color)

    return img, masked_img
    




# bb_label_mask("../../data/SUMIT/rs_json_gt_sampled/3.json", cv2.imread("../../data/SUMIT/rs_images_sampled/train/3.png"))
        


    