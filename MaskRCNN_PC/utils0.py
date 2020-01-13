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

    id_role_dic = {}
    id_axis_dic = {}

    x_tick_bb_list = []
    y_tick_bb_list = []

    x_tick_label_list = []
    y_tick_label_list = []

    axis_title = []

    chart_title = []

    legend_label = []


    for item in j_file["input"]["task3_output"]["text_roles"]:
        id_role_dic[item["id"]] = item["role"]

    for axis in j_file["input"]["task4_output"]["axes"]:
        for item in j_file["input"]["task4_output"]["axes"][axis]:
            id_axis_dic[item["id"]] = axis
            if axis == "x-axis":
                x_tick_bb_list.append([item["tick_pt"]["x"], item["tick_pt"]["y"]])
            elif axis == "y-axis":
                y_tick_bb_list.append([item["tick_pt"]["x"], item["tick_pt"]["y"]])

    for item in j_file["input"]["task2_output"]["text_blocks"]:
        x0 = item["bb"]["x0"]
        y0 = item["bb"]["y0"]
        x1 = x0 + item["bb"]["width"]
        y1 = y0 + item["bb"]["height"]

        if id_role_dic[item["id"]] == "chart_title":
            if x1 < x0:
                x_inter = x0 
                x0 = x1 
                x1 = x_inter 
            if y1 < y0:
                y_inter = y0 
                y0 = y1 
                y1 = y_inter
            chart_title = [x0, y0, x1, y1]

        elif id_role_dic[item["id"]] == "axis_title":
            if x1 < x0:
                x_inter = x0 
                x0 = x1 
                x1 = x_inter 
            if y1 < y0:
                y_inter = y0 
                y0 = y1 
                y1 = y_inter
            axis_title.append([x0, y0, x1, y1])

        elif id_role_dic[item["id"]] == "tick_label":
            if id_axis_dic[item["id"]] == "x-axis":
                x_tick_label_list.append([x0,y0])
                x_tick_label_list.append([x1,y1])
            else:
                y_tick_label_list.append([x0,y0])
                y_tick_label_list.append([x1,y1])

        elif id_role_dic[item["id"]] == "legend_label":
            legend_label.append([x0,y0])
            legend_label.append([x1,y1])

    x0 = j_file["input"]["task4_output"]["_plot_bb"]["x0"]
    y0 = j_file["input"]["task4_output"]["_plot_bb"]["y0"]
    x1 = j_file["input"]["task4_output"]["_plot_bb"]["x0"] + j_file["input"]["task4_output"]["_plot_bb"]["width"]
    y1 = j_file["input"]["task4_output"]["_plot_bb"]["y0"] + j_file["input"]["task4_output"]["_plot_bb"]["height"]
    if x1 < x0:
        x_inter = x0 
        x0 = x1 
        x1 = x_inter 
    if y1 < y0:
        y_inter = y0 
        y0 = y1 
        y1 = y_inter

    plot_area = [x0,y0,x1,y1]


    x_axis_area = min_max_point(x_tick_bb_list + x_tick_label_list)
    y_axis_area = min_max_point(y_tick_bb_list + y_tick_label_list)

    # print(x_axis_area, y_axis_area)

    chart_title = chart_title
    axis_title = axis_title
    
    

    bb_list.append(plot_area)
    bb_list.append(x_axis_area)
    bb_list.append(y_axis_area)
    if chart_title != []:
        bb_list.append(chart_title)
    if legend_label != []:
        legend_area = min_max_point(legend_label)
        bb_list.append(legend_area)
    else:
        legend_area = []
    if axis_title != []:
        bb_list.append(axis_title[0])
        bb_list.append(axis_title[1])

    chart_title_label = [3]*len(chart_title)
    legend_area_label = [4]*len(legend_area)
    axis_title_label = [5]*len(axis_title)


    label_list = [0,1,2] + chart_title + legend_area_label + axis_title_label

    # print(bb_list)

    for bb in bb_list:
        # print(bb)
        x0, y0, x1, y1 = bb 
        # print(x0, y0, x1, y1)
        mask = np.zeros((h,w), dtype = np.uint8)
        cv2.rectangle(mask, (x0,y0), (x1,y1), (1), -1)
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
        


    