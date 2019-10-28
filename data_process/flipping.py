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

flipped_images_path = "../data/SUMIT/flipped_images_sampled/"
flipped_gts_path = "../data/SUMIT/flipped_json_gt_sampled/"

if not os.path.exists(flipped_images_path):
    os.mkdir(flipped_images_path)
if not os.path.exists(flipped_gts_path):
    os.mkdir(flipped_gts_path)

def img_gt_reader(f_name):
    img_path = images_path + f_name + ".png"
    gt_path = gts_path + f_name + ".json" 
    img = cv2.imread(img_path)
    gt = json.load(open(gt_path, 'r'))
    return img, gt

def gt_flip_func(c_value, flipcode, c_property):
    r'''
    flipcode:
    0: vertical flipping
    1: horizontal flipping
    -1: vertical and horizontal flipping
    '''
    img_h, img_w = img_size[0], img_size[1]

    if flipcode == 0:
        if c_property == "y":
            c_value = - c_value + img_h
        elif c_property == "h":
            c_value = - c_value
        else:
            c_value = c_value

    elif flipcode == 1:
        if c_property == "x":
            c_value = - c_value + img_w
        elif c_property == "w":
            c_value = - c_value
        else:
            c_value = c_value

    elif flipcode == -1:
        if c_property == "x":
            c_value = - c_value + img_w
        elif c_property == "y":
            c_value = - c_value + img_h
        else:
            c_value = - c_value

    return c_value


def flip(img, gt, flipcode):
    r'''
    flipcode:
    0: vertical flipping
    1: horizontal flipping
    -1: vertical and horizontal flipping
    '''
    img = cv2.flip(img, flipcode)
    needed_dic = gt 
    for text_bb in needed_dic["input"]["task2_output"]["text_blocks"]:
        text_bb["bb"]["x0"] = gt_flip_func(text_bb["bb"]["x0"], flipcode, "x")
        text_bb["bb"]["y0"] = gt_flip_func(text_bb["bb"]["y0"], flipcode, "y")
        text_bb["bb"]["width"] = gt_flip_func(text_bb["bb"]["width"], flipcode, "w")
        text_bb["bb"]["height"] = gt_flip_func(text_bb["bb"]["height"], flipcode, "h")

    needed_dic["input"]["task4_output"]["_plot_bb"]["y0"] = gt_flip_func(needed_dic["input"]["task4_output"]["_plot_bb"]["y0"], flipcode, "y")
    needed_dic["input"]["task4_output"]["_plot_bb"]["x0"] = gt_flip_func(needed_dic["input"]["task4_output"]["_plot_bb"]["x0"], flipcode, "x")
    needed_dic["input"]["task4_output"]["_plot_bb"]["height"] = gt_flip_func(needed_dic["input"]["task4_output"]["_plot_bb"]["height"], flipcode, "h")
    needed_dic["input"]["task4_output"]["_plot_bb"]["width"] = gt_flip_func(needed_dic["input"]["task4_output"]["_plot_bb"]["width"], flipcode, "w")

    for axis in needed_dic["input"]["task4_output"]["axes"]:
        for sub_dic in needed_dic["input"]["task4_output"]["axes"][axis]:
            sub_dic["tick_pt"]["x"] = gt_flip_func(sub_dic["tick_pt"]["x"], flipcode, "x")
            sub_dic["tick_pt"]["y"] = gt_flip_func(sub_dic["tick_pt"]["y"], flipcode, "y")

    for lengend_pair in needed_dic["input"]["task5_output"]:
        for text_bb in needed_dic["input"]["task5_output"][lengend_pair]:
            text_bb["bb"]["x0"] = gt_flip_func(text_bb["bb"]["x0"], flipcode, "x")
            text_bb["bb"]["y0"] = gt_flip_func(text_bb["bb"]["y0"], flipcode, "y")
            text_bb["bb"]["width"] = gt_flip_func(text_bb["bb"]["width"], flipcode, "w")
            text_bb["bb"]["height"] = gt_flip_func(text_bb["bb"]["height"], flipcode, "h")
                
    return img, gt



def main(f_name):
    global img_size
    flipcode = [0,1,-1]
    img, gt = img_gt_reader(f_name)
    img_size = img.shape
    
    flip_c = random.sample(flipcode,1)[0]

    flipped_img, flipped_gt = flip(img, gt, flip_c)
    
    cv2.imwrite(flipped_images_path+f_name+".png", flipped_img)
    with open(flipped_gts_path+f_name+".json", 'w') as f:
        f.write(json.dumps(flipped_gt, indent=4))

if __name__ == "__main__":
    # main("3")
    # verify(flipped_images_path, flipped_gts_path, "3")

    with open("../data/SUMIT/sample_list.json",'r') as f:
        sample_list = json.load(f)

    pool = multiprocessing.Pool()
    for i in tqdm(pool.imap(main, sample_list), total = len(sample_list)):
        pass