import os 
import cv2 
import json 
import sys 
import random

def verify(images_path, gts_path, file_name):
    gt_dic = json.load(open(gts_path+file_name+".json",'r'))
    img =cv2.imread(images_path+file_name+".png")

    for text_bb in gt_dic["input"]["task2_output"]["text_blocks"]:
        x0 = text_bb["bb"]["x0"]
        y0 = text_bb["bb"]["y0"]
        width = text_bb["bb"]["width"]
        height = text_bb["bb"]["height"]
        drawrectangle(img, x0,y0,width,height)

    for axis in gt_dic["input"]["task4_output"]["axes"]:
        for item in gt_dic["input"]["task4_output"]["axes"][axis]:
            x0 = item["tick_pt"]["x"]
            y0 = item["tick_pt"]["y"]
            drawrectangle(img, x0-5,y0-5,10,10)
    
    for text_bb in gt_dic["input"]["task5_output"]["legend_pairs"]:
        x0 = text_bb["bb"]["x0"]
        y0 = text_bb["bb"]["y0"]
        width = text_bb["bb"]["width"]
        height = text_bb["bb"]["height"]
        drawrectangle(img, x0,y0,width,height)

    cv2.imshow("gt_bb", img)
    cv2.waitKey(0)

def drawrectangle(img,x0,y0,width,height):
    return cv2.rectangle(img, (x0,y0),(x0+width,y0+height), (0,0,255),1)