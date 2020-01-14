import os 
import cv2 
import json 

gts_path = "../data/SUMIT/rs_json_gt_sampled/"
pattcomb_gts_path = "../data/SUMIT/pc_json_gt_sampled/"

file_name_list = json.load(open("../data/SUMIT/sample_list.json",'r'))

def area(bb):
    return (bb[2] - bb[0])*(bb[3]-bb[1])

def examine(f_name):
    j_file = json.load(open(pattcomb_gts_path + f_name + ".json", 'r'))
    for item in j_file:
        if item[0] == "i":
            print(f_name)
        if area(item[1]) == 0:
            print(f_name)
            print(item)


for f_name in file_name_list:
    examine(f_name)
             
