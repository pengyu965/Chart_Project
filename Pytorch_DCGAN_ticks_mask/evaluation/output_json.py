r"""
The input .npy has 6 channels per pixel, which means there are 6 classes. 
The classes from top channel to the bottom channel [0:5] are:
(
    [0] Chart Title
    [1] Axis Title
    [2] Ticks Label
    [3] Lengend_Label
    [4] Ticks_Masks
    [5] Background
)

"""

import os 
import json 
import cv2 
import numpy as np 
import tqdm 
import multiprocessing 
from contour_point import get_corner

input_npy_path = "./predict_result/"
output_json_path = "./output_json/"
if os.path.exists(output_json_path) == False:
    os.mkdir(output_json_path)


def output_json(input_npy):
    arr = np.load(os.path.join(input_npy_path, input_npy))
    x, y, z = arr.shape 
    o_json = {}
    o_json["input"] = {}

    o_json["input"]["task2_output"] = {}
    o_json["input"]["task2_output"]["text_blocks"] = []

    for idi in range(0,6):
        print(np.max(arr[:,:,idi]))
        image = (arr[:,:,idi]*255).astype(np.uint8)
        bbs, centers = get_corner(image)
        cv2.imshow("example", image)
        cv2.waitKey(0)

        print(bbs)


output_json("10665.npy")




