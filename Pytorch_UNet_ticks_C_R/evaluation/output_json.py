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
from tqdm import tqdm 
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

    # # Visualize each Channel
    # for idi in range(0,6):
    #     image = (arr[:,:,idi]*255).astype(np.uint8)
    #     # cl_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     bbs, centers = get_corner(image)
    #     cv2.imshow("example", image)
    #     cv2.waitKey(0)


    # Text Bounding Box
    for idi in range(0,4):
        image = (arr[:,:,idi]*255).astype(np.uint8)
        # cl_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bbs, centers = get_corner(image)

        for bb in bbs:
            x0 = bb[0]
            y0 = bb[1]
            width = bb[2]
            height = bb[3]
            # Filter out the too small area (noises)
            if width*height > 20:
                # cv2.rectangle(cl_image, (x0,y0), (x0+width, y0+height), (0,0,255),2)
                text_bb = {}
                text_bb["bb"] = {}
                text_bb["bb"]["height"] = height 
                text_bb["bb"]["width"] = width 
                text_bb["bb"]["x0"] = x0 
                text_bb["bb"]["y0"] = y0 

                text_bb["id"] = 0
                text_bb["text"] = "None" 

                o_json["input"]["task2_output"]["text_blocks"].append(text_bb)
    
    # Ticks Points
    o_json["input"]["task4_output"] = {}
    ticks_points = []
    bbs, centers = get_corner((arr[:,:,4]*255).astype(np.uint8))
    for bb in bbs:
        x0 = bb[0]
        y0 = bb[1]
        width = bb[2]
        height = bb[3]
        if width*height > 4:
            cx = x0 + int(width*1./2)
            cy = y0 + int(height*1./2)
            ticks_points.append([cx,cy])
    
    o_json["input"]["task4_output"]["points_list"] = ticks_points

    with open(output_json_path+input_npy[:-3]+"json", 'w') as f:
        f.write(json.dumps(o_json, indent=4))



             
        # cv2.imshow("example", cl_image)
        # cv2.waitKey(0)

        # print(bbs)


# output_json("83978.npy")


pool = multiprocessing.Pool()
for i in tqdm(pool.imap(output_json, os.listdir(input_npy_path)), total = len(os.listdir(input_npy_path))):
    pass




