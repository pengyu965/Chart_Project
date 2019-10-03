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
from coordinates import get_bbox
import sys

input_npy_path = "./predict_result/"
output_json_path = "./output_json/"

# json_gt_path = "../../data/SUMIT/rs_padded_json_gt_sampled/"
json_gt_path = "../../data/SUMIT/rs_json_gt_sampled/"
if os.path.exists(output_json_path) == False:
    os.mkdir(output_json_path)

IOU_THRESHOLD = 0.3

role_class_dic ={
    0:"chart_title",
    1:"axis_title",
    2:"tick_label",
    3:"legend_label"
}

def IoU_Score(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def remove_duplicates(List):
    arr = np.array(List)
    if arr.shape != (0,):
        new_arr = np.unique(arr, axis = 0)
    else:
        new_arr = arr
    new_List = new_arr.tolist()
    
    return new_List

def most_frequent_number(List): 
    no_dup_list = remove_duplicates(List)
    max_num = 0
    for element in no_dup_list:
        num = List.count(element)
        if num > max_num:
            max_num = num 
            return_element = element
    return return_element, max_num


def output_json(input_npy):
    arr = np.load(os.path.join(input_npy_path, input_npy))

    json_gt = json.load(open(json_gt_path+input_npy[:-3]+"json",'r'))
    gt_task_2 = json_gt["input"]["task2_output"]["text_blocks"]

        # x0 = item["bb"]["x0"]
        # y0 = item["bb"]["y0"]
        # x1 = x0 + item["bb"]["width"]
        # y1 = y0 + item["bb"]["height"]


    x, y, z = arr.shape 
    o_json = {}
    o_json["input"] = {}

    o_json["input"]["task2_output"] = {}
    o_json["input"]["task2_output"]["text_blocks"] = []
    o_json["input"]["task3_output"] = {}
    o_json["input"]["task3_output"]["text_roles"] = []


    # Text Bounding Box
    for idi in range(0,4):
        image = (arr[:,:,idi]*255).astype(np.uint8)

        bbs, centers = get_bbox(image)

        for bb in bbs:
            x0 = bb[0]
            y0 = bb[1]
            width = bb[2]
            height = bb[3]

            res_bb = [x0, y0, x0+width, y0+height]
            # Filter out the too small area (noises)
            if width*height > 20:
                text_bb = {}
                text_bb["bb"] = {}
                text_bb["bb"]["height"] = height 
                text_bb["bb"]["width"] = width 
                text_bb["bb"]["x0"] = x0 
                text_bb["bb"]["y0"] = y0 

                text_bb["id"] = "None"
                text_bb["text"] = "None" 
                text_bb["role"] = role_class_dic[idi]

                for item in gt_task_2:
                    gt_x0 = item["bb"]["x0"]
                    gt_y0 = item["bb"]["y0"]
                    gt_x1 = gt_x0 + item["bb"]["width"]
                    gt_y1 = gt_y0 + item["bb"]["height"]
                    gt_id = item["id"]
                    gt_text = item["text"]
                    gt_bb = [gt_x0,gt_y0,gt_x1,gt_y1]
                    iou_score = IoU_Score(res_bb, gt_bb)

                    # print(item, res_bb, gt_bb, iou_score)
                    
                    if iou_score > IOU_THRESHOLD:
                        text_bb["id"] = gt_id
                        text_bb["text"] = gt_text

                        text_role = {}
                        text_role["id"] = gt_id
                        text_role["role"] = role_class_dic[idi]
                        
                        o_json["input"]["task3_output"]["text_roles"].append(text_role)




                o_json["input"]["task2_output"]["text_blocks"].append(text_bb)
    
    # Ticks Points
    o_json["input"]["task4_output"] = {}
    o_json["input"]["task4_output"]["axes"] = {}
    o_json["input"]["task4_output"]["axes"]["x-axis"] = []
    o_json["input"]["task4_output"]["axes"]["y-axis"] = []
    # tick_points = []
    


    # Internal offset voting
    tick_bbs, tick_centers = get_bbox((arr[:,:,4]*255).astype(np.uint8))

    points_list_internal = []

    for tick_bb in tick_bbs:
        tick_x0 = tick_bb[0]
        tick_y0 = tick_bb[1]
        tick_x1 = tick_x0 + tick_bb[2]
        tick_y1 = tick_y0 +tick_bb[3]

        internal_voted_tick_points = []
        for i in range(tick_x0, tick_x1):
            for j in range(tick_y0, tick_y1):
                if arr[j,i,4] != 0:
                    vector_x = arr[j,i,6]
                    vector_y = arr[j,i,7]
                    tick_point = [int(i+vector_x),int(j+vector_y)]
                    internal_voted_tick_points.append(tick_point)
        
        maximum_internal_point, internal_maximum_vote = most_frequent_number(internal_voted_tick_points)
        points_list_internal.append(maximum_internal_point)
        # cv2.circle(imgg, tuple(maximum_internal_point), 2, (255), -1)


    all_final_tick = []
    for item in o_json["input"]["task2_output"]["text_blocks"]:
        label_x0 = item["bb"]["x0"]
        label_y0 = item["bb"]["y0"]
        label_x1 = label_x0 + item["bb"]["width"]
        label_y1 = label_y0 + item["bb"]["height"]

        label_center_x  = label_x0 + int(item["bb"]["width"]*1./2)
        label_center_y  = label_y0 + int(item["bb"]["height"]*1./2)

        external_voted_tick_points = []
        if item["role"] == "tick_label":
            for i in range(label_x0, label_x1):
                for j in range(label_y0, label_y1):
                    vector_x = arr[j,i,6]
                    vector_y = arr[j,i,7]
                    tick_point = [int(i+vector_x),int(j+vector_y)]
                    external_voted_tick_points.append(tick_point)

            maximum_external_point, external_maximum_vote = most_frequent_number(external_voted_tick_points)
            tick_dic = {}
            for point in points_list_internal:
                dis = np.linalg.norm([maximum_external_point[0]-point[0], maximum_external_point[1]-point[1]])
                if dis < 10:
                    final_coord = (np.array(maximum_external_point) + np.array(point))*1/2
                    # final_coord = final_coord.astype(np.int)
                    tick_dic["id"] = item["id"]
                    tick_dic["tick_pt"] = {}
                    tick_dic["tick_pt"]["x"] = int(final_coord[0])
                    tick_dic["tick_pt"]["y"] = int(final_coord[1])

                    # if item["id"] == 10:
                    #     print(point[0]-label_center_x, point[1]-label_center_y)

                    # Process X, Y
                    if abs(point[0]-label_center_x) <= abs(point[1]-label_center_y):
                        o_json["input"]["task4_output"]["axes"]["x-axis"].append(tick_dic)
                    else:
                        o_json["input"]["task4_output"]["axes"]["y-axis"].append(tick_dic)


    #         cv2.circle(imgg, tuple(maximum_external_point), 2, (255), -1)

    

    # cv2.imshow("examplle", imgg)
    # cv2.waitKey(0)


    with open(output_json_path+input_npy[:-3]+"json", 'w') as f:
        f.write(json.dumps(o_json, indent=4))



             
        # cv2.imshow("example", cl_image)
        # cv2.waitKey(0)

        # print(bbs)


# output_json("190701.npy")

pool = multiprocessing.Pool()
for i in tqdm(pool.imap(output_json, os.listdir(input_npy_path)), total = len(os.listdir(input_npy_path))):
    pass




