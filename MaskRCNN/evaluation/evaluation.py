import os 
import json 
import numpy as np 
import cv2
import bb_selector 
import gt_convertor

IOU_THRESHOLD = 0.5

def bbox_iou(boxA, boxB):
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

def result_eval():
    # File name should be *.json

    match_count = 0
    total_iou = 0

    gt_length = 0
    result_length = 0

    file_l = os.listdir("./prediction_dic/")

    for f_name in file_l:

        result_path = "./prediction_dic/" + f_name
        gt_path = "../../data/PMC/tasks345_data/rs_json_gt_new/" + f_name

        result_l = bb_selector.selector(result_path)
        gt_l = gt_convertor.bb_label_score_combine(gt_path)

        # print(f_name)
        # print(result_l)
        # print(gt_l)
        for i in range(len(result_l)):
            for j in range(len(gt_l)):
                if result_l[i][1] == gt_l[j][1]:
                    iou_score = bbox_iou(result_l[i][0], gt_l[j][0])
                    if iou_score > IOU_THRESHOLD:
                        match_count += 1
                        total_iou += iou_score
        
        gt_length += len(gt_l)
        result_length += len(result_l)
    
    iou_r = total_iou / max(gt_length, result_length)
    recall_r = match_count / gt_length
    precision_r = match_count / result_length

    print("iou score:", iou_r)
    print("recall:", recall_r)
    print("precision:", precision_r)



if __name__ == "__main__":
    result_eval()