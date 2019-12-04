import os 
import json 
import numpy as np 
import cv2


def bb_iou(boxA, boxB):
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

def bb_label_score_combine(bb, labels, scores):
    combined_l = []
    for i in range(len(bb)):
        combined_l.append([bb[i], labels[i], scores[i]])
    return combined_l 

def selector(file):
    j_file = json.load(open(file, 'r'))
    boxes_l = j_file["boxes"]
    labels_l = j_file["labels"]
    scores_l = j_file["scores"]

    combined_l = bb_label_score_combine(boxes_l, labels_l, scores_l)

    i = 0 
    while i < len(combined_l):
        item_1 = combined_l[i]
        j = i+1 
        while j < len(combined_l):
            item_2 = combined_l[j]
            iou = bb_iou(item_1[0], item_2[0])
            if iou > 0.3:
                if item_1[2] < item_1[2]:
                    combined_l.remove(item_1)
                    break 
                else:
                    combined_l.remove(item_2)
            
            j += 1
        i += 1
    
    # print(combined_l)
    # f_name, _ = os.path.splitext(os.path.split(file)[1])
    # img = cv2.imread("../../data/PMC/tasks345_data/rs_images/"+f_name+".png")
    # print(img.shape)
    # for i in range(len(combined_l)):
    #     x0, y0, x1, y1 = combined_l[i][0]
    #     cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0),2)

    # cv2.imshow("verify", img)
    # cv2.waitKey()

    return combined_l

def main():
    selector("./prediction_dic/PMC1434731___1471-2458-6-50-4.json")

if __name__ == "__main__":
    main()


