import os 
import json 
import cv2 
import numpy as np 
from tqdm import tqdm 
import multiprocessing 
import sys

LOW_THRESHOLD = 0.01
HIGH_THRESHOLD = 0.02



def get_score(gts, ress, lt, ht):
    score = 0.
    for gt in gts:
        gt_x = gt[0]
        gt_y = gt[1]
        dis = []
        for res in ress:
            res_x = res[0]
            res_y = res[1]

            dis.append(np.linalg.norm([res_x-gt_x, res_y-gt_y]))
        min_d = np.min(np.array(dis))

        if min_d <= lt:
            score += 1.
        elif min_d >= ht:
            score += 0.
        else:
            score += 1. -((min_d - lt)/(ht-lt))
    
    return score


def task4_eval(result_path, gt_path):
    total_recall = 0.
    total_precision = 0.

    for file in os.listdir(result_path):
        result_js = json.load(open(os.path.join(result_path, file), 'r'))
        gt_js = json.load(open(os.path.join(gt_path, file), 'r'))

        result_points_list = result_js["input"]["task4_output"]["points_list"]

        gt_points_list = []

        for axis in gt_js["input"]["task4_output"]["axes"]:
            for item in gt_js["input"]["task4_output"]["axes"][axis]:
                x = item["tick_pt"]["x"]
                y = item["tick_pt"]["y"]
                gt_points_list.append([x,y])


        h, w, = 512, 512
        diag = ((h ** 2) + (w ** 2)) ** 0.5

        lt, ht = LOW_THRESHOLD * diag, HIGH_THRESHOLD * diag
        
        score = get_score(gt_points_list, result_points_list, lt, ht)
        
        recall = score / len(gt_points_list) if len(gt_points_list) > 0 else 1.

        precision = score / len(result_points_list) if len(result_points_list) > 0 else 1.
        if recall != 1 or precision !=1: 
            print(file, "Recall ===>", recall, "Precision ===>", precision)

        total_recall += recall 
        total_precision += precision
    
    total_recall /= len(os.listdir(result_path))
    total_precision /= len(os.listdir(result_path))

    if total_recall == 0 and total_precision == 0:
        f_measure = 0
    else:
        f_measure = 2 * total_recall * total_precision / (total_recall + total_precision)

    print('Average Recall:', total_recall)
    print('Average Precision:', total_precision)
    print('Average F-Measure:', f_measure)

if __name__ == '__main__':
    try:
        task4_eval(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(e)
        print('Usage Guide: python task4_evaluation.py <result_folder> <ground_truth_folder>')