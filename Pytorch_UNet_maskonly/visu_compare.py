import numpy
import numpy  as np
import cv2
import json 

file_name = "198007"

result_js = json.load(open("./output_json/{}.json".format(file_name), 'r'))
gt_js = json.load(open("./test_gt/{}.json".format(file_name), 'r'))
img = cv2.imread("../../data/SUMIT/rs_images_sampled/test/{}.png".format(file_name))


gt_points_list = []

for axis in gt_js["input"]["task4_output"]["axes"]:
    for item in gt_js["input"]["task4_output"]["axes"][axis]:
        x = item["tick_pt"]["x"]
        y = item["tick_pt"]["y"]
        gt_points_list.append([x,y])

result_points = result_js["input"]["task4_output"]["points_list"]

for point in gt_points_list:
    cv2.circle(img, tuple(point), 3, (0,0,255),2)

for point in result_points:
    cv2.circle(img, tuple(point), 5, (0,255,0),2)

cv2.imwrite("img_gt_result.png", img)

