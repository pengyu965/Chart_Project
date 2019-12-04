import os 
import json 
import numpy as np 
import cv2

textrole_label = {
    "axis_title" : 1,
    "tick_label" : 2,
    "legend_label":3,
    "tick_mark" :  4,
    "chart_title": 5
} 

def bb_label_score_combine(file):
    j_file = json.load(open(file, 'r'))
    combined_l = []
    id_role = {}

    for item in j_file["input"]["task3_output"]["text_roles"]:
        id_role[item["id"]] = item["role"]

    for item in j_file["input"]["task2_output"]["text_blocks"]:
        x0 = item["bb"]["x0"]
        y0 = item["bb"]["y0"]
        x1 = x0 + item["bb"]["width"]
        y1 = y0 + item["bb"]["height"]

        if x1 < x0:
            x_inter = x0 
            x0 = x1 
            x1 = x_inter 
        if y1 < y0:
            y_inter = y0 
            y0 = y1 
            y1 = y_inter

        text_id = item["id"]


        text_role = id_role[text_id]

        if text_role not in textrole_label.keys():
            label = 8
        else:
            label = textrole_label[text_role]

        combined_l.append([[x0, y0, x1, y1], label])

    for axis in j_file["input"]["task4_output"]["axes"]:
        for item in j_file["input"]["task4_output"]["axes"][axis]:
            x0 = item["tick_pt"]["x"]-5
            y0 = item["tick_pt"]["y"]-5
            x1 = x0+10
            y1 = y0+10

            label = textrole_label["tick_mark"]
            combined_l.append([[x0, y0, x1, y1], label])

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
    bb_label_score_combine("../../data/PMC/tasks345_data/rs_json_gt_new/PMC1434731___1471-2458-6-50-4.json")


if __name__ == "__main__":
    main()

