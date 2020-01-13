import os 
import json 
import cv2 
from tqdm import tqdm 
import multiprocessing 
from verify_gt_bb import verify

imgs_path = "../data/SUMIT/rs_images_sampled/train/"
gts_path = "../data/SUMIT/rs_json_gt_sampled/"
pattcomb_gts_path = "../data/SUMIT/pc_json_gt_sampled/"


def min_max_point(bb_list):
    x_min = bb_list[0][0]
    y_min = bb_list[0][1]
    x_max = bb_list[0][0]
    y_max = bb_list[0][1]

    for item in bb_list:
        if item[0] < x_min:
            x_min = item[0]
        if item[1] < y_min:
            y_min = item[1]
        if item[0] > x_max:
            x_max = item[0]
        if item[1] > y_max:
            y_max = item[1]

    return [x_min, y_min, x_max, y_max]

def drawrectangle(img,p_list):
    cv2.rectangle(img, (p_list[0], p_list[1]), (p_list[2], p_list[3]), (255,0,0), 1)
     

def processing(f_name):
    j_file = json.load(open(os.path.join(gts_path, f_name+".json")))

    id_role_dic = {}
    id_axis_dic = {}

    x_tick_bb_list = []
    y_tick_bb_list = []

    x_tick_label_list = []
    y_tick_label_list = []

    axis_title = []

    chart_title = []

    legend_label = []


    for item in j_file["input"]["task3_output"]["text_roles"]:
        id_role_dic[item["id"]] = item["role"]

    for axis in j_file["input"]["task4_output"]["axes"]:
        for item in j_file["input"]["task4_output"]["axes"][axis]:
            id_axis_dic[item["id"]] = axis
            if axis == "x-axis":
                x_tick_bb_list.append([item["tick_pt"]["x"], item["tick_pt"]["y"]])
            elif axis == "y-axis":
                y_tick_bb_list.append([item["tick_pt"]["x"], item["tick_pt"]["y"]])

    for item in j_file["input"]["task2_output"]["text_blocks"]:
        x0 = item["bb"]["x0"]
        y0 = item["bb"]["y0"]
        x1 = x0 + item["bb"]["width"]
        y1 = y0 + item["bb"]["height"]

        if id_role_dic[item["id"]] == "chart_title":
            chart_title.append([x0, y0, x1, y1])

        elif id_role_dic[item["id"]] == "axis_title":
            axis_title.append([x0, y0, x1, y1])

        elif id_role_dic[item["id"]] == "tick_label":
            if id_axis_dic[item["id"]] == "x-axis":
                x_tick_label_list.append([x0,y0])
                x_tick_label_list.append([x1,y1])
            else:
                y_tick_label_list.append([x0,y0])
                y_tick_label_list.append([x1,y1])

        elif id_role_dic[item["id"]] == "legend_label":
            legend_label.append([x0,y0])
            legend_label.append([x1,y1])

    plot_area = [
        j_file["input"]["task4_output"]["_plot_bb"]["x0"],
        j_file["input"]["task4_output"]["_plot_bb"]["y0"],
        j_file["input"]["task4_output"]["_plot_bb"]["x0"] + j_file["input"]["task4_output"]["_plot_bb"]["width"],
        j_file["input"]["task4_output"]["_plot_bb"]["y0"] + j_file["input"]["task4_output"]["_plot_bb"]["height"],
    ]

    x_axis_area = min_max_point(x_tick_bb_list + x_tick_label_list)
    y_axis_area = min_max_point(y_tick_bb_list + y_tick_label_list)

    chart_title = chart_title
    axis_title = axis_title

    legend_area = min_max_point(legend_label)
    print(plot_area)

    img = cv2.imread(os.path.join(imgs_path, f_name+".png"))
    print(os.path.join(imgs_path, f_name+".png"))
    drawrectangle(img, x_axis_area)
    drawrectangle(img, y_axis_area)
    drawrectangle(img, plot_area)
    cv2.imshow("example", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    processing("3")    






    