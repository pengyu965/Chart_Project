r'''
For MaskRCNN_PC, for combined area, segmentation and detection
'''

import os 
import json 
import cv2 
from tqdm import tqdm 
import multiprocessing 
from verify_gt_bb import verify

imgs_path = "../data/SUMIT/rs_images_sampled/train/"
gts_path = "../data/SUMIT/rs_json_gt_sampled/"
pattcomb_gts_path = "../data/SUMIT/pc_json_gt_sampled/"

if not os.path.exists(pattcomb_gts_path):
    os.mkdir(pattcomb_gts_path)

def min_max_point(bb_list):
    if bb_list == []:
        return []
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

def bb_area(bb):
    return (bb[2] - bb[0])*(bb[3]-bb[1])   

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

        if x1 < x0:
            x_inter = x0 
            x0 = x1 
            x1 = x_inter 
        if y1 < y0:
            y_inter = y0 
            y0 = y1 
            y1 = y_inter

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

    x0 = j_file["input"]["task4_output"]["_plot_bb"]["x0"]
    y0 = j_file["input"]["task4_output"]["_plot_bb"]["y0"]
    x1 = j_file["input"]["task4_output"]["_plot_bb"]["x0"] + j_file["input"]["task4_output"]["_plot_bb"]["width"]
    y1 = j_file["input"]["task4_output"]["_plot_bb"]["y0"] + j_file["input"]["task4_output"]["_plot_bb"]["height"]
    if x1 < x0:
        x_inter = x0 
        x0 = x1 
        x1 = x_inter 
    if y1 < y0:
        y_inter = y0 
        y0 = y1 
        y1 = y_inter

    plot_area = [x0,y0,x1,y1]

    x_axis_area = min_max_point(x_tick_bb_list + x_tick_label_list)
    y_axis_area = min_max_point(y_tick_bb_list + y_tick_label_list)

    if bb_area(x_axis_area) == 0:
        x_axis_area = [x_axis_area[0]-5, x_axis_area[1]-5, x_axis_area[2]+5, x_axis_area[3]+5]
    if bb_area(y_axis_area) == 0:
        y_axis_area = [y_axis_area[0]-5, y_axis_area[1]-5, y_axis_area[2]+5, y_axis_area[3]+5]


    chart_title = chart_title
    axis_title = axis_title

    legend_area = min_max_point(legend_label)


    # print(x_axis_area) # min max [x,x,x,x]
    # print(y_axis_area) # min max [x,x,x,x]
    # print(plot_area) # [x,x,x,x]
    # print(legend_area) # min max [x,x,x,x]
    # print(chart_title) # [[],[]]
    # print(axis_title) # [[],[]]
    final_list = []
    if x_axis_area != []:
        final_list.append(["axis_area", x_axis_area])
    if y_axis_area != []:
        final_list.append(["axis_area", y_axis_area])
    if plot_area != []:
        final_list.append(["plot_area", plot_area])
    if legend_area != []:
        final_list.append(["legend_area", legend_area])

    for item in chart_title:
        if item != []:
            final_list.append(["chart_title", item])
    for item in axis_title:
        if item != []:
            final_list.append(["axis_title", item])

    # print(final_list)

    with open(pattcomb_gts_path + f_name + ".json", 'w') as f:
        f.write(json.dumps(final_list, indent= 4))



    # img = cv2.imread(os.path.join(imgs_path, f_name+".png"))

    # for item in final_list:
    #     drawrectangle(img, item[1])
    # # drawrectangle(img, x_axis_area)
    # # drawrectangle(img, y_axis_area)
    # # drawrectangle(img, plot_area)
    # # drawrectangle(img, legend_area)
    # # drawrectangle(img, chart_title)
    # # drawrectangle(img, axis_title)
    # cv2.imshow("example", img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    # processing("3")    

    pool = multiprocessing.Pool()
    file_name_list = json.load(open("../data/SUMIT/sample_list.json",'r'))
    for i in tqdm(pool.imap(processing, file_name_list), total= len(file_name_list)):
        pass






    