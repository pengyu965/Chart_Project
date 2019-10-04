import os
import cv2
from PIL import Image 
from tqdm import tqdm
import multiprocessing 
import json
from verify_gt_bb import verify

# img_path = "../data/SUMIT/padded_images_sampled/"
# gt_path = "../data/SUMIT/padded_json_gt_sampled/"

# rs_img_path = "../data/SUMIT/rs_padded_images_sampled/"
# rs_gt_path = "../data/SUMIT/rs_padded_json_gt_sampled/"

img_path = "../data/SUMIT/images_sampled/"
gt_path = "../data/SUMIT/json_gt_sampled/"

rs_img_path = "../data/SUMIT/rs_images_sampled/"
rs_gt_path = "../data/SUMIT/rs_json_gt_sampled/"

# img_path = "../data/SUMIT/task345_test/images/"
# gt_path = "../data/SUMIT/task345_test/json_gt/"

# rs_img_path = "../data/SUMIT/task345_test/rs_images/"
# rs_gt_path = "../data/SUMIT/task345_test/rs_json_gt/"

if os.path.exists(rs_img_path) == False:
    os.mkdir(rs_img_path)
if os.path.exists(rs_gt_path) == False:
    os.mkdir(rs_gt_path)

def ticks_gt_gen(file_name):
    img = cv2.imread(img_path+file_name+".png")
    h,w,c = img.shape
    rs_w, rs_h = 512,512
    rs_img = cv2.resize(img, (rs_w, rs_h), interpolation=cv2.INTER_AREA)
    gt_dic = json.load(open(gt_path + file_name+".json",'r'))
    # needed_dic = gt_dic["task6"]
    needed_dic = gt_dic
    for text_bb in needed_dic["input"]["task2_output"]["text_blocks"]:
        for item in text_bb["bb"]:
            if item == "height" or item =="y0":
                text_bb["bb"][item] = round(text_bb["bb"][item] * rs_h/h)
            if item == "width" or item =="x0":
                text_bb["bb"][item] = round(text_bb["bb"][item] * rs_w/w)

    #     cv2.rectangle(img, (text_bb["bb"]["x0"], text_bb["bb"]["y0"]), (text_bb["bb"]["x0"]+text_bb["bb"]["width"],text_bb["bb"]["y0"]+text_bb["bb"]["height"]), [0,0,255],2)
    #     cv2.putText(img, str(text_bb["id"]),(text_bb["bb"]["x0"], text_bb["bb"]["y0"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    # cv2.imshow("example",img)
    # cv2.waitKey(0)
    # print(json.dumps(needed_dic, indent=4))
    needed_dic["input"]["task4_output"]["_plot_bb"]["height"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["height"]*rs_h/h)
    needed_dic["input"]["task4_output"]["_plot_bb"]["y0"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["y0"]*rs_h/h)
    needed_dic["input"]["task4_output"]["_plot_bb"]["x0"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["x0"]*rs_w/w)
    needed_dic["input"]["task4_output"]["_plot_bb"]["width"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["width"]*rs_w/w)
    
    # Revert the reverted x and y axis in horizontal bar & box plot
    chart_type = needed_dic["input"]["task1_output"]["chart_type"]

    if chart_type == "Horizontal box" or chart_type == "Grouped horizontal bar" or chart_type == "Stacked horizontal bar":
        needed_dic["input"]["task4_output"]["axes"]["x1-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("y-axis")
        needed_dic["input"]["task4_output"]["axes"]["y1-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("x-axis")

        needed_dic["input"]["task4_output"]["axes"]["x-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("x1-axis")
        needed_dic["input"]["task4_output"]["axes"]["y-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("y1-axis")

    for axis in needed_dic["input"]["task4_output"]["axes"]:
        for sub_dic in needed_dic["input"]["task4_output"]["axes"][axis]:
            sub_dic["tick_pt"]["x"] = round(sub_dic["tick_pt"]["x"]*rs_w/w)
            sub_dic["tick_pt"]["y"] = round(sub_dic["tick_pt"]["y"]*rs_h/h)

    for lengend_pair in needed_dic["input"]["task5_output"]:
        for text_bb in needed_dic["input"]["task5_output"][lengend_pair]:
            for item in text_bb["bb"]:
                if item == "height" or item =="y0":
                    text_bb["bb"][item] = round(text_bb["bb"][item] * rs_h/h)
                if item == "width" or item =="x0":
                    text_bb["bb"][item] = round(text_bb["bb"][item] * rs_w/w)


    cv2.imwrite(rs_img_path+file_name+".png", rs_img)
    with open(rs_gt_path+file_name+".json", 'w') as f:
        f.write(json.dumps(needed_dic, indent=4))

file_name_list = json.load(open("../data/SUMIT/sample_list.json",'r'))

pool = multiprocessing.Pool()
for i in tqdm(pool.imap(ticks_gt_gen, file_name_list), total = len(file_name_list)):
    pass


# for file_name in file_name_list:
#     ticks_gt_gen(file_name)
#     verify(rs_img_path, rs_gt_path, file_name)