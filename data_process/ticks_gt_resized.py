import os
import cv2
from PIL import Image 
from tqdm import tqdm
import multiprocessing 
import json
from verify_gt_bb import verify

img_path = "../data/SUMIT/rs_images_sampled/"
gt_path = "../data/SUMIT/json_gt_sampled/"
rs_gt_path = "../data/SUMIT/rs_json_gt_sampled/"

def ticks_gt_gen(gt_file):
    img = cv2.imread(img_path+gt_file[:-4]+"png")
    gt_dic = json.load(open(gt_path + gt_file,'r'))
    needed_dic = gt_dic["task6"]
    for text_bb in needed_dic["input"]["task2_output"]["text_blocks"]:
        for item in text_bb["bb"]:
            if item == "height" or item =="y0":
                text_bb["bb"][item] = round(text_bb["bb"][item] * 512/960)
            if item == "width" or item =="x0":
                text_bb["bb"][item] = round(text_bb["bb"][item] * 512/1280)

    #     cv2.rectangle(img, (text_bb["bb"]["x0"], text_bb["bb"]["y0"]), (text_bb["bb"]["x0"]+text_bb["bb"]["width"],text_bb["bb"]["y0"]+text_bb["bb"]["height"]), [0,0,255],2)
    #     cv2.putText(img, str(text_bb["id"]),(text_bb["bb"]["x0"], text_bb["bb"]["y0"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    # cv2.imshow("example",img)
    # cv2.waitKey(0)
    # print(json.dumps(needed_dic, indent=4))
    needed_dic["input"]["task4_output"]["_plot_bb"]["height"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["height"]*512/960)
    needed_dic["input"]["task4_output"]["_plot_bb"]["y0"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["y0"]*512/960)
    needed_dic["input"]["task4_output"]["_plot_bb"]["x0"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["x0"]*512/1280)
    needed_dic["input"]["task4_output"]["_plot_bb"]["width"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["width"]*512/1280)
    for axis in needed_dic["input"]["task4_output"]["axes"]:
        for sub_dic in needed_dic["input"]["task4_output"]["axes"][axis]:
            sub_dic["tick_pt"]["x"] = round(sub_dic["tick_pt"]["x"]*512/1280)
            sub_dic["tick_pt"]["y"] = round(sub_dic["tick_pt"]["y"]*512/960)

    for lengend_pair in needed_dic["input"]["task5_output"]:
        for text_bb in needed_dic["input"]["task5_output"][lengend_pair]:
            for item in text_bb["bb"]:
                if item == "height" or item =="y0":
                    text_bb["bb"][item] = round(text_bb["bb"][item] * 512/960)
                if item == "width" or item =="x0":
                    text_bb["bb"][item] = round(text_bb["bb"][item] * 512/1280)



    with open(rs_gt_path+gt_file, 'w') as f:
        f.write(json.dumps(needed_dic, indent=4))

pool = multiprocessing.Pool()
for i in tqdm(pool.imap(ticks_gt_gen, os.listdir(gt_path)), total = len(os.listdir(gt_path))):
    pass



# ticks_gt_gen("2.json")


# verify(img_path, rs_gt_path, "160700")