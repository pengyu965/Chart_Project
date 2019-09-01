import os 
import cv2 
import json 
import sys 
import random
import multiprocessing
from tqdm import tqdm
from matplotlib import pyplot as plt 
from verify_gt_bb import verify

images_path = "../data/SUMIT/images_sampled/"
gts_path = "../data/SUMIT/json_gt_sampled/"

padded_images_path = "../data/SUMIT/padded_images_sampled/"
padded_gts_path = "../data/SUMIT/padded_json_gt_sampled/"

if os.path.exists(padded_images_path) == False:
    os.mkdir(padded_images_path)
if os.path.exists(padded_gts_path) == False:
    os.mkdir(padded_gts_path)

#             Top 
#       ---------------
#       |    1280     |
#  Left | 960         | Right
#       |             |
#       ---------------
#           Bottom


padding_types = [
    0, # Maintain Original
    0, # Maintain Original
    0, # Maintain Original
    1, # Padding Left
    2, # Padding Right
    3, # Padding Top
    4, # Padding Bottom 
    5, # Padding Left, Top
    6, # Padding Left, Right
    7, # Padding Left, Bottom
    8, # Padding Top, Right
    9, # Padding Top, Bottom
    10 # Padding Right, Bottom
]

width_padding_sizes = [160,320,640] # 1/8, 1/4, 1/2
height_padding_sizes = [120,240,480] # 1/8, 1/4, 1/2

def padding(img, gt_json, top=0, bottom=0, left=0, right=0):
    padded_img = cv2.copyMakeBorder(img, top,bottom,left,right, cv2.BORDER_CONSTANT, value=(255,255,255))

    needed_dic = gt_json["task6"]
    for text_bb in needed_dic["input"]["task2_output"]["text_blocks"]:
        text_bb["bb"]["x0"] = text_bb["bb"]["x0"] + left 
        text_bb["bb"]["y0"] = text_bb["bb"]["y0"] + top 
        
    
    needed_dic["input"]["task4_output"]["_plot_bb"]["y0"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["y0"]+top)
    needed_dic["input"]["task4_output"]["_plot_bb"]["x0"] = round(needed_dic["input"]["task4_output"]["_plot_bb"]["x0"]+left)

    chart_type = needed_dic["input"]["task1_output"]["chart_type"]

    # Revert the reverted x and y axis in horizontal bar & box plot
    if chart_type == "Horizontal box" or chart_type == "Grouped horizontal bar" or chart_type == "Stacked horizontal bar":
        needed_dic["input"]["task4_output"]["axes"]["x1-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("y-axis")
        needed_dic["input"]["task4_output"]["axes"]["y1-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("x-axis")

        needed_dic["input"]["task4_output"]["axes"]["x-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("x1-axis")
        needed_dic["input"]["task4_output"]["axes"]["y-axis"] = needed_dic["input"]["task4_output"]["axes"].pop("y1-axis")


    for axis in needed_dic["input"]["task4_output"]["axes"]:
        for sub_dic in needed_dic["input"]["task4_output"]["axes"][axis]:
            sub_dic["tick_pt"]["x"] = round(sub_dic["tick_pt"]["x"]+left)
            sub_dic["tick_pt"]["y"] = round(sub_dic["tick_pt"]["y"]+top)

    for lengend_pair in needed_dic["input"]["task5_output"]:
        for text_bb in needed_dic["input"]["task5_output"][lengend_pair]:
            text_bb["bb"]["x0"] = text_bb["bb"]["x0"] + left 
            text_bb["bb"]["y0"] = text_bb["bb"]["y0"] + top

    return padded_img, needed_dic


def main(file_name):
    image_path = os.path.join(images_path, file_name+".png")
    gt_path = os.path.join(gts_path, file_name+".json")
    
    img = cv2.imread(image_path)
    gt_json = json.load(open(gt_path,'r'))
    # plt.figure(figsize=(192,96))
    # for padding_type in padding_types:
    padding_type = random.sample(padding_types, 1)[0]
    if padding_type == 0:
        padded_img, needed_dic = padding(img, gt_json, 0,0,0,0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,1,1)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif padding_type == 1:
        padding_size = random.sample(width_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, 0,0,padding_size[0],0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,6)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif  padding_type == 2:
        padding_size = random.sample(width_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, 0,0,0,padding_size[0])
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,7)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif  padding_type == 3:
        padding_size = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, padding_size[0],0,0,0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,8)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif  padding_type == 4:
        padding_size = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, 0,padding_size[0],0,0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,9)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif  padding_type == 5:
        width_padding_size = random.sample(width_padding_sizes,1)
        height_padding_size = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, height_padding_size[0],0,width_padding_size[0],0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,10)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif  padding_type == 6:
        width_padding_size1 = random.sample(width_padding_sizes,1)
        width_padding_size2 = random.sample(width_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, 0,0,width_padding_size1[0],width_padding_size2[0])
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,11)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif padding_type == 7:
        width_padding_size = random.sample(width_padding_sizes,1)
        height_padding_size = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, 0,height_padding_size[0],width_padding_size[0],0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,12)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif padding_type == 8:
        width_padding_size = random.sample(width_padding_sizes,1)
        height_padding_size = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, height_padding_size[0],0,0,width_padding_size[0])
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,13)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif padding_type == 9:
        height_padding_size1 = random.sample(height_padding_sizes,1)
        height_padding_size2 = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, height_padding_size1[0],height_padding_size2[0],0,0)
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,14)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])
    elif padding_type == 10:
        width_padding_size = random.sample(width_padding_sizes,1)
        height_padding_size = random.sample(height_padding_sizes,1)
        padded_img, needed_dic = padding(img, gt_json, 0,height_padding_size[0],0,width_padding_size[0])
        # padded_img = cv2.resize(padded_img,(512,512),interpolation=cv2.INTER_AREA)
        # plt.subplot(3,5,15)
        # plt.imshow(padded_img)
        # plt.xticks([])
        # plt.yticks([])

    # cv2.imshow("padding example", padded_img)
    # cv2.waitKey(0)
    cv2.imwrite(padded_images_path+file_name+".png", padded_img)
    open(padded_gts_path+file_name+".json",'w').write(json.dumps(needed_dic, indent=4))
    # plt.savefig("../presentation/padding_sample.png")
    # plt.show()

# main("122233")
# verify(padded_images_path, padded_gts_path, "122233")
file_name_list = json.load(open("../data/SUMIT/sample_list.json",'r'))
pool = multiprocessing.Pool()
for i in tqdm(pool.imap(main, file_name_list), total = len(file_name_list)):
    pass 








