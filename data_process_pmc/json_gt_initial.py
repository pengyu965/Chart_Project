import os 
import json 
import shutil 
import multiprocessing
from tqdm import tqdm 

# This script is used for counting 
tasks345_images_path = "../data/PMC/tasks345_data/images/"

tasks345_json_gt_path = "../data/PMC/tasks345_data/json_gt/"

role_list = []
axis_list = []
other_count = 0

for file in os.listdir(tasks345_images_path):
    json_f = json.load(open(tasks345_json_gt_path+file[:-4]+".json",'r'))

    task3_dic = json_f["task3"]

    for item in task3_dic["output"]["text_roles"]:
        if item["role"] not in role_list:
            role_list.append(item["role"])
        
        if item["role"] == "other":
            other_count += 1
    
    task4_dic = json_f["task4"]

    for item in task4_dic["output"]["axes"]:
        if item not in axis_list:
            axis_list.append(item)

print(role_list)
print(other_count, len(os.listdir(tasks345_images_path)))
print(axis_list)


new_json_gt_path = "../data/PMC/tasks345_data/json_gt_new/"

if not os.path.exists(new_json_gt_path):
    os.mkdir(new_json_gt_path)

for file in os.listdir(tasks345_images_path):
    json_f = json.load(open(tasks345_json_gt_path+file[:-4]+".json",'r'))
    new_dic = {}
    new_dic["input"] = {}
    new_dic["input"]["task1_output"] = json_f["task3"]["input"]["task1_output"]
    new_dic["input"]["task2_output"] = json_f["task3"]["input"]["task2_output"]
    new_dic["input"]["task3_output"] = json_f["task3"]["output"]
    new_dic["input"]["task4_output"] = json_f["task4"]["output"]
    new_dic["input"]["task4_output"]["axes"].pop('x-tick-type', None)
    new_dic["input"]["task4_output"]["axes"].pop('x2-tick-type', None)
    new_dic["input"]["task4_output"]["axes"].pop('y-tick-type', None)
    new_dic["input"]["task4_output"]["axes"].pop('y2-tick-type', None)
    new_dic["input"]["task5_output"] = json_f["task5"]["output"]

    with open(new_json_gt_path+file[:-4]+".json", 'w') as f:
        f.write(json.dumps(new_dic, indent=4))


