import os 
import shutil 
import json
import multiprocessing
from tqdm import tqdm

images_path = "../data/SUMIT/images/"
gt_path = "../data/SUMIT/json_gt/"

images_sampled_path = "../data/SUMIT/images_sampled/"
gt_sampled_path = "../data/SUMIT/json_gt_sampled/"

if os.path.exists(images_sampled_path) == False:
    os.mkdir(images_sampled_path)
if os.path.exists(gt_sampled_path) == False:
    os.mkdir(gt_sampled_path)

def sample_mv(name):
    shutil.copy2(images_path+name+".png", images_sampled_path)
    j_file = json.load(open(gt_path + name + ".json", 'r'))
    needed_dic = j_file["task6"]
    with open(gt_sampled_path + name + ".json", 'w') as f:
        f.write(json.dumps(needed_dic, indent=4))
    # shutil.copy2(gt_path+name+".json", gt_sampled_path)

with open("../data/SUMIT/sample_list.json",'r') as f:
    sample_list = json.load(f)

pool = multiprocessing.Pool()
for i in tqdm(pool.imap(sample_mv, sample_list), total = len(sample_list)):
    pass
