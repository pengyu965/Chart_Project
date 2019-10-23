import os 
import json 
import random 

sumit_data_path = "../data/SUMIT/"
gt_path = sumit_data_path+"/json_gt/"

test_file_list = json.load(open(sumit_data_path+"task345_test/sample_list.json",'r'))

file_list = os.listdir(gt_path)

ini_sampled_list = random.sample(file_list, 12000)

sample_list = []

for file in ini_sampled_list:
    if len(sample_list) >= 10000:
        break
    if file[:-5] not in test_file_list:
        sample_list.append(file[:-5])

print(len(sample_list))
sample_list.sort(key=int)
with open(sumit_data_path+"sample_list.json", 'w') as f:
    f.write(json.dumps(sample_list, indent=0))

