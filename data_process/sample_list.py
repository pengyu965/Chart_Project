import os 
import json 

sumit_data_path = "../data/SUMIT/"
gt_path = sumit_data_path+"/rs_json_gt_sampled/"

sample_list = []
for file in os.listdir(gt_path):
    sample_list.append(file[:-5])

sample_list.sort(key=int)
with open(sumit_data_path+"sample_list.json", 'w') as f:
    f.write(json.dumps(sample_list, indent=0))

