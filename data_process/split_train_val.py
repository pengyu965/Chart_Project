import os 
import json 
import random

sumit_data = "./Pytorch_Chart_Classification/data/pmc/data.json"

sumit_list = json.load(open(sumit_data, 'r'))
sumit_len = len(sumit_list)

sumit_val = random.sample(sumit_list, int(0.1*sumit_len))

with open("./Pytorch_Chart_Classification/data/pmc/val.json", 'w') as f:
    f.write(json.dumps(sumit_val, indent= 4))

sumit_train = []

for x in sumit_list:
    if x not in sumit_val:
        sumit_train.append(x)

with open("./Pytorch_Chart_Classification/data/pmc/train.json", 'w') as f:
    f.write(json.dumps(sumit_train, indent = 4))