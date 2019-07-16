import os 
import json 
import random

chartsense_path = "./data/CHARTSENSE/"

chart_list = []
for file in os.listdir(chartsense_path):
    for image in os.listdir(os.path.join(chartsense_path, file)):
        dic = {} 
        dic["chart"] = "."+os.path.join(chartsense_path,file,image)
        dic["type"] = file 

        chart_list.append(dic)

def Diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 

chart_list = random.sample(chart_list, len(chart_list))
train_chart_list = random.sample(chart_list, int(0.9*len(chart_list)))
val_chart_list = Diff(chart_list, train_chart_list)

with open("./Pytorch_Chart_Classification/data/CHARTSENSE/train.json",'w') as f:
    f.write(json.dumps(train_chart_list, indent=4))

with open("./Pytorch_Chart_Classification/data/CHARTSENSE/val.json",'w') as f:
    f.write(json.dumps(val_chart_list, indent=4))