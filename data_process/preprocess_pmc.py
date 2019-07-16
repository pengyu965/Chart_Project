import os 
import json

pmc_path = "./data/PMC/"

pmc_ann_json = os.path.join(pmc_path, "annotations", "task1", "random_name_inverse_index.json")

with open(pmc_ann_json, 'r') as f:
    dic = json.load(f)

typelist = list(dic.values())

chartdata = []

for sample in typelist:
    chartdic = {}
    chartdic["chart"] = "./PMC/images/task1/"+sample[1]
    chartdic["type"] = sample[0]
    chartdata.append(chartdic)

with open("./Pytorch_Chart_Classification/data/pmc_data.json", 'w') as f:
    f.write(json.dumps(chartdata, indent = 4))