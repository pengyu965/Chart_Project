import os
import json 

chartsense_path = "./CHARTSENSE/"
pmc_path = "./PMC/"
sumit_path = "./SUMIT/"

dataset_type_dic = {}

chartsense_types = os.listdir(chartsense_path)

pmc_types = []
pmc_json_path = os.path.join(pmc_path, "annotations", "task1", "output_JSON")

for jsonfile in os.listdir(pmc_json_path):
    
    with open(os.path.join(pmc_json_path, jsonfile), 'r') as f:
        pmc_j_dic = json.load(f)

    pmc_type_str = pmc_j_dic["task1"]["output"]["chart_type"]
    if pmc_type_str not in pmc_types:
        pmc_types.append(pmc_type_str)

sumit_types = []
sumit_json_path = os.path.join(sumit_path, "json_gt")

for jsonfile in os.listdir(sumit_json_path):
    with open(os.path.join(sumit_json_path, jsonfile), 'r') as f:
        sumit_j_dic = json.load(f)

    sumit_type_str = sumit_j_dic["task1"]["output"]["chart_type"]

    if sumit_type_str not in sumit_types:
        sumit_types.append(sumit_type_str)

dataset_type_dic["CHARTSENSE"] = chartsense_types
dataset_type_dic["PMC"] = pmc_types
dataset_type_dic["SUMIT"] = sumit_types

print(json.dumps(dataset_type_dic, indent=4))

with open("./dataset_charttype.json", 'w') as f:
    f.write(json.dumps(dataset_type_dic, indent=4))






