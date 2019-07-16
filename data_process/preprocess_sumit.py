import os 
import json 

sumit_path = "./data/sumit/"
data_list = []

for file in os.listdir(os.path.join(sumit_path, "json_gt")):
    file_name, _ = os.path.splitext(file)

    class_dic = {}
    with open(os.path.join(sumit_path, "json_gt", file),'r') as f:
        j_file = json.load(f)

    class_dic["chart"] = os.path.join(sumit_path, "images", file_name+".png")
    class_dic["type"] = j_file["task1"]["output"]["chart_type"]

    data_list.append(class_dic)

with open("./data/sumit_data.json", 'w') as f:
    f.write(json.dumps(data_list, indent=4))