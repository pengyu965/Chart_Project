import os 
import shutil 

cmp_path = "../../data/SUMIT/rs_images_sampled/test/"

src_gt_path = "../../data/SUMIT/rs_json_gt_sampled/"

des_path = "./test_gt/"

for img in os.listdir(cmp_path):
    name, _ = os.path.splitext(img)
    gt_name = name+".json"
    shutil.copy2(os.path.join(src_gt_path, gt_name), des_path)
