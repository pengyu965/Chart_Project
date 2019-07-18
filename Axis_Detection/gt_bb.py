import os 
import json
import cv2
import numpy as np 
import shutil
from tqdm import tqdm

data_path = "../data/SUMIT/images/"
json_gt_path = "../data/SUMIT/json_gt/"

color_lib = [
    (255,255,0),
    (255,0,255),
    (0,255,255),
    (135,206,250),
    (255,192,203),
    (191,62,255),
    (255,215,0),
    (255,128,0),
    (100,149,237),
    (0,255,255),
    (202,255,112),
    (255,165,0),
    (250,128,114)
]

for image in tqdm(os.listdir(data_path)):
    image_name, _ = os.path.splitext(image)

    img = cv2.imread(os.path.join(data_path,image))
    json_gt = json.load(open(os.path.join(json_gt_path, image_name+".json"), 'r'))


    task4 = json_gt["task4"]
    try: 
        color_id = 0
        for axis in task4["output"]["axes"]:
            for tick in task4["output"]["axes"][axis]:
                x = tick["tick_pt"]["x"]
                y = tick["tick_pt"]["y"]
                if color_id >= len(color_lib):
                    color_id = 0
                cv2.rectangle(img,(x-10,y-10), (x+10,y+10),color_lib[color_id],2)
                color_id += 1
        cv2.imwrite("../original_data/SUMIT/axes_bb/"+image, img)
        # cv2.imshow("Image Example",img)
        # cv2.waitKey(0)
    except:
        shutil.move(os.path.join(data_path,image), "../original_data/SUMIT/badsamples/images/")
        shutil.move(os.path.join(json_gt_path, image_name+".json"), "../original_data/SUMIT/badsamples/json_gt/")
    # cv2.destroyAllWindows()
    



