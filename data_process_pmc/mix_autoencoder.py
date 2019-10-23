import os 
import json 
import shutil 
from tqdm import tqdm 

chartsense_path = "/home/pengyu/Desktop/chart_classification/originaldata/"
directorys_list = os.listdir(chartsense_path)

for directory in tqdm(directorys_list):
    if directory != "map" and directory != "pie" and directory != "radar" and directory != "table" and directory != "venndiagram":

        images_path = os.path.join(chartsense_path, directory)
        images_list = os.listdir(images_path)
        for image in tqdm(images_list):
            new_name = directory+"_"+image 
            shutil.copy2(images_path+"/"+image, "../data/MIX/"+new_name)

## Totally, 3500 images copied from chartsense