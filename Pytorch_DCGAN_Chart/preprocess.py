import os
import cv2
from PIL import Image 
from tqdm import tqdm

bb_path = "./sumit/bb/"
image_path = "./sumit/images/"

for image in tqdm(os.listdir(bb_path)):
    try:
        gt_images = cv2.resize(cv2.imread(bb_path+image), (128,128))
        cv2.imwrite("./sumit/rs_bb/"+image,gt_images)
    except:
        print(bb_path+image)
    
    try:
        input_images = cv2.resize(cv2.imread(image_path+image), (128,128))
        cv2.imwrite("./sumit/rs_images/"+image,input_images)
    except:
        print(image_path+image)
    