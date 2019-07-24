import os 
import json 
from tqdm import tqdm 
import cv2
import multiprocessing
import numpy as np

data_path = "../data/SUMIT/rs_images_sampled/"
json_gt_path = "../data/SUMIT/rs_json_gt_sampled/"

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
def content_remover(image):
# for image in tqdm(os.listdir(data_path)):
    image_name, _ = os.path.splitext(image)

    img = cv2.imread(os.path.join(data_path,image))
    json_gt = json.load(open(os.path.join(json_gt_path, image_name+".json"), 'r'))
    mask_img = np.ones((512,512)).astype(np.uint8)*255

    content_bb = json_gt["input"]["task4_output"]["_plot_bb"]
    x0 = content_bb["x0"]
    y0 = content_bb["y0"]
    x1 = x0 + content_bb["width"]
    y1 = y0 + content_bb["height"]

    cv2.rectangle(img,(x0,y0), (x1,y1),(255,255,255),-1)
    cv2.rectangle(mask_img, (x0,y0), (x1,y1),(0,0,0),-1)
    # cv2.imshow("Image Example",img)
    # cv2.waitKey(0)
    cv2.imwrite("../data/SUMIT/rs_content_removed_sampled/"+image, img)
    cv2.imwrite("../data/SUMIT/rs_content_mask_sampled/"+image, mask_img)



content_remover("3.png")

if os.path.exists("../data/SUMIT/rs_content_removed_sampled/") == False:
    os.mkdir("../data/SUMIT/rs_content_removed_sampled/")
if os.path.exists("../data/SUMIT/rs_content_mask_sampled/") == False:
    os.mkdir("../data/SUMIT/rs_content_mask_sampled/")
    
pool = multiprocessing.Pool()
for i in tqdm(pool.imap(content_remover, os.listdir(data_path)), total = len(os.listdir(data_path))):
    pass
    


