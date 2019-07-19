import os
import cv2
from PIL import Image 
from tqdm import tqdm
import multiprocessing 
import time

# bb_path = "./data/sumit/bb/"
image_path = "./data/sumit/images/"

def img_resize(image):

        # try:
        #     gt_images = cv2.resize(cv2.imread(bb_path+image), (128,128))
        #     cv2.imwrite("./data/sumit/rs_bb/"+image,gt_images)
        # except:
        #     print(bb_path+image)
        
    try:
        input_images = cv2.resize(cv2.imread(image_path+image), (512,512), interpolation= cv2.INTER_AREA)
        cv2.imwrite("./data/sumit/rs_images/"+image,input_images)
    except:
        print(image_path+image)

start_time = time.time()

pool = multiprocessing.Pool()
pool.map(img_resize, os.listdir(image_path))
pool.close()
print(time.time()-start_time)

start_time = time.time()
pool = multiprocessing.Pool()
for i in tqdm(pool.imap(img_resize, os.listdir(image_path)), total = len(os.listdir(image_path))):
    pass
pool.close()
print(time.time()-start_time)

    