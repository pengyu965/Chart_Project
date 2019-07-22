import cv2
import os
import numpy as np
import multiprocessing 
from tqdm import tqdm

rs_images_path = "../data/SUMIT/rs_images_sampled/"
rs_linemap_path = "../data/SUMIT/rs_linemap_sampled/"

def Hough(image_file):
    img = cv2.imread(rs_images_path + image_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/2,300)
    blank_img = np.zeros((512,512),np.uint8)
    for x in range(len(lines)):
        for rho,theta in lines[x]:
            # print(rho, theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1920*(-b))
            y1 = int(y0 + 1920*(a))
            x2 = int(x0 - 1920*(-b))
            y2 = int(y0 - 1920*(a))

            # cv2.circle(img, (x0,y0), 5, (0,0,255), 2)
            # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.line(blank_img,(x1,y1),(x2,y2),(255,255,255),2)
    
    # cv2.imshow("edges",edges)
    # cv2.waitKey(0)
    # cv2.imshow('original', img)
    # cv2.waitKey(0)
    # cv2.imshow('houghlines',blank_img)
    # cv2.waitKey(0)
    cv2.imwrite(rs_linemap_path+image_file, blank_img)



pool = multiprocessing.Pool()
for i in tqdm(pool.imap(Hough, os.listdir(rs_images_path)), total = len(os.listdir(rs_images_path))):
    pass
# PMC_path = "fda"
# for