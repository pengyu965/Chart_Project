import os 
import cv2 
import json 
import numpy as np


def get_corner(img_path):
    img = cv2.imread(img_path, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary = cv2.bitwise_not(gray)
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow("ff", thresh)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(hierarchy)


    for contour in contours:
        # print(contour)
        (x,y,w,h) = cv2.boundingRect(contour)
        # print(x,y,w,h)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    
    cv2.imshow("whh", img)
    cv2.waitKey(0)


get_corner("./3600.png")