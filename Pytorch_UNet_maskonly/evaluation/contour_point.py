import os 
import cv2 
import json 
import numpy as np


def get_corner(img_path):
    if type(img_path) == np.ndarray:
        img = img_path
    else:
        img = cv2.imread(img_path, -1)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = img
    # binary = cv2.bitwise_not(gray)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("ff", thresh)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bbs = []
    centers = []

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        bbs.append([x,y,w,h])
        cx = x + int(w*1./2)
        cy = y + int(h*1./2)
        centers.append([cx, cy])

        # cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        # cv2.circle(img, (cx,cy),2,(0,0,255),-1)
    
    # cv2.imshow("whh", img)
    # cv2.waitKey(0)

    return bbs, centers


# bbs, centers = get_corner("3600.png")


# print(bbs)
# print("="*6)
# print(centers)