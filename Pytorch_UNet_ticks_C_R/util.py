import os 
import random 
import time
import cv2
from PIL import Image
import numpy as np

def image_norm(arr):
    if len(arr.shape) > 2:
        (x, y, _) = arr.shape
    else:
        (x, y) = arr.shape
    max_v = np.max(arr)
    min_v = np.min(arr)
    new_arr = np.zeros_like(arr)

    if len(arr.shape) > 2:
        for i in range(x):
            for j in range(y):
                new_arr[i,j,:] = (arr[i,j,:] - min_v)*255.0/(max_v-min_v)
    else:
        for i in range(x):
            for j in range(y):
                new_arr[i,j] = (arr[i,j] - min_v)*255.0/(max_v-min_v)
        
    
    return new_arr

def remove_duplicates(List):
    arr = np.array(List)
    if arr.shape != (0,):
        new_arr = np.unique(arr, axis = 0)
    else:
        new_arr = arr
    new_List = new_arr.tolist()
    
    return new_List

def most_frequent_number(List): 
    no_dup_list = remove_duplicates(List)
    max_num = 0
    for element in no_dup_list:
        num = List.count(element)
        if num > max_num:
            max_num = num 
    return max_num

def out_vis(arr, original_image, regression_vis = False):
    # Input is numpy array, [H,W,C]
    color_lib = [
        (255,255,0),
        (255,0,255),
        (0,255,255),
        (135,206,250),
        (255,192,203),
        (120,120,120),
        (191,62,255),
        (255,215,0),
        (255,128,0),
        (100,149,237),
        (0,255,255),
        (202,255,112),
        (255,165,0),
        (250,128,114)
    ]

    x, y, z = arr.shape
    new_arr = np.zeros((x,y,3))

    for i in range(x):
        for j in range(y):
            pixel = arr[i,j,:6]
            idi = np.argmax(pixel)
            new_arr[i,j,:] = np.array(color_lib[idi])
    
    if regression_vis == True:
        internal_points = []
        association_points = []
        for i in range(x):
            for j in range(y):
                _class = np.argmax(arr[i,j,:6])
                if _class == 2 and np.random.rand()>0.8:
                    # cv2.circle(new_arr, (int(i+arr[j,i,6]*100), int(j+arr[j,i,7]*100)), 0, (0,0,255), -1)
                    association_points.append([int(i+arr[i,j,6]), int(j+arr[i,j,7])]) # (Y,X)
                    # print(arr[j,i,6],arr[j,i,7])
                if _class == 4 and np.random.rand()>0.8:
                    # cv2.circle(new_arr, (int(i+arr[j,i,6]*100), int(j+arr[j,i,7]*100)), 0, (0,255,0), -1)
                    internal_points.append([int(i+arr[i,j,6]), int(j+arr[i,j,7])]) # (Y,X)

        total_points = association_points + internal_points
        no_duplicate_list = remove_duplicates(total_points)
        max_num = most_frequent_number(total_points)

        for point in no_duplicate_list:
            if (point in association_points) and (point in internal_points):
                num = total_points.count(point)
                color = np.array([255,255,255])*1.*num/max_num
                cv2.circle(new_arr, (point[1], point[0]),0, (int(color[0]), int(color[1]), int(color[2])), -1)
            
            if (point in association_points) and (point not in internal_points):
                num = total_points.count(point)
                color = np.array([0,0,255])*1.*num/max_num
                cv2.circle(new_arr, (point[1], point[0]),0, (int(color[0]), int(color[1]), int(color[2])), -1)

            if (point not in association_points) and (point in internal_points):
                num = total_points.count(point)
                color = np.array([0,255,0])*1.*num/max_num
                cv2.circle(new_arr, (point[1], point[0]),0, (int(color[0]), int(color[1]), int(color[2])), -1)
                
        new_arr = 0.5*new_arr + 0.5*original_image
    return new_arr


def channel_binarization(arr):
    # Input is np array, [H,W,C]
    x, y, z = arr.shape
    new_arr = np.zeros((x,y,6))

    for i in range(x):
        for j in range(y):
            pixel = arr[i,j,:6]
            idi = np.argmax(pixel)
            new_channel = np.zeros((6))
            new_channel[idi] = 1
            new_arr[i,j,:] = new_channel

    final_arr = np.concatenate((new_arr, arr[:,:,6:]), axis = 2)

    return final_arr
                


