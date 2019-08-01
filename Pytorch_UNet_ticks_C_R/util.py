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

def out_vis(arr):
    color_lib = [
        (255,255,0),
        (255,0,255),
        (0,255,255),
        (135,206,250),
        (255,192,203),
        (0,0,0),
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
    
    for i in range(x):
        for j in range(y):
            _class = np.argmax(arr[i,j,:6])
            if _class == 2:
                cv2.arrowedLine(new_arr, tuple(i,j), tuple(arr[i,j,6:]), (0,0,255),1)
            if _class == 4:
                cv2.arrowedLine(new_arr, tuple(i,j), tuple(arr[i,j,6:]), (0,255,0),1)

    return new_arr


def channel_binarization(arr):
    x, y, z = arr.shape
    new_arr = np.zeros((x,y,6))

    for i in range(x):
        for j in range(y):
            pixel = arr[i,j,:6]
            idi = np.argmax(pixel)
            new_channel = np.zeros((z))
            new_channel[idi] = 1
            new_arr[i,j,:] = new_channel

    final_arr = np.concatenate((new_arr, arr[:,:,6:]), axis = 2)

    return final_arr
                


