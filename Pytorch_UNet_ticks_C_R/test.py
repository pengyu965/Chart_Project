import cv2
import numpy as np 
import os 

a = np.load("../../data/SUMIT/rs_masks_sampled/545.npy")
b = np.zeros((512,512,3))
x,y,z = a.shape

for i in range(x):
    for j in range(y):
        if a[j,i,1] != 0 or a[j,i,2] != 0:
            # cv2.circle(b, (int(i+a[j,i,1]),int(j+a[j,i,2])),3,(255,255,255),-1)

            print(a[j,i,1:])


cv2.imshow("example", b)
cv2.waitKey(0)