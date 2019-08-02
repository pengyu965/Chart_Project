import cv2
import numpy as np 
import os 
x = 512
y = 512

new_arr = np.zeros((x,y,3))



cv2.arrowedLine(new_arr, (10,10), (int(10+1*50), int(10+1*50)), (0,0,255),1)

cv2.arrowedLine(new_arr, (10,20), (int(10+1*50), int(20+1*50)), (0,0,255),1)
        
cv2.imshow("example", new_arr)
cv2.waitKey(0)