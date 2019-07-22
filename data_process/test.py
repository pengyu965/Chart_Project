import os 
import cv2 


img = cv2.imread("../data/SUMIT/rs_images/1.png")

cv2.rectangle(img,(round(829*512/1280),round(15*512/960)),(round((829+429)*512/1280),round(33*512/960)),(0,0,255),2)

cv2.imshow("example", img)
cv2.waitKey(0)
