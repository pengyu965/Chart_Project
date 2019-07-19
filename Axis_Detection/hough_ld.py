import cv2
import numpy as np

# def Hough(image_path):
# img = cv2.imread('../data/SUMIT/images/0.png')
# img = cv2.resize(img, (512,512), interpolation= cv2.INTER_AREA)
# cv2.imwrite("./resized.png",img)
# # cv2.imshow("resized image", img)
# # cv2.waitKey(0)

# img = cv2.imread(image_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,800)
for x in range(len(lines)):
    # print(x)
    for rho,theta in lines[x]:
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1920*(-b))
        y1 = int(y0 + 1920*(a))
        x2 = int(x0 - 1920*(-b))
        y2 = int(y0 - 1920*(a))

        # cv2.circle(img, (x0,y0), 5, (0,0,255), 2)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('houghlines',img)
cv2.waitKey(0)
    

# PMC_path = "fda"
# for