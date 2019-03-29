# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract

image = cv2.imread("t.png")
image = cv2.resize(image,(0,0),fx = 50,fy=50)
print(pytesseract.image_to_string(image,config='-psm 10'))
cv2.imshow("i",image)
cv2.waitKey(0)
