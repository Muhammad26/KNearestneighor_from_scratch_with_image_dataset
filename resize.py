# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 03:48:25 2022

@author: Ibrahim
"""

import cv2
import glob

Data = glob.glob("D:\MS courses\Third Semester\Deep learning\Project\pets\*.jpg")
Images = []
for IMAGE in Data:
    img = cv2.imread(IMAGE)
    resize = cv2.resize(img,(150 ,150),  interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    Images.append(gray)

    cv2.imwrite(IMAGE,gray)

cv2.waitKey(0)