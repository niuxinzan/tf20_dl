#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mser_sift_feature_select.py
# Created by iFantastic on 2020/10/23

#基于mser+sift的特征提取方法
# 对应opencv-python=3.4.2.16和opencv-contrib-python=3.4.2.16
import cv2
from matplotlib import pyplot as plt
#read image
cap = cv2.VideoCapture(0)
while cap.isOpened():
    retval, img_roig = cap.read()
    gray = cv2.cvtColor(img_roig, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray
    ##创建一个MSER检测器，并检测图像的MSER区域
    ##kpkp保存检测到的keypoint
    mser = cv2.MSER_create()
    regions, boxes = mser.detectRegions(gray)
    kpkp = mser.detect(img)
    print(len(mser.detect(img)))

    ##用红框框出检测到的MSER区域，boxes保存这些区域的左上角的坐标和区域的宽和高
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #创建一个SIFT特征提取器
    siftt= cv2.xfeatures2d.SIFT_create()
    print(len(regions))
    print(len(boxes))
    kp = siftt.detect(img, None)
    ##计算kpkp的局部描述子
    des = siftt.compute(img, kpkp)
    print(len(des[0]))
    ##在图像上画出这些keypoint
    cv2.drawKeypoints(img, kpkp, gray)
    cv2.imshow('haha',gray)
    # q键退出
    k = cv2.waitKey(10)
    if (k & 0xff == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
