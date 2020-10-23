#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mser_sift_feature_select.py
# Created by iFantastic on 2020/10/23

#基于mser+sift的特征提取方法
# 对应opencv-python=3.4.2.16和opencv-contrib-python=3.4.2.16
import cv2
#read image
cap = cv2.VideoCapture(0)
while cap.isOpened():
    retval, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('origin',img)

    #SIFT
    sift= cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(gray, None)

    #kp, des = sift.detectAndCompute(gray,None)  #des是描述子，for match， should use des, bf = cv2.BFMatcher();smatches = bf.knnMatch(des1,des2, k=2
    cv2.drawKeypoints(gray, keypoints, img)
    cv2.imshow('testSift', img)

    # q键退出
    k = cv2.waitKey(10)
    if (k & 0xff == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
