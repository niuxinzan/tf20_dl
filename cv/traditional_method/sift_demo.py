#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :sift_demo.py
# Created by iFantastic on 2020/10/23
# https://www.cnblogs.com/SakuraYuki/p/13341480.html
#（注意，现仅个别opencv版本支持开源免费的SIFT、SURF算法函数，如3.4.2.16）
import cv2
import numpy as np

img = cv2.imread('data/liqin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray, None)  # 找到关键点

img = cv2.drawKeypoints(gray, kp, img)  # 绘制关键点

cv2.imshow('sp', img)
cv2.waitKey(0)