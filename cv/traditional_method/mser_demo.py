#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mser_demo.py
# Created by iFantastic on 2020/10/23

#（注意，现仅个别opencv版本支持开源免费的SIFT、SURF算法函数，如3.4.2.16）
#https://www.cnblogs.com/SakuraYuki/p/13341480.html
import cv2

img = cv2.imread('data/liqin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create()
regions, boxes = mser.detectRegions(gray)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('sp', img)
cv2.waitKey(0)