#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :opencv_face_detection.py
# Created by iFantastic on 2020/10/23
'''
进行人脸检测时
    根据人脸的特征
    haar抽取图片中的特征方式
    把特征数据封装在文件中
        python环境的安装包 ---》lib ---> site-packages --- cv2 --- data
    将特征文件存放在于当前的py文件相同的目录下
'''
import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,img = cap.read()
    # 根据特征文件生成一个人脸检测器
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    # 检测图片中的人脸
    face_zones = face_detector.detectMultiScale(img)
    print(face_zones)
    # [[201  91  96  96]]  二维的
    '''
    [201  91  96  96] 检测到的人脸区域
      x    y   宽  高
    因为一张图片中可能包含好几个人脸 每个人脸都是一维的数据
    '''

    # 获取人脸区域
    for x, y, w, h in face_zones:
        # 用一个矩形把人脸区域圈起来
        '''
        img, 要圈中的图片 
        pt1, 左上角的坐标点
        pt2, 右下角的坐标点
        color, 线的颜色 [蓝 绿 红]
        thickness=None 线宽
        '''
        cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=[0,0,255], thickness=2)


    # 显示图片
    cv2.imshow('liqin', img)
    # q键退出
    k = cv2.waitKey(10)
    if (k & 0xff == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()