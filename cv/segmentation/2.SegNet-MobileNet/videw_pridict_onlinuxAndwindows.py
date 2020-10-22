#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :videw_pridict.py
# Created by iFantastic on 2020/10/22
import cv2
import tensorflow as tf
from tensorflow.keras import models,losses
import time
import copy
from PIL import Image
import numpy as np
import sys
class_colors = [[100,22,200],[110,33,210],[120,44,220],[130,55,230],[140,66,240],[150,77,250],[160,88,230],[170,99,220],[180,100,210],[190,120,10],[200,140,20],[210,160,30],[220,180,40],[230,200,50],[240,220,60],[250,240,70],[160,250,80],[170,240,90],[180,230,10],[190,220,110],[200,200,120],[210,180,140],[220,170,150],[230,160,120],[240,150,170],[250,140,220],[1000,130,210],[120,120,20],[130,110,140],[140,100,70],[150,90,50],[170,80,90]]
NCLASSES = 32
HEIGHT = 416
WIDTH = 416
fps =0.0
@tf.function
def loss_function(y_true,y_pred):
    loss = losses.categorical_crossentropy(y_true,y_pred)
    return  loss
if __name__ == '__main__':
    # parameter = sys.argv
    # 加载模型
    parameter = ["d:/data/saved_model_mobile/"]
    mymodel = models.load_model(parameter[0],custom_objects={'loss_function':loss_function})
    # 启动摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        t1 = time.time()
        ret,img = cap.read()
        #转变成Image
        img = Image.fromarray(np.uint8(img))

        old_img = copy.deepcopy(img)
        orininal_h = np.shape(img)[0]
        orininal_w = np.shape(img)[1]

        img = img.resize((HEIGHT,WIDTH))
        img = np.array(img)
        img = img / 255
        img = np.reshape(img ,newshape=(-1,HEIGHT,WIDTH,3))
        pr = mymodel.predict(img)[0]
        pr = np.reshape(pr,(int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)
        seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))

        colors = class_colors
        for c in range(NCLASSES):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        img = Image.blend(old_img, seg_img, 0.3)
        img =np.array(img)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        cv2.imshow('image', img)
        # q键退出
        k = cv2.waitKey(10)
        if (k & 0xff == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()