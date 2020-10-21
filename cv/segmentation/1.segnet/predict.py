#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :predoct.py
# Created by iFantastic on 2020/10/21
from PIL import Image
import tensorflow as tf
from networks.segnet import SegNet
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import os

class_colors = [[0,0,0],[0,255,0]]

NCLASSES = 2
HEIGHT = 416
WIDTH = 416

model = SegNet(n_class=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
model.load_weights(r"C:\Users\buf\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",by_name=True)

if __name__ == '__main__':
    #读取真值数据
    label_file=r'd:/data\camvid\camvid\labels\Seq05VD_f04740_P.png'
    label = Image.open(label_file)
    label = np.array(label)
    print(label)

    # img_file=r'd:/data\camvid\camvid\images\0016E5_01740.png'
    # img = Image.open(img_file)
    # plt.imshow(img)
    # plt.show()
    # old_img = copy.deepcopy(img)
    # orininal_h = np.array(img).shape[0]
    # orininal_w = np.array(img).shape[1]
    # img = img.resize((416,416))
    # img = np.array(img)
    # print(np.shape(img))
    # img = img/255.
    # img = img.reshape(-1,416,416,3)
    # pr = model.predict(img)[0]
    # pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2), NCLASSES)).argmax(axis=-1)
    #
    # seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
    # colors = class_colors
    # for c in range(NCLASSES):
    #     # seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
    #     seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
    #     # seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    # # Image.fromarray将数组转换成image格式
    # seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
    # # 将两张图片合成一张图片
    # image = Image.blend(old_img, seg_img, 0.3)
    # image.save("./img_out/" + 'xxx.jpg')

