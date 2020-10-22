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
from tensorflow.keras import losses
@tf.function
def loss_function(y_true,y_pred):
    loss = losses.categorical_crossentropy(y_true,y_pred)
    return  loss
class_colors = [[100,22,200],[110,33,210],[120,44,220],[130,55,230],[140,66,240],[150,77,250],[160,88,230],[170,99,220],[180,100,210],[190,120,10],[200,140,20],[210,160,30],[220,180,40],[230,200,50],[240,220,60],[250,240,70],[160,250,80],[170,240,90],[180,230,10],[190,220,110],[200,200,120],[210,180,140],[220,170,150],[230,160,120],[240,150,170],[250,140,220],[1000,130,210],[120,120,20],[130,110,140],[140,100,70],[150,90,50],[170,80,90]]

NCLASSES = 32
HEIGHT = 416
WIDTH = 416

# model = SegNet(n_class=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
# model.load_weights(r"C:\Users\buf\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",by_name=True)
model = tf.keras.models.load_model('d:/data/saved_model/',custom_objects={'loss_function':loss_function}) #因为模型保存的时候没有保存自定义的函数，所以需要custom制定下
if __name__ == '__main__':
    #读取真值数据
    # label_file=r'd:/data\camvid\camvid\labels\Seq05VD_f04740_P.png'
    # label = Image.open(label_file)
    # label = np.array(label)
    # print(label)

    img_file=r'd:/data\camvid\camvid\images\0016E5_01740.png'
    img = Image.open(img_file)
    plt.imshow(img)
    plt.show()
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]
    img = img.resize((416,416))
    img = np.array(img)
    print(np.shape(img))
    img = img/255.
    img = img.reshape(-1,416,416,3)
    pr = model.predict(img)[0]
    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2), NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
    colors = class_colors
    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    # Image.fromarray将数组转换成image格式
    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
    # 将两张图片合成一张图片
    image = Image.blend(old_img, seg_img, 0.3)
    image.save("./img_out/" + 'xxx.jpg')

