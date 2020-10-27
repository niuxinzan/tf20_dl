#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :depthwise_conv2d_describe.py
# Created by iFantastic on 2020/10/26

# 针对于mobilenet中的深度可分离卷积操作的解释
import tensorflow as tf
from tensorflow.keras import layers

img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
img = tf.concat(values=[img1,img2],axis=3)
filter1 = tf.constant(value=0, shape=[3,3,1,1],dtype=tf.float32)
filter2 = tf.constant(value=1, shape=[3,3,1,1],dtype=tf.float32)
filter3 = tf.constant(value=2, shape=[3,3,1,1],dtype=tf.float32)
filter4 = tf.constant(value=3, shape=[3,3,1,1],dtype=tf.float32)
filter_out1 = tf.concat(values=[filter1,filter2],axis=2)
filter_out2 = tf.concat(values=[filter3,filter4],axis=2)
filter = tf.concat(values=[filter_out1,filter_out2],axis=3)

out_img = tf.nn.depthwise_conv2d(input=img, filter=filter, strides=[1,1,1,1], padding='VALID')
print(out_img)
import  numpy as np
print(np.shape(img))
out_img2=layers.DepthwiseConv2D(kernel_size=(2,2),padding='same')(img)
print(out_img2)

out_img3 = layers.Conv2D(filters=1,kernel_size=(2,2),padding='same')(img)
print(out_img3)