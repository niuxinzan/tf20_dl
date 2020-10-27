#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mobilenetv1.py
# Created by iFantastic on 2020/10/23
import tensorflow as tf
from tensorflow.keras import models,layers,Sequential
# 参考论文，自己实现的
# 常规卷积块
def conv_block(filters):
    layer_seq = Sequential([layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()
                ])
    return layer_seq
# 深度可分离捐几块
def depthwise_conv_block(point_conv_filter_size=32,strideSize=(1,1)):
        layer_sql= Sequential([layers.DepthwiseConv2D(kernel_size=(3,3),strides=strideSize,use_bias=False,padding='same'),
                               layers.BatchNormalization(),
                               layers.ReLU(6),
                               layers.Conv2D(filters=point_conv_filter_size,kernel_size=(1,1),padding='same',use_bias=False),
                               layers.BatchNormalization(),
                               layers.ReLU(6)])
        return layer_sql
class MobileNet_v1(tf.keras.Model):
    def __init__(self,label_size):
        super(MobileNet_v1,self).__init__()
        self.conv_block = conv_block(32)
        self.depthwise_conv_block1 = depthwise_conv_block(64)
        self.depthwise_conv_block2 = depthwise_conv_block(128,strideSize=(2,2))
        self.depthwise_conv_block3 = depthwise_conv_block(128)
        self.depthwise_conv_block4 = depthwise_conv_block(256,strideSize=(2,2))
        self.depthwise_conv_block5 = depthwise_conv_block(256)
        self.depthwise_conv_block6 = depthwise_conv_block(512,strideSize=(2,2))
        #5个
        self.depthwise_conv_block7 = depthwise_conv_block(512)
        self.depthwise_conv_block8 = depthwise_conv_block(512)
        self.depthwise_conv_block9 = depthwise_conv_block(512)
        self.depthwise_conv_block10 = depthwise_conv_block(512)
        self.depthwise_conv_block11 = depthwise_conv_block(512)
        #
        self.depthwise_conv_block12 = depthwise_conv_block(1024,strideSize=(2,2))
        self.depthwise_conv_block13 = depthwise_conv_block(1024)
        #
        self.averagePool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(label_size,activation='softmax')




    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, (-1, 28, 28, 1))
        x=self.conv_block(x)
        x =self.depthwise_conv_block1(x)
        x =self.depthwise_conv_block2(x)
        x =self.depthwise_conv_block3(x)
        x =self.depthwise_conv_block4(x)
        x =self.depthwise_conv_block5(x)
        x =self.depthwise_conv_block6(x)
        # 5个
        x =self.depthwise_conv_block7(x)
        x =self.depthwise_conv_block8(x)
        x =self.depthwise_conv_block9(x)
        x =self.depthwise_conv_block10(x)
        x =self.depthwise_conv_block11(x)
        #
        x =self.depthwise_conv_block12 (x)
        x =self.depthwise_conv_block13 (x)
        #
        x =self.averagePool(x)
        x =self.dense(x)
        return x


