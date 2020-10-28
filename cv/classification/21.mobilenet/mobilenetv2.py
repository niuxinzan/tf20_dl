#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mobilenetv2.py
# Created by iFantastic on 2020/10/23
# 参考：https://zhuanlan.zhihu.com/p/98874284
# 参考：https://blog.csdn.net/qq_36758914/article/details/106910609
import tensorflow as tf
from tensorflow.keras import layers
# 卷积块
class Conv_block(layers.Layer):
    def __init__(self,filters,kernel_size,stride,**kwargs):
        super(Conv_block,self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU(6)
    def call(self, inputs,training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = self.relu1(x)
        return  x
# 反转残差模块，正常的残差模块通道数是大小大，这里是小大小
class Depthwise_res_block(layers.Layer):
    def __init__(self,filters,kernel,stride,t,input_shannel_size,resdiual = False):
        '''
        :param filters:第三个卷积层的卷积核数
        :param kernel:深度可分离卷积的卷积核尺寸
        :param stride:深度可分录卷积的步长
        :param t:通道扩展系数，输入通道数的t倍
        :param input_shannel_size:输入的通道数
        :param resdiual:是否采用resdiual模块
        '''
        super(Depthwise_res_block,self).__init__()
        self.resdiual=resdiual
        total_channel=t*input_shannel_size
        self.conv1 = Conv_block(filters=total_channel,kernel_size=(1,1),stride=(1,1))

        self.depthwise2 = layers.DepthwiseConv2D(kernel_size=kernel,strides=stride,padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU(6)

        self.conv3 = layers.Conv2D(filters=filters,kernel_size=(1,1),strides=(1,1),padding='same')
        self.bn3 = layers.BatchNormalization()
        self.linear = tf.keras.layers.Activation(tf.keras.activations.linear)


    def call(self, inputs,training=None, **kwargs):
        tmp = inputs
        x = self.conv1(inputs)
        x = self.depthwise2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)
        if self.resdiual:
            x = layers.add([x,tmp])
        return x



def MobileNet_v2(label_size=10):
    inputs = layers.Input(shape=(None,28,28,1))
    x = tf.reshape(inputs, (-1, 28, 28, 1))
    x = Conv_block(filters=32,kernel_size=(3,3),stride=(2,2))(x)

    # 一层不使用残差
    # input_shannel_size =(tf.shape(x))[-1]
    input_shannel_size =x.shape[-1]
    x = Depthwise_res_block(filters=16,kernel=(3,3),stride=(1,1),t=1,input_shannel_size=input_shannel_size,resdiual = False)(x)

    for i in range(2):
        input_shannel_size = x.shape[-1]

        if i==0:
            x = Depthwise_res_block(filters=24,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
        else:
            x = Depthwise_res_block(filters=24,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
    for i in range(3):
        input_shannel_size = x.shape[-1]

        if i==0:
            x = Depthwise_res_block(filters=32,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
        else:
            x = Depthwise_res_block(filters=32,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)

    for i in range(4):
        input_shannel_size = x.shape[-1]

        if i==0:
            x = Depthwise_res_block(filters=64,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
        else:
            x = Depthwise_res_block(filters=64,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
    for i in range(3):
        input_shannel_size = x.shape[-1]

        if i==0:
            x = Depthwise_res_block(filters=96,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
        else:
            x = Depthwise_res_block(filters=96,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
    for i in range(3):
        input_shannel_size = x.shape[-1]

        if i==0:
            x = Depthwise_res_block(filters=160,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
        else:
            x = Depthwise_res_block(filters=160,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
    for i in range(1):
        input_shannel_size = x.shape[-1]

        if i==0:
            x = Depthwise_res_block(filters=320,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
        else:
            x = Depthwise_res_block(filters=320,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
    x = Conv_block(filters=1280,kernel_size=(1,1),stride=(1,1))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1,1,1280))(x)
    x = layers.Conv2D(filters=label_size,kernel_size=(1,1),strides=(1,1),padding='same')(x)
    x = layers.Reshape((label_size,))(x)
    x = layers.Softmax()(x)
    model = tf.keras.Model(inputs,x)
    return model
'''
ValueError: You cannot build your model by calling `build` if your layers do not support float type inputs. Instead, in order to instantiate and build your model, `call` your model on real tensor data (of the correct dtype).
'''
class MobileNet_v21(tf.keras.Model):
    def __init__(self,label_size=10):
        super(MobileNet_v21,self).__init__()
        self.label_size = label_size
    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, (-1, 28, 28, 1))
        x = Conv_block(filters=32,kernel_size=(3,3),stride=(2,2))(x)

        # 一层不使用残差,下面的写法要求输入是numpy
        # input_shannel_size =(tf.shape(x))[-1]
        input_shannel_size =x.shape[-1]
        x = Depthwise_res_block(filters=16,kernel=(3,3),stride=(1,1),t=1,input_shannel_size=input_shannel_size,resdiual = False)(x)

        for i in range(2):
            input_shannel_size = x.shape[-1]

            if i==0:
                x = Depthwise_res_block(filters=24,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
            else:
                x = Depthwise_res_block(filters=24,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
        for i in range(3):
            input_shannel_size = x.shape[-1]

            if i==0:
                x = Depthwise_res_block(filters=32,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
            else:
                x = Depthwise_res_block(filters=32,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)

        for i in range(4):
            input_shannel_size = x.shape[-1]

            if i==0:
                x = Depthwise_res_block(filters=64,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
            else:
                x = Depthwise_res_block(filters=64,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
        for i in range(3):
            input_shannel_size = x.shape[-1]

            if i==0:
                x = Depthwise_res_block(filters=96,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
            else:
                x = Depthwise_res_block(filters=96,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
        for i in range(3):
            input_shannel_size = x.shape[-1]

            if i==0:
                x = Depthwise_res_block(filters=160,kernel=(3,3),stride=(2,2),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
            else:
                x = Depthwise_res_block(filters=160,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
        for i in range(1):
            input_shannel_size = x.shape[-1]

            if i==0:
                x = Depthwise_res_block(filters=320,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=False)(x)
            else:
                x = Depthwise_res_block(filters=320,kernel=(3,3),stride=(1,1),t=6,input_shannel_size=input_shannel_size,resdiual=True)(x)
        x = Conv_block(filters=1280,kernel_size=(1,1),stride=(1,1))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1,1,1280))(x)
        x = layers.Conv2D(filters=self.label_size,kernel_size=(1,1),strides=(1,1),padding='same')(x)
        x = layers.Reshape((self.label_size,))(x)
        x = layers.Softmax()(x)
        return x



