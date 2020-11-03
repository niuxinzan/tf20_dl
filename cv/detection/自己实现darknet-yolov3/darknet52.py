#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :darknet52.py
# Created by iFantastic on 2020/11/2

# backbone：darknet52
from tensorflow.keras import layers
import tensorflow as tf
import sys
sys.path.append("./")
class DBL_block(layers.Layer):
    def __init__(self,filters,kernel_size,strides):
        super(DBL_block,self).__init__()
        self.conv1 = layers.Conv2D(filters=filters,strides=strides, kernel_size=kernel_size,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.leky = layers.LeakyReLU()
    def call(self, inputs, training = None,**kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.leky(x)
        return x

class Residual_block(layers.Layer):
    def __init__(self,filters,repeat):
        super(Residual_block,self).__init__()
        self.repeat = repeat
        self.conv1 = DBL_block(filters,1,1)
        self.conv2 = DBL_block(2*filters,3,1)
    def call(self, inputs,training =None,**kwargs):
        temp = inputs
        for _ in range(self.repeat):
            x = self.conv1(temp)
            x = self.conv2(x)
            x = layers.add([x,temp])
            temp = x
        return temp

class Conv_set_block(layers.Layer):
    def __init__(self,filters):
        super(Conv_set_block,self).__init__()
        self.conv1 = DBL_block(filters, 1, 1)
        self.conv2 = DBL_block(2*filters, 3, 1)
        self.conv3 = DBL_block(filters, 1, 1)
        self.conv4 = DBL_block(2*filters, 3, 1)
        self.conv5 = DBL_block(filters, 1, 1)
    def call(self, inputs, training = None,**kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

'''
ValueError: You cannot build your model by calling `build` if your layers do not support float type inputs. Instead, in order to instantiate and build your model, `call` your model on real tensor data (of the correct dtype).
重启下就好了呢！！！
'''
class DarkNet52(tf.keras.Model):
    def __init__(self,num_class):
        super(DarkNet52,self).__init__()
        # super(DarkNet52, self).build(input_shape=(None,416,416,3))
        self.num_class = num_class
        self.dbl_block_0 = DBL_block(filters=32,kernel_size=(3,3),strides=1)
        self.down_sample_0 = DBL_block(filters=64,kernel_size=(3,3),strides=2)

        self.Residual_block_1 = Residual_block(32,1)
        self.down_sample_1 = DBL_block(filters=128,kernel_size=(3,3),strides=2)

        self.Residual_block_2 = Residual_block(64, 1)
        self.down_sample_2 = DBL_block(filters=256, kernel_size=(3, 3), strides=2)

        self.Residual_block_3 = Residual_block(128, 1)
        self.down_sample_3 = DBL_block(filters=512, kernel_size=(3, 3), strides=2)

        self.Residual_block_4 = Residual_block(256, 1)
        self.down_sample_4 = DBL_block(filters=1024, kernel_size=(3, 3), strides=2)

        self.Residual_block_5 = Residual_block(512, 1)

        self.Conv_set_block_1 = Conv_set_block(512)
        self.Conv_1_1 = DBL_block(filters=1024,kernel_size=(3,3),strides=1)
        self.Conv_1_2 = layers.Conv2D(filters=3*(self.num_class+5),kernel_size=1,strides=1)

        self.Conv_1_1_1 = DBL_block(filters=256,kernel_size=(1,1),strides=1)
        self.upsample_1_1 = layers.UpSampling2D()

        self.Conv_set_block_2 = Conv_set_block(256)
        self.Conv_2_1 = DBL_block(filters=512, kernel_size=(3, 3), strides=1)
        self.Conv_2_2 = layers.Conv2D(filters=3 * (self.num_class + 5), kernel_size=1, strides=1)

        self.Conv_2_1_1 = DBL_block(filters=128, kernel_size=(1, 1), strides=1)
        self.upsample_2_1 = layers.UpSampling2D()

        self.Conv_set_block_3 = Conv_set_block(128)
        self.Conv_3_1 = DBL_block(filters=256, kernel_size=(3, 3), strides=1)
        self.Conv_3_2 = layers.Conv2D(filters=3 * (self.num_class + 5), kernel_size=1, strides=1)

    def call(self, inputs, training=None, **kwargs):
        # inputs = tf.reshape(inputs,(-1,416,6,3))
        x = self.dbl_block_0(inputs)
        x = self.down_sample_0(x)

        x = self.Residual_block_1(x)
        x = self.down_sample_1 (x)

        x = self.Residual_block_2 (x)
        x = self.down_sample_2 (x)

        x = self.Residual_block_3 (x)
        route_1 = x
        x = self.down_sample_3(x)


        x = self.Residual_block_4(x)
        route_2 = x
        x = self.down_sample_4(x)

        x = self.Residual_block_5(x)
        conv = x

        conv = self.Conv_set_block_1(conv)
        conv_lobj_branch = self.Conv_1_1(conv)
        conv_lbbox = self.Conv_1_2(conv_lobj_branch)

        conv =  self.Conv_1_1_1(conv)
        conv = self.upsample_1_1(conv)
        conv = tf.concat([conv, route_2], axis=-1)

        conv = self.Conv_set_block_2 (conv)
        conv_mobj_branch = self.Conv_2_1 (conv)
        conv_mbbox = self.Conv_2_2 (conv_mobj_branch)

        conv = self.Conv_2_1_1 (conv)
        conv = self.upsample_2_1(conv)
        conv = tf.concat([conv,route_1],axis=-1)
        conv = self.Conv_set_block_3(conv)
        conv_sobj_branch = self.Conv_3_1(conv)
        conv_sbbox = self.Conv_3_2(conv_sobj_branch)
        return [conv_lbbox,conv_mbbox,conv_sbbox]
yolov3 = DarkNet52(num_class=20)
yolov3.build(input_shape=(None,416,416,3))
yolov3.summary()
