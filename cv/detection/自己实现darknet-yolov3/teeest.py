#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :teeest.py
# Created by iFantastic on 2020/11/3
import tensorflow as tf
from tensorflow.keras import regularizers
def MLPBlock1():


    inputs = tf.keras.layers.Input((28,28,1))
    dense_1 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense1')
    dense_2 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense2')
    dense_3 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense3')
    dense_4 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense4')
    dense_5 = tf.keras.layers.Dense(60,name='dense5')
    x = dense_1(inputs)
    x = tf.nn.relu(x)

    x = dense_2(x)
    x = tf.nn.relu(x)

    x = dense_3(x)
    x = tf.nn.relu(x)

    x = dense_4(x)
    # x = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense4')(x)
    x = tf.nn.relu(x)

    x = dense_5(x)
    return tf.keras.Model(inputs,x)
class MLPBlock(tf.keras.Model):

    def __init__(self):
        super(MLPBlock, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense1')
        self.dense_2 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense2')
        self.dense_3 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense3')
        # self.dense_4 = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense4')
        self.dense_5 = tf.keras.layers.Dense(60,name='dense5')

    def call(self, inputs,**kwargs):
        x = self.dense_1(inputs)
        x = tf.nn.relu(x)

        x = self.dense_2(x)
        x = tf.nn.relu(x)

        x = self.dense_3(x)
        x = tf.nn.relu(x)

        x = self.dense_4(x)
        # x = tf.keras.layers.Dense(500, kernel_regularizer=regularizers.l2(0.0),name='dense4')(x)
        x = tf.nn.relu(x)

        x = self.dense_5(x)
        return x
sss=MLPBlock1()
sss.build(input_shape=(None,28,28,1))
sss.summary()