#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :train.py
# Created by iFantastic on 2020/10/23
import tensorflow as tf
from tensorflow.keras import datasets
import  numpy as np
# mobile v1版本
# from mobilenetv1 import MobileNet_v1
from mobilenetv2 import MobileNet_v2
from mobilenet_v3_small import MobileNetV3Small
(x,y),(x_val,y_val) = datasets.mnist.load_data()
print('niuxinzan',np.shape(x),np.shape(y))
x = x/255.
y = tf.one_hot(y,depth=10)
x_val = x_val/255.
y_val = tf.one_hot(y_val,depth=10)
# model = MobileNet_v1(label_size=10)
# model = MobileNet_v2(label_size=10)
model = MobileNetV3Small()
model.build(input_shape=(None,28,28,1))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['categorical_crossentropy','Recall','AUC'])

model.fit(x=x,y=y,batch_size=32,epochs=10,validation_data=(x_val,y_val),validation_freq=1)

print(model.evaluate(x=x_val,y=y_val))
