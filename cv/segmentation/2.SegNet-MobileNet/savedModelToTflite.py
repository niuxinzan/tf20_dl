#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :savedModelToTflite.py
# Created by iFantastic on 2020/10/22
# -*- coding:utf-8 -*-

'''
将savedmodel转换成tflite的程序
注意：该转换程序需要在linux系统下运行。
tensorflow版本2.3.0-rc2,低版本转换时会报ops不知此的错误。
有两种解决方案：一个是26、27行注释掉的设置，一个是升级tensorflow版本
建议升级版本
'''
import os
import tensorflow as tf
from tensorflow.keras import losses
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
@tf.function
def loss_function(y_true,y_pred):
    loss = losses.categorical_crossentropy(y_true,y_pred)
    return  loss
model_saved=tf.keras.models.load_model('/APP/niuxinzan/segmentiation_20201021/saved_model_mobile',
                                       custom_objects={'loss_function':loss_function})
converter = tf.lite.TFLiteConverter.from_keras_model(model_saved)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("./segmentiation_mobileNet.tflite", "wb").write(tflite_model)