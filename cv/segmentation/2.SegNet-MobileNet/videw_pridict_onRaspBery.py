#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :videw_pridict.py
# Created by iFantastic on 2020/10/22
'''
此程序为树莓派上或pc端执行tflite模型的程序，可能会报错，不要紧
需要将版本升级到tensorflow版本2.3.0-rc2
# 下面代码是在windows端运行的代码
'''
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import cv2
# import numpy as np
# import time
#
# import tensorflow as tf
# import time
# import copy
# from PIL import Image
# class_colors = [[100,22,200],[110,33,210],[120,44,220],[130,55,230],[140,66,240],[150,77,250],[160,88,230],[170,99,220],[180,100,210],[190,120,10],[200,140,20],[210,160,30],[220,180,40],[230,200,50],[240,220,60],[250,240,70],[160,250,80],[170,240,90],[180,230,10],[190,220,110],[200,200,120],[210,180,140],[220,170,150],[230,160,120],[240,150,170],[250,140,220],[1000,130,210],[120,120,20],[130,110,140],[140,100,70],[150,90,50],[170,80,90]]
# NCLASSES = 32
# HEIGHT = 416
# WIDTH = 416
# fps =0.0
# test_image_dir = './test_images/'
# #model_path = "./model/quantize_frozen_graph.tflite"
# model_path = "d:/data/segmentiation_mobileNet.tflite"
#
# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# print(str(input_details[0]))
# output_details = interpreter.get_output_details()
# print(str(output_details))
# # 启动摄像头
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     t1 = time.time()
#     ret, img = cap.read()
#     # 转变成Image
#     img = Image.fromarray(np.uint8(img))
#
#     old_img = copy.deepcopy(img)
#     orininal_h = np.shape(img)[0]
#     orininal_w = np.shape(img)[1]
#
#     img = img.resize((HEIGHT, WIDTH))
#     img = np.array(img).astype(np.float32)
#     img = img / 255
#     img = np.reshape(img, newshape=(-1, HEIGHT, WIDTH, 3))
#
#     interpreter.set_tensor(input_details[0]['index'], img)
#
#     # 注意注意，我要调用模型了
#     interpreter.invoke()
#     pr = interpreter.get_tensor(output_details[0]['index'])
#     pr = np.reshape(pr, (int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)
#     seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
#
#     colors = class_colors
#     for c in range(NCLASSES):
#         seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
#         seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
#         seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
#     seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
#     img = Image.blend(old_img, seg_img, 0.3)
#     img = np.array(img)
#     fps = (fps + (1. / (time.time() - t1))) / 2
#     print("fps= %.2f" % (fps))
#     cv2.imshow('image', img)
#     # q键退出
#     k = cv2.waitKey(10)
#     if (k & 0xff == ord('q')):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#下面代码是在raspberry端运行的代码，但是因为raspberry的tflite版本太低导致运行报错
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import time
import importlib.util
import argparse
import time
import copy
from PIL import Image
class_colors = [[100,22,200],[110,33,210],[120,44,220],[130,55,230],[140,66,240],[150,77,250],[160,88,230],[170,99,220],[180,100,210],[190,120,10],[200,140,20],[210,160,30],[220,180,40],[230,200,50],[240,220,60],[250,240,70],[160,250,80],[170,240,90],[180,230,10],[190,220,110],[200,200,120],[210,180,140],[220,170,150],[230,160,120],[240,150,170],[250,140,220],[1000,130,210],[120,120,20],[130,110,140],[140,100,70],[150,90,50],[170,80,90]]
NCLASSES = 32
HEIGHT = 416
WIDTH = 416
fps =0.0

parser = argparse.ArgumentParser()
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
args = parser.parse_args()
use_TPU = args.edgetpu
# import tensorflow as tf
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


# model_path = "./model/quantize_frozen_graph.tflite"
model_path = "./segmentiation_mobileNet.tflite"

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

# 启动摄像头
cap = cv2.VideoCapture(0)
while cap.isOpened():
    t1 = time.time()
    ret, img = cap.read()
    # 转变成Image
    img = Image.fromarray(np.uint8(img))

    old_img = copy.deepcopy(img)
    orininal_h = np.shape(img)[0]
    orininal_w = np.shape(img)[1]

    img = img.resize((HEIGHT, WIDTH))
    img = np.array(img).astype(np.float32)
    img = img / 255
    img = np.reshape(img, newshape=(-1, HEIGHT, WIDTH, 3))
    interpreter.set_tensor(input_details[0]['index'], img)

    # 注意注意，我要调用模型了
    interpreter.invoke()
    pr = interpreter.get_tensor(output_details[0]['index'])
    pr = np.reshape(pr, (int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)
    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))

    colors = class_colors
    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
    img = Image.blend(old_img, seg_img, 0.3)
    img = np.array(img)
    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    cv2.imshow('image', img)
    # q键退出
    k = cv2.waitKey(10)
    if (k & 0xff == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
