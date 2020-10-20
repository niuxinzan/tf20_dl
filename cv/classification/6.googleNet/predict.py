#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :predict.py
# Created by iFantastic on 2020/10/20

from model import GoogLeNet
from PIL import Image
import  numpy as np
import  json
import matplotlib.pyplot as plt
import tensorflow as tf
if __name__ == '__main__':
    im_height = 224
    im_width = 224
    #读入图片
    img = Image.open('daisy_test.jpg')
    img = img.resize((im_height,im_width))
    plt.imshow(img)
    plt.show()
    #对原图像进行处理
    img = (np.array(img)/255.-0.5)/0.5
    # 当图像只有一张时，需要增加一维度
    img = np.expand_dims(img ,0)
    # 读取标签说明文件
    try:
        json_file = open('class_indices.json','r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    #构造网络
    # saved model use following method
    # model = tf.keras.models.load_model('./saved_model/')
    #save wieight model use following method
    model = GoogLeNet(class_num=5,im_height=im_height,im_width=im_width,aux_logits=False)
    model.load_weights('save_weights/myGoogLeNet.h5',by_name=True) #加载模型参数
    model.summary()
    result  = model.predict(img)
    predict_class = np.argmax(result)
    print('预测出来的类别为{0}'.format(class_indict[str(predict_class)]))
