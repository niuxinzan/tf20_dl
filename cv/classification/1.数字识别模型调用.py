#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :数字识别.py
# Created by iFantastic on 2020/10/18
import tensorflow as tf
print(tf.__version__) #2.2.0
import cv2
import numpy as np
model_file = 'd:/data/digital_model/'
img_file = 'd:/data/shuzi.jpg'
if __name__ == '__main__':
    (x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    print(np.shape(x))
    cv2.imshow("shili",x[0])

    cv2.imwrite(img_file,x[0])
    network = tf.keras.models.load_model(model_file)
    network.summary()
    img = cv2.imread(img_file,0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.expand_dims(img, 0)
    img = np.resize(img,(1,28*28))
    result = network.predict(img)
    # result = network(img)[0]
    predict = np.argmax(result)
    print('print value:{0}'.format(str(predict)))
    cv2.waitKey(0)






