#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :train.py
# Created by iFantastic on 2020/10/20

import tensorflow as tf
from DenseNet import mynet
from matplotlib import pyplot as plt
if __name__ == '__main__':
    mynet.build(input_shape=(None, 28, 28, 1))
    mynet.summary()
    # (x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    # x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    #
    # mynet.compile(loss='sparse_categorical_crossentropy',
    #               optimizer=tf.keras.optimizers.SGD(),
    #               metrics=['accuracy'])
    #
    # history = mynet.fit(x_train, y_train,
    #                     batch_size=64,
    #                     epochs=5,
    #                     validation_split=0.2)
    # # test_scores = mynet.evaluate(x_test, y_test, verbose=2)
    #
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.legend(['training', 'validation'], loc='upper left')
    # plt.show()

