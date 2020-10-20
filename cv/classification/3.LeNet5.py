#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :LeNet5.py
# Created by iFantastic on 2020/10/18
import tensorflow as tf
from tensorflow.keras import models,layers,datasets,optimizers,metrics,Sequential

if __name__ == '__main__':
    #加载数据集
    (x,y),(x_val,y_val) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x,dtype=tf.float32)/255.
    y = tf.convert_to_tensor(y,dtype=tf.int32)
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    train_dataset = train_dataset.batch(32).repeat(10)
    #搭建网络
    network = Sequential([layers.Conv2D(6,kernel_size=3,strides=1),
                          layers.MaxPooling2D(pool_size=2,strides=2),
                          layers.ReLU(),
                          layers.Conv2D(16,kernel_size=3,strides=1),
                          layers.MaxPooling2D(pool_size=2,strides=2),
                          layers.ReLU(),
                          layers.Flatten(),
                          layers.Dense(120,activation='relu'),
                          layers.Dense(84,activation='relu'),
                          layers.Dense(10)
                          ])
    network.build(input_shape=[None,28,28,1])
    network.summary()
    #模型训练
    optimizer = optimizers.SGD(lr=0.0001)
    acc_meter = metrics.Accuracy()

    for step,(x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28,28,1))
            y_onehot = tf.one_hot(y,depth=10)
            out= network(x)
            loss = tf.square(y_onehot-out)
            loss = tf.reduce_sum(loss)/32
            grads = tape.gradient(loss,network.trainable_variables)
            optimizer.apply_gradients(zip(grads,network.trainable_variables))
            acc_meter.update_state(tf.argmax(out,axis=1),y)

        if step%20 == 0:
            print('step: ',step,'acc_meter:',acc_meter.result().numpy(),'loss:',float(loss))
            acc_meter.reset_states()

