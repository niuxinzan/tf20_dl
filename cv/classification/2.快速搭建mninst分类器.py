#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :快速搭建mninst分类器.py
# Created by iFantastic on 2020/10/18
import  tensorflow as tf
from tensorflow.keras import Model,layers,datasets,optimizers,metrics,Sequential
if __name__=='__main__':
    #加载数据
    (x,y),(x_val,y_val)=datasets.mnist.load_data()
    #训练集
    x = tf.convert_to_tensor(x,dtype=tf.float32)/255.
    y = tf.convert_to_tensor(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    train_dataset = train_dataset.shuffle(60000).batch(128)
    #验证集
    x_val = tf.convert_to_tensor(x_val,dtype=tf.float32)/255.
    y_val = tf.convert_to_tensor(y_val,dtype=tf.int32)
    y_val = tf.one_hot(y_val,depth=10)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    val_dataset = val_dataset.batch(128)
    #搭建神经网络
    network = Sequential([layers.Reshape(input_shape=[28,28],target_shape=[28*28]),
                          layers.Dense(512,activation='relu'),
                          layers.Dense(256,activation='relu'),
                          layers.Dense(128,activation='relu'),
                          layers.Dense(64,activation='relu'),
                          layers.Dense(32,activation='relu'),
                          layers.Dense(10)
                          ])
    network.build(input_shape=[None,28,28])
    network.summary()

    #装配模型
    network.compile(optimizer=optimizers.Adam(lr=0.0001),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                    )
    #训练模型
    history = network.fit(train_dataset,epochs=20,validation_data=val_dataset,validation_freq=2)
    network.evaluate(val_dataset)