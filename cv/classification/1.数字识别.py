#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :数字识别.py
# Created by iFantastic on 2020/10/18
import tensorflow as tf
print(tf.__version__) #2.2.0
from tensorflow.keras import layers,models,optimizers,datasets,Sequential,metrics

if __name__ == '__main__':
    (x, y),(x_val, y_val)=datasets.mnist.load_data()
    x = tf.convert_to_tensor(x,dtype=tf.float32)/255
    y = tf.convert_to_tensor(y,dtype=tf.int32)
    print(x.shape,y.shape)

    #构建数据集对象
    train_dataset=tf.data.Dataset.from_tensor_slices((x,y)) #构建数据集对象
    train_dataset= train_dataset.batch(32).repeat(10)

    #搭建神经网络
    network=Sequential([layers.Dense(256,activation='relu'),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(10)])
    network.build(input_shape=(None,28*28))
    network.summary()

    #训练神经网络（计算梯度、迭代更新网络参数）
    optimizer =optimizers.SGD(lr=0.001)
    acc_meter=metrics.Accuracy() #创建准确度测量器

    for step,(x,y) in enumerate(train_dataset): #一次输入batch组数据进行训练
        with tf.GradientTape() as tape: #构建梯度记录环境
            x = tf.reshape(x,(-1,28*28))
            out = network(x)
            y_onehot = tf.one_hot(y,depth=10)
            loss = tf.square(out - y_onehot)
            loss = tf.reduce_sum(loss)/32 #注意此处的32与batch size 相对应

            grads =tape.gradient(loss,network.trainable_variables) #计算网络中的各个参数的梯度
            optimizer.apply_gradients(zip(grads,network.trainable_variables)) #更新网络参数
            acc_meter.update_state(tf.argmax(out,axis=1),y) #比较预测值与标签，并计算精度
        if step%200 ==0:
            print('Step:',step,' loss:',float(loss),' Accuracy:',acc_meter.result().numpy())
            network.save('model/')
        acc_meter.reset_states() #每一个step后，准确率清零



