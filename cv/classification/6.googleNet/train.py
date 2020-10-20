#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mdoel.py
# Created by iFantastic on 2020/10/19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import GoogLeNet
import tensorflow as tf
import  json
import os

if __name__ == '__main__':
    data_root = 'E:/data/flower_data/flower_data/'  # 根路径
    print(data_root)
    train_dir = data_root+'train/' #训练集路径
    print(train_dir)
    validation_dir = data_root+'val' #验证集路径

    #创建文件save_weights用来存放训练好的模型
    if not os.path.exists('save_weights'):
        os.mkdir('save_weights')

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 1

    def pre_function(img):
        img = img/255. #归一化
        img = (img -0.5 )*2.0 #标准化
        return img
    #定义训练集图像处理器，并对图像进行预处理
    train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                               horizontal_flip=True)
    #定义验证集图像生成器，并对图像进行预处理
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)
    # 使用图像生成器从文件中读取数据，默认对标签进行了one-hot编码
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height,im_width),
                                                               class_mode='categorical')#分类方式
    total_train = train_data_gen.n #训练样本数
    class_indices = train_data_gen.class_indices #数字编码标签字典{类别名称：索引}
    print(class_indices)
    inverse_dict = dict((val, key) for key, val in class_indices.items())  # 转换字典中键与值的位置
    json_str = json.dumps(inverse_dict,indent=4) #将转换后的字典写入文件class_indices.json
    with open ('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 使用图像生成器从验证集validation_dir中读取样本
    val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             target_size=(im_height, im_width),
                                                             class_mode='categorical')
    # 验证集的数量
    total_val = val_data_gen.n
    model = GoogLeNet(im_height=im_height,im_width=im_width,class_num=5,aux_logits=True) #实例化模型
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()

    #使用keras底层api进行网络训练
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # 定义损失函数（这种方式需要one-hot编码）
    optimizer = tf.keras.optimizers.Adam(lr=0.0001) #定义优化器

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acuracy') #定义平均准确率

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy') #定义平均准确率
    @tf.function
    def train_step(images,labels):
        with tf.GradientTape() as tape:
            aux1,aux2,output=model(images,training=True)
            loss1=loss_object(labels,aux1)
            loss2=loss_object(labels,aux2)
            loss3=loss_object(labels,output)
            loss = loss1*0.3 +loss2 *0.3 +loss3
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels,output)
    @tf.function
    def test_step(images,labels):
        _, _, output = model(images, training=False)
        t_loss = loss_object(labels,output)
        test_loss(t_loss)
        test_accuracy(labels,output)
    best_test_loss = float('inf')
    for epoch in range(1,epochs+1):
        train_loss.reset_states()  # 训练损失值清零
        train_accuracy.reset_states()  # clear history info
        test_loss.reset_states()  # clear history info
        test_accuracy.reset_states()  # clear history info
        for step in range(total_train // batch_size):
            images, labels = next(train_data_gen)
            train_step(images,labels)
        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            test_step(test_images, test_labels)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            model.save_weights("./save_weights/myGoogLeNet.h5")  # 保存模型为.h5格式
            model.save('saved_model')



