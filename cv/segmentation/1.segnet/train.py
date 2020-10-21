#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :train.py
# Created by iFantastic on 2020/10/21
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import numpy as np
import os
import sys
sys.path.append('./')
from networks.segnet import SegNet
NCLASSES = 32
HEIGHT = 416
WIDTH = 416
def get_data(imges_root_dir:str,label_root_dir:str):
    images={}
    labels={}
    for i,j,k in os.walk(imges_root_dir):
        for image_name  in k:
            name = image_name.split(".")[0]
            img_path = i+"/"+image_name
            images[name]=img_path
    for i, j, k in os.walk(label_root_dir):
        for label_name  in k:
            name = (label_name.split(".")[0])[0:len(label_name.split(".")[0])-2]
            label_path = i+"/"+label_name
            labels[name]=label_path
    image_label_pair=[]
    for key in images.keys():
        if key in labels.keys():
            image_label_pair.append(str(images[key]+";"+labels[key]))
    print(image_label_pair)

    # 打乱行，打乱数据有利于训练
    np.random.seed(10101)  # 设置随机种子，
    np.random.shuffle(image_label_pair)
    np.random.seed(None)

    # 切分训练样本，90% 训练；10% 验证
    num_val = int(len(image_label_pair) * 0.1)
    num_train = len(image_label_pair) - num_val
    # print(num_val)
    return image_label_pair, num_train, num_val
def generate_arrays_from_file(lines, batch_size=4):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)  # 打乱300行
            name = lines[i].split(';')[0]  # 训练图片名称
            # 从文件中读取图像

            img = Image.open(name)
            img = img.resize((WIDTH, HEIGHT))  # 调整为416*416
            img = np.array(img)
            img = img/255
            X_train.append(img)  # 把训练集图片存放在X_train中

            name = (lines[i].split(';')[1]).replace("\n", "")  # 图片标签的名称
            # 从文件中读取标签图像
            img = Image.open(name)
            img = img.resize((int(WIDTH/2), int(HEIGHT/2)))  # 调整大小为208*208
            img = np.array(img)
            seg_labels = np.zeros((int(HEIGHT/2), int(WIDTH/2), NCLASSES))  # 初始化一个全零的208*208*32的张量
            for c in range(NCLASSES):
                # seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
                seg_labels[:, :, c] = (img[:, :] == c).astype(int)  # 热独码：208*208的img标签->208*208*32的seg_labels标签
            # (208,208,32)->(208*208,32)
            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            Y_train.append(seg_labels)
            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train), np.array(Y_train))
def loss_function(y_true,y_pred):
    loss = losses.categorical_crossentropy(y_true,y_pred)
    return  loss

# 加载数据
lines, _, _ = get_data(r'D:\data\camvid\camvid\images',r'D:\data\camvid\camvid\labels')
# lines, _, _ = get_data(r'./camvid/camvid/images',r'./camvid/camvid/labels')
# 打乱的数据更有利于训练
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
# 90%用于训练，10%用于估计。
num_val = int(len(lines) * 0.1)
num_train = len(lines) - num_val
# 加载模型
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # 下载后保存的文件名
weights_path = get_file(filename, WEIGHTS_PATH_NO_TOP, cache_subdir='models')
print(weights_path)

model = SegNet(n_class=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
model.load_weights(weights_path,by_name=True)
model.summary()

#
# 保存的方式，3代保存一次
log_dir = "models/"
checkpoint_period = ModelCheckpoint(
                                log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                monitor='val_loss',
                                save_weights_only=True,
                                save_best_only=True,
                                period=6
                                # save_freq=3
                            )
# 学习率下降的方式，val_loss3次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        verbose=2
                    )
# 是否需要早停，当val_loss连续10个epochs不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=2
                    )

model.compile(loss=loss_function,  # 交叉熵损失函数
              optimizer=optimizers.Adam(lr=1e-3),  # 优化器
              metrics=['accuracy'])  # 评价标准
batch_size = 4
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

# 开始训练
model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),  # 训练集
                    steps_per_epoch=max(1, num_train // batch_size),  # 每一个epos的steps数
                    validation_data=generate_arrays_from_file(lines[num_train:], batch_size),  # 验证集
                    validation_steps=max(1, num_val // batch_size),
                    epochs=50,
                    initial_epoch=0,
                    callbacks=[checkpoint_period, reduce_lr, early_stopping])  # 回调
model.save('saved_model')


