#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :mdoel.py
# Created by iFantastic on 2020/10/19
from tensorflow.keras import layers,models,Sequential
def GoogLeNet(im_height=224, im_width=224, class_num=1000, aux_logits=False):
    #输入224*224*3的图像
    input_image = layers.Input(shape=(im_height,im_width,3),dtype='float32')
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
    x = layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
    x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    # inception模块
    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)
    x = layers.MaxPool2D(pool_size=3,strides=2,padding='same', name="maxpool_3")(x)
    # inception模块
    x = Inception(192,96,208,16,48,64,name='inception_4a')(x)
    #判断是否使用辅助分类器1：训练时使用，测试时去掉
    if aux_logits:
        aux1 = InceptionAux(class_num,name='aux_1')(x)
    # inception模块
    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    # 判断是否使用辅助分类器2，训练时使用，测试时去掉
    if aux_logits:
        aux2 = InceptionAux(class_num,name='aux_2')(x)
    # inception模块
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)
    # Inception模块
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    # 平均池化层
    x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)
    # 拉直
    x = layers.Flatten(name="output_flatten")(x)
    x = layers.Dropout(rate=0.4, name="output_dropout")(x)
    x = layers.Dense(class_num, name="output_dense")(x)
    aux3 = layers.Softmax(name="aux_3")(x)
    # 判断是否使用辅助分类器
    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model

class Inception (layers.Layer):
    def __init__(self,ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        '''
        ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj分别对应Inception中各个卷积核的个数,
        **kwargs可变长度字典变量，存放层名称
        '''
        super(Inception,self).__init__(**kwargs)
        #分支一
        self.branch1=layers.Conv2D(ch1x1,kernel_size=1,activation='relu')
        #分支二
        self.branch2=Sequential([layers.Conv2D(ch3x3red,kernel_size=1,activation='relu'),
                                 layers.Conv2D(ch3x3, kernel_size=3, activation='relu',padding='same'),
                                 ])
        #分支三
        self.branch3 = Sequential([layers.Conv2D(ch5x5red,kernel_size=1,activation='relu'),
                                   layers.Conv2D(ch5x5,kernel_size=5,activation='relu',padding='same')])
        #分支四
        self.branch4 = Sequential([layers.MaxPool2D(pool_size=3,strides=1,padding='same'),
                                   layers.Conv2D(filters=pool_proj,kernel_size=1,activation='relu')])
    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4]) #将4个分支按照输出通道进行连接
        return outputs

class InceptionAux(layers.Layer):
    # num_classes表示输出分类节点数，**kwargs存放每层名称
    def __init__(self,num_classes,**kwargs):
        super(InceptionAux,self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5,strides=3) #平均池化
        self.conv = layers.Conv2D(128,kernel_size=1,activation='relu')
        self.fc1 = layers.Dense(1024,activation='relu')
        self.fc2 = layers.Dense(num_classes)
        self.softmax = layers.Softmax()
    def call(self, inputs, **kwargs):
        x =self.averagePool(inputs)
        x = self.conv(x)
        x = layers.Flatten()(x) # 拉直
        x = layers.Dropout(rate=0.5)(x)
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
