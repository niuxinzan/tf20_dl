#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :train.py
# Created by iFantastic on 2020/11/2
from tensorflow.keras import layers,models,Model
import tensorflow as tf
import numpy as np
import os
from darknet52 import DarkNet52
from read_txt import ReadTxt
from make_label import GenerateLabel
from configuration import CATEGORY_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, \
    save_model_dir, save_frequency, load_weights_before_training, load_weights_from_epoch,PASCAL_VOC_IMAGE, \
    test_images_during_training, test_images
from loss import YoloLoss
TXT_DIR="./data.txt"

gpus = tf.config.list_physical_devices(device_type="GPU")
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# dataset
def get_length_of_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count
def generate_dataset():
    txt_dataset = tf.data.TextLineDataset(filenames=TXT_DIR)

    train_count = get_length_of_dataset(txt_dataset)
    train_dataset = txt_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, train_count


train_dataset, train_count = generate_dataset()
yolo_loss = YoloLoss()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=3000,
        decay_rate=0.96,
        staircase=True
    )
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
# metrics
loss_metric = tf.metrics.Mean()



def train_step(image_batch, label_batch):
    with tf.GradientTape() as tape:
        yolo_output = yolov3(image_batch, training=True)
        loss = yolo_loss(y_true=label_batch, y_pred=yolo_output)
    gradients = tape.gradient(loss, yolov3.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, yolov3.trainable_variables))
    loss_metric.update_state(values=loss)


def parse_dataset_batch(dataset):
    image_name_list = []
    boxes_list = []
    len_of_batch = dataset.shape[0]
    for i in range(len_of_batch):
        image_name, boxes = ReadTxt(line_bytes=dataset[i].numpy()).parse_line()
        image_name_list.append(image_name)
        boxes_list.append(boxes)
    boxes_array = np.array(boxes_list)
    return image_name_list, boxes_array

yolov3 = DarkNet52(num_class=CATEGORY_NUM)
yolov3.build(input_shape=(None,416,416,3))
yolov3.summary()


def generate_label_batch(true_boxes):
    true_label = GenerateLabel(true_boxes=true_boxes, input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH]).generate_label()
    return true_label

def resize_image_with_pad(image):
    image_tensor = tf.image.resize_with_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor
def process_single_image(image_filename):
    img_raw = tf.io.read_file(image_filename)
    image = tf.io.decode_jpeg(img_raw, channels=CHANNELS)
    image = resize_image_with_pad(image=image)
    image = tf.dtypes.cast(image, dtype=tf.dtypes.float32)
    image = image / 255.0
    return image

def process_image_filenames(filenames):
    image_list = []
    for filename in filenames:
        image_path = os.path.join(PASCAL_VOC_IMAGE, filename)
        image_tensor = process_single_image(image_path)
        image_list.append(image_tensor)
    return tf.concat(values=image_list, axis=0)


for epoch in range(0, EPOCHS):
    step = 0
    for dataset_batch in train_dataset:
        step += 1
        images, boxes = parse_dataset_batch(dataset=dataset_batch)
        labels = generate_label_batch(true_boxes=boxes)
        train_step(image_batch=process_image_filenames(images), label_batch=labels)
        print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch,
                                                               EPOCHS,
                                                               step,
                                                               tf.math.ceil(train_count / BATCH_SIZE),
                                                               loss_metric.result()))

    loss_metric.reset_states()

    if epoch % save_frequency == 0:
        # net.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')
        yolov3.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch))

    # if test_images_during_training:
        # visualize_training_results(pictures=test_images, model=yolov3, epoch=epoch)

