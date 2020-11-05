#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authon   :buf
# @Email    :niuxinzan@cennavi.com.cn
# @File     :fsdf.py
# Created by iFantastic on 2020/11/4

import tensorflow as tf
xy_grid = tf.meshgrid(tf.range(2), tf.range(2))
print(xy_grid)
xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
print(xy_grid)
xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
print(xy_grid)
xy_grid = tf.cast(xy_grid, tf.float32)
print(xy_grid)