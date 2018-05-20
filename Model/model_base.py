#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:04:53 2018

@author: wenyuan
"""
import tensorflow as tf

class CNN_Base(object):
    def __init__(self, is_training, data_format, batch_norm_decay = 0.999,
                 batch_norm_epsilon = 0.001):
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first', 'channels_last'), \
            "Not valide image data format!"
        self._data_format = data_format
    
    def forward_pass(self, x):
        raise NotImplementedError(
                'forward_pass() is implemented in Model sub classes')
    
    def _conv(self, x, filters, kernel_size, strides, padding = 'SAME', 
              trainable = True):
        x = tf.layers.conv2d(inputs = x, filters = filters, 
                             kernel_size = kernel_size, strides = strides,
                             padding = padding, trainable = True)
        return x

    def _batch_norm(self, x):
        if self._data_format == 'channels_first':
          data_format = 'NCHW'
        else:
          data_format = 'NHWC'
        x = tf.contrib.layers.batch_norm(
            x,
            decay=self._batch_norm_decay,
            center=True,
            scale=True,
            epsilon=self._batch_norm_epsilon,
            is_training=self._is_training,
            fused=True,
            data_format=data_format)
        return x
    
    def _relu(self, x):
        return tf.nn.relu(x)
    
    def _max_pool(self, x, pool_size):
        x = tf.layers.max_pooling2d(x, pool_size, strides = (2, 2), \
                                    data_format = self._data_format)
        return x
    
    def _fully_connected(self, x, out_dim):
        x = tf.layers.dense(x, out_dim)
        return x
    
    def _conv_batch_relu(self, x, filters, kernel_size, strides):
        x = self._conv(x, filters, kernel_size, strides)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x