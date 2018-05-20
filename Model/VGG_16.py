#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:22:58 2018

@author: wenyuan
"""

import tensorflow as tf

from Model import model_base

class VGG16(model_base.CNN_Base):
    
    def __init__(self, config):        
        super(VGG16, self).__init__(config.IS_TRAINING, config.DATA_FORMAT, \
             config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self._num_classes = config.NUM_CLASSES
        self._filters = [64, 128, 256, 512, 512]
        self._batch_size = config.BATCH_SIZE
        
    def forward_pass(self, x):
        with tf.name_scope('Conv_Block_0'):
            x = self._conv_batch_relu(x, filters = self._filters[0], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[0], \
                                  kernel_size = 3, strides = (1,1))
            x = self._max_pool(x, pool_size = 2)
        
        with tf.name_scope('Conv_Block_1'):
            x = self._conv_batch_relu(x, filters = self._filters[1], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[1], \
                                  kernel_size = 3, strides = (1,1))
            x = self._max_pool(x, pool_size = 2)
        
        with tf.name_scope('Conv_Block_2'):
            x = self._conv_batch_relu(x, filters = self._filters[2], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[2], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[2], \
                                  kernel_size = 3, strides = (1,1))
            x = self._max_pool(x, pool_size = 2)
        
        with tf.name_scope('Conv_Block_3'):
            x = self._conv_batch_relu(x, filters = self._filters[3], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[3], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[3], \
                                  kernel_size = 3, strides = (1,1))
            x = self._max_pool(x, pool_size = 2)
        
        with tf.name_scope('Conv_Block_4'):
            x = self._conv_batch_relu(x, filters = self._filters[4], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[4], \
                                  kernel_size = 3, strides = (1,1))
            x = self._conv_batch_relu(x, filters = self._filters[4], \
                                  kernel_size = 3, strides = (1,1))
            x = self._max_pool(x, pool_size = 2)
        
        
        with tf.name_scope('Fully_Connected'):
            with tf.name_scope('Tensor_Flatten'):
                x = tf.reshape(x, shape = [self._batch_size, -1])
            x = self._fully_connected(x, 4096)
            x = self._fully_connected(x, 4096)
            x = self._fully_connected(x, self._num_classes)
            
        return x
        

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.getcwd()))
    from config import Config
    class TestConfig(Config):
        IS_TRAINING = True
    
    config = TestConfig
    
    model = VGG16(TestConfig)
    x = tf.random_normal(shape = (config.BATCH_SIZE, 224, 224, 3))
    x = model.forward_pass(x)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_out = sess.run(x)
    