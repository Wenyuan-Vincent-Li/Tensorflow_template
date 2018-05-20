#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:32:57 2018

@author: wenyuan
"""
import tensorflow as tf
from train_base import Train_base

class Train(Train_base):
    def __init__(self, config):
        super(Train, self).__init__(config.LEARNING_RATE, config.MOMENTUM)
        self.config = config
    
    def build_train_graph(self, x, target, model):
        main_graph = model(self.config)
        y = main_graph.forward_pass(x)
        loss = self.loss(target, y)
        return loss
            
    def loss(self, target, network_output):
        with tf.name_scope('Loss'):
            loss = self._cross_entropy_loss(target, network_output)
        return loss
            
if __name__=="__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.getcwd()))
    from config import Config
    from Model.VGG_16 import VGG16 as model
    
    class TestConfig(Config):
        IS_TRAINING = True
    
    config = TestConfig
    x = tf.random_normal(shape = (config.BATCH_SIZE, 224, 224, 3))
    t = tf.random_uniform(shape = (config.BATCH_SIZE,), minval = 0, maxval = 3,
                          dtype = tf.int32)
    t = tf.one_hot(t, depth = config.NUM_CLASSES)
    training = Train(TestConfig)
    loss = training.build_train_graph(x,t, model)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_out = sess.run(loss)