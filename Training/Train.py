#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:32:57 2018

@author: wenyuan
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from train_base import Train_base
from Inputpipeline.ProstateDataSet import ProstateDataSet as DataSet

class Train(Train_base):
    def __init__(self, config):
        super(Train, self).__init__(config.LEARNING_RATE, config.MOMENTUM)
        self.config = config
    
    def train(self, model):
        tf.reset_default_graph()
        image, label, init_op_train, init_op_val \
            = self._input_fn_train_val()
            
        with tf.device('/cpu:0'): #Todo: parameterize
            loss, accuracy, update_op, reset_op \
                = self._build_train_graph(image, label, model)
            optimizer = self._SGD_w_Momentum_optimizer()
            train_op = self._train_op(optimizer, loss)
    
        LOSS = []
        with tf.Session() as sess:
            init_var = tf.group(tf.global_variables_initializer(), \
                         tf.local_variables_initializer())
            sess.run(init_var)
            for epoch in range(self.config.EPOCHS):
                sess.run(init_op_train)
                while True:
                    try:        
                        _, loss_out = \
                            sess.run([train_op, loss])
                        LOSS.append(loss_out)
                    except tf.errors.OutOfRangeError:
                        break
                ## Perform test on validation
                sess.run([init_op_val, reset_op])
                loss_val = []
                while True:
                    try:        
                        loss_out, accuracy_out, _ = \
                            sess.run([loss, accuracy, update_op])
                        loss_val.append(loss_out)
                    except tf.errors.OutOfRangeError:
                        print("The current loss for epoch {} is {:.2f}, accuracy is {:.2f}."\
                              .format(epoch, np.mean(loss_val), accuracy_out))
                        break
        return LOSS
    
    
    def _input_fn_single_set(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                dataset = DataSet(self.config.DATA_DIR, self.config, \
                                          'train', use_augmentation = True)
                image_batch, label_batch, filename_placeholder, iterator \
                    = dataset.inputpipline_singleset()
                filename = dataset.get_filenames()
                image_batch.set_shape([self.config.BATCH_SIZE, \
                         self.config.IMAGE_HEIGHT, \
                         self.config.IMAGE_WIDTH, 3])
        return image_batch, label_batch, \
               filename_placeholder, iterator, filename
    
    def _input_fn_train_val(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                dataset_train = DataSet(self.config.DATA_DIR, self.config, \
                                          'train', use_augmentation = True)
                dataset_val = DataSet(self.config.DATA_DIR, self.config, \
                                          'val', use_augmentation = False)
                image_batch, label_batch, init_op_train, init_op_val \
                    = dataset_train.inputpipline_train_val(dataset_val)
                image_batch.set_shape([self.config.BATCH_SIZE, \
                         self.config.IMAGE_HEIGHT, \
                         self.config.IMAGE_WIDTH, 3])
        return image_batch, label_batch, init_op_train, init_op_val
        
    def _build_train_graph(self, x, target, model):
        main_graph = model(self.config)
        y = main_graph.forward_pass(x)
        loss = self._loss(target, y)
        accuracy, update_op, reset_op = self._metric(target, y)
        return loss, accuracy, update_op, reset_op
    
    def _metric(self, labels, logits):
        with tf.name_scope('Metric') as scope:
            prediction = tf.argmax(logits, axis = -1)
            labels = tf.argmax(labels, axis = -1)
            accuracy, update_op = self._accuracy_metric(labels, prediction)
            vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(vars)
        return accuracy, update_op, reset_op
    
    def _loss(self, target, network_output):
        with tf.name_scope('Loss'):
            loss = self._cross_entropy_loss(target, network_output)
        return loss

#==============================================================================
# Test code
def _main_wo_input():
     import sys
     import os
     sys.path.append(os.path.dirname(os.getcwd()))
     from config import Config
     from Model.VGG_16 import VGG16 as model
     
     class TestConfig(Config):
         IS_TRAINING = True
     
     config = TestConfig
     x = tf.random_normal(shape = (config.BATCH_SIZE, 28, 28, 3))
     t = tf.random_uniform(shape = (config.BATCH_SIZE,), minval = 0, maxval = 3,
                           dtype = tf.int32)
     t = tf.one_hot(t, depth = config.NUM_CLASSES)
     training = Train(config)
     loss = training._build_train_graph(x, t, model)
     
     with tf.Session() as sess:
         init = tf.group(tf.global_variables_initializer(), \
                         tf.local_variables_initializer())
         sess.run(init)
         loss_out = \
             sess.run([loss])
     return loss_out
#==============================================================================
            
if __name__=="__main__":
    from config import Config
    from Model.VGG_16 import VGG16 as model
    
    class TestConfig(Config):
        IS_TRAINING = True
        DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "Dataset")
    
    config = TestConfig
    
    training = Train(config)
    loss = training.train(model)
    