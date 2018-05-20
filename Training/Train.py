#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:32:57 2018

@author: wenyuan
"""
import sys
import os
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
        image, label, filename_placeholder, iterator, filename \
            = self._input_fn()
        image.set_shape([self.config.BATCH_SIZE, \
                         self.config.IMAGE_HEIGHT, \
                         self.config.IMAGE_WIDTH, 3])
        with tf.device('/cpu:0'): #Todo: parameterize
            loss = self._build_train_graph(image, label, model)
            optimizer = self._SGD_w_Momentum_optimizer()
            train_op = self._train_op(optimizer, loss)
    
        LOSS = []
        with tf.Session() as sess:
            init_var = tf.group(tf.global_variables_initializer(), \
                         tf.local_variables_initializer())
            sess.run(init_var)
            for epoch in range(self.config.EPOCHS):
                sess.run(iterator.initializer, \
                     feed_dict = {filename_placeholder: [filename]})
                while True:
                    try:        
                        _, loss_out = \
                            sess.run([train_op, loss])
                        LOSS.append(loss_out)
                    except tf.errors.OutOfRangeError:
                        break
        return LOSS
    
    def _input_fn(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                dataset = DataSet(self.config.DATA_DIR, self.config, \
                                          'train', use_augmentation = True)
                image_batch, label_batch, filename_placeholder, iterator \
                    = dataset.inputpipline()
                filename = dataset.get_filenames()
        return image_batch, label_batch, \
               filename_placeholder, iterator, filename
        
    def _build_train_graph(self, x, target, model):
        main_graph = model(self.config)
        y = main_graph.forward_pass(x)
        loss = self._loss(target, y)
        return loss
    
    def _metric(self, labels, logits):
        with tf.name_scope('Metric'):
            prediction = tf.argmax(logits, axis = -1)
            labels = tf.argmax(labels, axis = -1)
            accuracy = self._accuracy_metric(labels, prediction)
        return accuracy
    
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
    