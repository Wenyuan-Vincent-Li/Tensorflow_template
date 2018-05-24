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
from Training.Summary import Summary
import Training.utils as utils

class Train(Train_base):
    def __init__(self, config, log_dir, **kwargs):
        super(Train, self).__init__(config.LEARNING_RATE, config.MOMENTUM)
        self.config = config
        if self.config.SUMMARY:
            if self.config.SUMMARY_TRAIN_VAL:
                self.summary_train = Summary(log_dir, config, log_type = 'train', \
                                    log_comments = kwargs.get('comments', ''))
                self.summary_val = Summary(log_dir, config, log_type = 'val', \
                                    log_comments = kwargs.get('comments', ''))
            else:
                self.summary = Summary(log_dir, config, \
                                     log_comments = kwargs.get('comments', ''))
    
    def train(self, model):
        tf.reset_default_graph()
        # Input node
        image, label, init_op_train, init_op_val \
            = self._input_fn_train_val()
        # Build up the train graph    
        with tf.device('/cpu:0'): #Todo: parameterize
            loss, accuracy, update_op, reset_op, histogram \
                = self._build_train_graph(image, label, model)
            with tf.name_scope('Train'):
                optimizer = self._SGD_w_Momentum_optimizer()
                train_op, grads = self._train_op_w_grads(optimizer, loss)
            
        tf.logging.debug(utils.variable_name_string())
        # Add summary
        if self.config.SUMMARY:
            if self.config.SUMMARY_TRAIN_VAL:                
                summary_dict_train = {}
                summary_dict_val = {}
                if self.config.SUMMARY_SCALAR:
                    scalar_train = {'train_loss': loss}
                    scalar_val = {'val_loss': loss, 'val_accuracy': accuracy}
                    summary_dict_train['scalar'] = scalar_train
                    summary_dict_val['scalar'] = scalar_val
                if self.config.SUMMARY_IMAGE:
                    image = {'input_image': image}
                    summary_dict_train['image'] = image 
                if self.config.SUMMARY_HISTOGRAM:
                    histogram ['Conv_Block_0_Weight'] = \
                        [var for var in tf.global_variables() if var.name=='conv2d/kernel:0'][0]
                    histogram = utils.grads_dict(grads, histogram)
                    summary_dict_train['histogram'] = histogram
                merged_summary_train = self.summary_train.add_summary(summary_dict_train)
                merged_summary_val = self.summary_val.add_summary(summary_dict_val)
                                
        with tf.Session() as sess:
            if self.config.SUMMARY and self.config.SUMMARY_GRAPH:
                if self.config.SUMMARY_TRAIN_VAL:
                    self.summary_train._graph_summary(sess.graph)
            
            init_var = tf.group(tf.global_variables_initializer(), \
                         tf.local_variables_initializer())
            sess.run(init_var)
            for epoch in range(self.config.EPOCHS):
                sess.run(init_op_train)
                while True:
                    try:        
                        _, loss_out, summary_out = \
                            sess.run([train_op, loss, merged_summary_train])
                    except tf.errors.OutOfRangeError:
                        break  
                if self.config.SUMMARY_TRAIN_VAL:
                    self.summary_train.summary_writer.add_summary(summary_out, epoch)
                
                ## Perform test on validation
                sess.run([init_op_val, reset_op])
                loss_val = []
                while True:
                    try:        
                        loss_out, accuracy_out, _, summary_out = \
                            sess.run([loss, accuracy, update_op, merged_summary_val])
                        loss_val.append(loss_out)
                    except tf.errors.OutOfRangeError:
                        tf.logging.info("The current loss for epoch {} is {:.2f}, accuracy is {:.2f}."\
                              .format(epoch, np.mean(loss_val), accuracy_out))
                        break
                if self.config.SUMMARY_TRAIN_VAL:
                    self.summary_val.summary_writer.add_summary(summary_out, epoch)
            if self.config.SUMMARY_TRAIN_VAL:
                self.summary_train.summary_writer.flush()
                self.summary_train.summary_writer.close()
                self.summary_val.summary_writer.flush()
                self.summary_val.summary_writer.close()
        return
    
    
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
        y, histogram = main_graph.forward_pass(x)
        loss = self._loss(target, y)
        accuracy, update_op, reset_op = self._metric(target, y)
        return loss, accuracy, update_op, reset_op, histogram
    
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
    tf.logging.set_verbosity(tf.logging.INFO)
    
    class TestConfig(Config):
        IS_TRAINING = True
        DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "Dataset")
    
    config = TestConfig    
    log_dir = os.path.join(os.getcwd(),"log")
    comments = 'This is a summary test'
    training = Train(config, log_dir, comments = comments)
    loss = training.train(model)
    