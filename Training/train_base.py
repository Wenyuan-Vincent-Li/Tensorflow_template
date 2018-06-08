#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:34:59 2018

@author: wenyuan
"""
import tensorflow as tf

class Train_base(object):
    def __init__(self, learning_rate, momentum = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def _input_fn(self):
        raise NotImplementedError(
                'metirc() is implemented in Model sub classes')
    
    def _build_train_graph(self):
        raise NotImplementedError(
                'loss() is implemented in Model sub classes')
    
    def _loss(self, target, network_output):
        raise NotImplementedError(
                'loss() is implemented in Model sub classes')
    
    def _metric(self, labels, network_output):
        raise NotImplementedError(
                'metirc() is implemented in Model sub classes')
    
    def _train_op(self, optimizer, loss):
        train_op = optimizer.minimize(loss, 
                                      global_step = tf.train.get_global_step())
        return train_op
    
    def _train_op_w_grads(self, optimizer, loss):
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads)
        return train_op, grads
        
    def _cross_entropy_loss(self, labels, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, \
                                                          logits = logits)
        loss = tf.reduce_mean(loss)
        return loss
    
    def _SGD_w_Momentum_optimizer(self):
        optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate,
                                               momentum = self.momentum)
        return optimizer
    
    def _accuracy_metric(self, labels, predictions):
        return tf.metrics.accuracy(labels, predictions)
        