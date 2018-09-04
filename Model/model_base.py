#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:04:53 2018

@author: wenyuan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_impl

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

    def _leakyrelu(self, x, leak=0.2, name="lrelu"):
        with tf.name_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

    def _softplus(self, x):
        return tf.nn.softplus(x, name='softplus')

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

    def _drop_out(self, x, rate=0.5):
        return tf.layers.dropout(x, rate=rate, training=self._is_training)

    def _add_noise(self, inputs, mean=0.0, stddev=0.001):
        with tf.name_scope('Add_Noise'):
            noise = tf.random_normal(shape=tf.shape(inputs),
                                     mean=mean,
                                     stddev=stddev,
                                     dtype=inputs.dtype,
                                     name='noise'
                                     )
            inputs = inputs + noise
        return inputs

    def _dense_WN(self,
                  inputs, units,
                  activation=None,
                  weight_norm=True,
                  use_bias=True,
                  kernel_initializer=None,
                  bias_initializer=init_ops.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=None,
                  reuse=None):
        '''
        Dense layer using weight normalizaton
        '''
        layer = Dense(units,
                      activation=activation,
                      weight_norm=weight_norm,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer,
                      kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint,
                      trainable=trainable,
                      name=name,
                      dtype=inputs.dtype.base_dtype,
                      _scope=name,
                      _reuse=reuse)
        return layer.apply(inputs)

class Dense(core_layers.Dense):
  '''
  Dense layer implementation using weight normalization.
  Code borrowed from:
  https://github.com/llan-ml/weightnorm/blob/master/dense.py
  '''
  def __init__(self, *args, **kwargs):
      self.weight_norm = kwargs.pop("weight_norm")
      super(Dense, self).__init__(*args, **kwargs)

  def build(self, input_shape):
      input_shape = tensor_shape.TensorShape(input_shape)
      if input_shape[-1].value is None:
          raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
      self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})
      kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
      if self.weight_norm:
          self.g = self.add_variable(
                "wn/g",
                shape=(self.units,),
                initializer=init_ops.ones_initializer(),
                dtype=kernel.dtype,
                trainable=True)
          self.kernel = nn_impl.l2_normalize(kernel, dim=0) * self.g
      else:
          self.kernel = kernel
      if self.use_bias:
          self.bias = self.add_variable(
                'bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
      else:
          self.bias = None
      self.built = True