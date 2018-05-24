i#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:50:38 2018

@author: wenyuan
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

def variable_name_string():
    name_string = ''
    for v in tf.global_variables():
        name_string += v.name + '\n'
    return name_string

def grads_dict(gradients, histogram_dict):
    
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient
        histogram_dict[variable.name + "/gradients"] = grad_values
        histogram_dict[variable.name + "/gradients_norm"] =\
                       clip_ops.global_norm([grad_values])
    return histogram_dict
    
if __name__=='__main__':
    name_list = ['conv2d/kernel:0', 'conv2d/bias:0', 'BatchNorm/gamma:0', \
                 'BatchNorm/beta:0', 'BatchNorm/moving_mean:0']
    print(variable_name_string(name_list))
    