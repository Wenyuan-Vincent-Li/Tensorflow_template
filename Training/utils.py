#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:50:38 2018

@author: wenyuan
"""
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.tools import inspect_checkpoint as chkp

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

def fn_inspect_checkpoint(ckpt_filepath, **kwargs):
    name = kwargs.get('tensor_name', '')
    if name == '':
        all_tensors = True
    else:
        all_tensors = False    
    chkp.print_tensors_in_checkpoint_file(ckpt_filepath, name, all_tensors)
    
## Test code
def _main_variable_name_string():
    print(variable_name_string())

def _main_inspect_checkpoint():
    from Training.Saver import Saver
    save_dir = os.path.join(os.getcwd(), "weight")
    saver = Saver(save_dir)
    _, filename, _ = saver._findfilename()
    fn_inspect_checkpoint(filename)
    fn_inspect_checkpoint(filename, tensor_name = 'conv2d/kernel')
    
if __name__=='__main__':
    _main_inspect_checkpoint()
    
    