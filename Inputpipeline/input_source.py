#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:14:07 2018

This script contains examples from different input source 
@author: wenyuan
"""

import numpy as np
import tensorflow as tf


no_example = 10
H = 1200
W = 1200
channel = 3

def random_generated_input():
    dataset = tf.data.Dataset.from_tensor_slices(
            {"input": tf.random_uniform([no_example, H, W, channel], \
                                        maxval = 255, dtype=tf.int32),
             "target": tf.random_uniform([no_example], maxval = 3, \
                                        dtype=tf.int32)})
    return dataset

def input_from_numpy(image, label):
    image = tf.convert_to_tensor(image, dtype=tf.int32, name="image")
    label = tf.convert_to_tensor(label, dtype = tf.int32, name="label")
    dataset = tf.data.Dataset.from_tensor_slices(
            {"input": image,
             "target": label})
    ## TODO: Input from filenames and create an iterator: not commonly used
    ## for large dataset
    return dataset

def input_from_numpy_as_placeholder(image, label):
    input_placeholder = tf.placeholder(image.dtype, image.shape)
    target_placeholder = tf.placeholder(label.dtype, label.shape)
    dataset = tf.data.Dataset.from_tensor_slices((input_placeholder, \
                                                  target_placeholder))
    return dataset

def input_from_tfrecord():
    filenames = tf.placeholder(tf.string, shape=[None])
    # make filenames as placeholder for training and validating purpose
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset

def _main_input_from_np():
    image = np.random.randint(low = 0, high = 255, \
                              size = (no_example, H, W, channel))
    label = np.random.randint(low = 0, high = 4, size = (no_example))
    dataset = input_from_numpy(image, label)
    print(dataset.output_types)
    print(dataset.output_shapes)

def _main_input_from_random():
    dataset = random_generated_input()
    print(dataset.output_types)
    print(dataset.output_shapes)

    
if __name__=="__main__":
    # _main_input_from_np()
    # _main_input_from_random()
    
    