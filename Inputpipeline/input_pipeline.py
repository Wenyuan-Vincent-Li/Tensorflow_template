#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:52:44 2018
This script contains some inputpipline fuction include batching, preprocessing,
etc.
@author: wenyuan
"""
import tensorflow as tf

HEIGHT = 1200
WIDTH = 1200
DEPTH = 1200

def tfrecord_parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'input': tf.FixedLenFeature([], tf.string),
            'target': tf.FixedLenFeature([], tf.int64),
        })
    
    image = tf.decode_raw(features['input'], tf.uint8)
    image.set_shape([HEIGHT * WIDTH * DEPTH])
    
    label = tf.cast(features['target'], tf.int32)
    image = tf.reshape(image, [HEIGHT, WIDTH, DEPTH])
    
    return image, label

    
    

def image_preprocess(image, config):
    """Preprocess a single image in [height, width, depth] layout.
       Modify this if label is in the same dimentsion of image, like those in 
       sementic segmentation problem.
    """
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [config.HEIGHT, config.WIDTH, 3])
    image = tf.image.random_brightness(image, max_delta = 0.5)
    image = tf.image.random_contrast(image, lower = 0, upper = 1)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, max_delta = 0.5)
    image = tf.image.random_saturation(image, lower = 0, upper = 1)
    image = tf.image.rot90(image, k = tf.random_uniform([], \
                                        maxval = 3, dtype=tf.int32))
    return image