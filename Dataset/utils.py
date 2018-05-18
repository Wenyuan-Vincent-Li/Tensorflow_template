#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:04:31 2018

@author: wenyuan
"""
import os
import numpy as np
import tensorflow as tf
from utils_dataset_spec import image_annotation_filename_pairs, \
                                image_read_and_process, label_read_and_process

dataset_dir = os.getcwd()

# Helper functions for defining tf types
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _image_channel_mean(image, accumulated_mean):
    '''
    Compute the accumulated channel mean
    Input: image -- current image
           accumulated_mean -- current accumulated mean
    '''
    accumulated_mean += np.mean(image,axis = (0,1))
    return accumulated_mean

def image_2_tfrecord(image_label_filename_pairs, tfrecords_filename):
    '''
    Convert image data and label to tfrecord
    '''
    with tf.python_io.TFRecordWriter(tfrecords_filename) as record_writer:
        accumulated_mean = np.zeros(3)
        for image_file, label_file in image_label_filename_pairs:
            image = image_read_and_process(image_file)
            label = label_read_and_process(label_file)
            ## compute the image channel_mean
            accumulated_mean = _image_channel_mean(image, accumulated_mean)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image.tobytes()),
                    'label': _int64_feature(label)
                    }))
            record_writer.write(example.SerializeToString())
        channel_mean = accumulated_mean / len(image_label_filename_pairs)
        ## TODO: deal with channel_mean (save it in csv file)
            
def main_image_2_tfrecord():
    '''
    Function example to convert image to tfrecord
    '''
    image_dir = os.path.join(dataset_dir, "Images")
    label_dir = os.path.join(dataset_dir, "Labels")
    filename_pairs = image_annotation_filename_pairs(image_dir, label_dir, \
                                                     subset = "train")
    tfrecords_filename = os.path.join(dataset_dir, "Tfrecord/" + "train" + ".tfrecords")
    image_2_tfrecord(filename_pairs, tfrecords_filename)
        
    
    
if __name__=="__main__":
    main_image_2_tfrecord()
    