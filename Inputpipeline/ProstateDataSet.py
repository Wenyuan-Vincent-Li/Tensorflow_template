#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:54:08 2018
This script demonstrate an example of inputpipline on using prostate dataset
@author: wenyuan
"""
import os
import tensorflow as tf


HEIGHT = 1200
WIDTH = 1200
DEPTH = 3

class ProstateDataSet(object):
    """
    Prostate Dataset
    """
    def __init__(self, data_dir, config, subset='train', use_augmentation=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.config = config
    
    def get_filenames(self):
        if self.subset in ['train', 'val']:
            return os.path.join(self.data_dir, 'Tfrecord/' \
                                + self.subset +'.tfrecords')
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)
    
    def input_from_tfrecord_placeholder(self):
        filename = tf.placeholder(tf.string, shape=[None], \
                                   name = "input_filenames")
        # make filenames as placeholder for training and validating purpose
        dataset = tf.data.TFRecordDataset(filename)
        return dataset, filename
    
    def input_from_tfrecord_filename(self):
        filename = self.get_filenames()
        dataset = tf.data.TFRecordDataset(filename)
        return dataset
    
    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([HEIGHT * WIDTH * DEPTH])
        
        label = tf.cast(features['label'], tf.int32)
        image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
        if self.use_augmentation:
            image, label = self.preprocessing(image, label)
        
        return image, label
    
    def preprocessing(self, image, label):
        image = tf.image.resize_image_with_crop_or_pad(image, \
                                                       self.config.IMAGE_HEIGHT, \
                                                       self.config.IMAGE_WIDTH)
        image = tf.random_crop(image, [self.config.IMAGE_HEIGHT, \
                                       self.config.IMAGE_WIDTH, 3])
#        image = tf.image.random_brightness(image, max_delta = 0.5)
#        image = tf.image.random_contrast(image, lower = 0, upper = 1)
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_flip_up_down(image)
#        image = tf.image.random_hue(image, max_delta = 0.1)
#        image = tf.image.random_saturation(image, lower = 0, upper = 1)
#        image = tf.image.rot90(image, k = tf.random_uniform(shape = [],\
#                                            maxval = 3, dtype=tf.int32))
        label = tf.one_hot(label, depth = self.config.NUM_CLASSES)
        return image, label
    
    def shuffle_and_repeat(self, dataset):
#==============================================================================
## Tested but don't work
#         dataset.apply(tf.contrib.data.shuffle_and_repeat(\
#                       buffer_size = \
#                       self.config.MIN_QUEUE_EXAMPLES + \
#                       3 * self.config.BATCH_SIZE, \
#                       count = self.config.EPOCHS))
#==============================================================================
#        dataset = dataset.shuffle(buffer_size = \
#                       self.config.MIN_QUEUE_EXAMPLES + \
#                       3 * self.config.BATCH_SIZE, \
#                       )
        dataset = dataset.repeat(1)    
        return dataset
    
    def batch(self, dataset):
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=self.config.BATCH_SIZE)
        return dataset
    
    def inputpipline_singleset(self):
        # 1 Read in tfrecords
        dataset, filename = self.input_from_tfrecord_placeholder()
        # 2 Parser tfrecords and preprocessing the data
        dataset = dataset.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
        
        # 3 Shuffle and repeat
        dataset = self.shuffle_and_repeat(dataset)
        # 4 Batch it up
        dataset = self.batch(dataset)
        # 5 Make iterator
        iterator = dataset.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()
        
        return image_batch, label_batch, filename, iterator
        ## return the input tensor, iterator, placeholder
    
    def inputpipline_train_val(self, other):
        # 1 Read in tfrecords
        dataset_train = self.input_from_tfrecord_filename()
        dataset_val = other.input_from_tfrecord_filename()
        # 2 Parser tfrecords and preprocessing the data
        dataset_train = dataset_train.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
        dataset_val = dataset_val.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
        # 3 Shuffle and repeat
        dataset_train = self.shuffle_and_repeat(dataset_train)        
        # 4 Batch it up
        dataset_train = self.batch(dataset_train)
        dataset_val = self.batch(dataset_val)
        # 5 Make iterator
        iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                           dataset_train.output_shapes)
        image_batch, label_batch = iterator.get_next()
        init_op_train = iterator.make_initializer(dataset_train)
        init_op_val = iterator.make_initializer(dataset_val)
        
        return image_batch, label_batch, init_op_train, init_op_val
        
        
'''
Following is testing code
'''
def _main_inputpipline_singleset():
    import sys
    sys.path.append(os.path.dirname(os.getcwd()))
    from config import Config
    
    tmp_config = Config()
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "Dataset")
    num_dataset = 0
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        dataset = ProstateDataSet(data_dir, tmp_config, \
                                  'train', use_augmentation = True)
        image_batch, label_batch, filename, iterator = dataset.inputpipline_singleset()
        with tf.Session() as sess:
            sess.run(iterator.initializer, \
                     feed_dict = {filename: [dataset.get_filenames()]})
            while True:
                try:        
                    image_batch_output, label_batch_output = \
                        sess.run([image_batch, label_batch])
                    num_dataset += 1
                except tf.errors.OutOfRangeError:
                    break
    return image_batch_output, label_batch_output

def _main_inputpipline_train_val():
    import sys
    sys.path.append(os.path.dirname(os.getcwd()))
    from config import Config
    
    tmp_config = Config()
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "Dataset")
    num_dataset = 0
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        dataset_train = ProstateDataSet(data_dir, tmp_config, \
                                  'train', use_augmentation = True)
        dataset_val = ProstateDataSet(data_dir, tmp_config, \
                                  'val', use_augmentation = False)
        
        image_batch, label_batch, init_op_train, init_op_val \
            = dataset_train.inputpipline_train_val(dataset_val)
        with tf.Session() as sess:
            sess.run(init_op_train)
            while True:
                try:        
                    image_batch_output, label_batch_output = \
                        sess.run([image_batch, label_batch])
                except tf.errors.OutOfRangeError:
                    break
            sess.run(init_op_val)
            while True:
                try:        
                    image_batch_output, label_batch_output = \
                        sess.run([image_batch, label_batch])
                    num_dataset += 1
                except tf.errors.OutOfRangeError:
                    break
    return image_batch_output, label_batch_output



def _display_image(image):
    from PIL import Image
    import numpy as np
    image = image.astype(np.uint8)
    img = Image.fromarray(image, 'RGB')
    img.show()
        
if __name__ == "__main__":
    image, label = _main_inputpipline_singleset()
#    image, label = _main_inputpipline_train_val()
#    _display_image(image[0, :, : ,:])
                

#==============================================================================
# Total dataset 10; Batch size 1; Epochs 1; Run Loop 10;
# Total dataset 10; Batch size 3; Epochs 1; Run Loop 4; Last Batch dimension (1, ...)
# Total dataset 10; Batch size 3; Epochs 3; Run Loop 10; Last Batch dimension (3, ...)
# Total dataset 10; Batch size 3; Epochs 2; Run Loop 7; Last Batch dimension (2, ...)
#==============================================================================
