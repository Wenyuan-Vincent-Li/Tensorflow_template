#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:36:13 2018

@author: wenyuan
"""
import tensorflow as tf
import os
from time import localtime, strftime
import sys
sys.path.append(os.path.dirname(os.getcwd()))

class Saver(object):
    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        self.saver = tf.train.Saver()
    
    def set_save_path(self, **kwargs):
        self.save_dir = os.path.join(self.save_dir, 'Run_' + \
                               strftime("%Y-%m-%d_%H_%M_%S", localtime()))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            tf.logging.debug('Create model saving dir!')
        
        if 'comments' in kwargs:
            self.comments = kwargs.get('comments')
            self._write_comments()
    
    def save(self, sess, save_name):
        save_name = os.path.join(self.save_dir, save_name)
        self.saver.save(sess, save_name)
    
    def restore(self, sess):
        self.save_dir, filename, start_epoch = self._findfilename()
        self.saver.restore(sess, filename)
        return start_epoch
        
    def _findfilename(self):
        dir_names = next(os.walk(self.save_dir))[1]
        dir_names = filter(lambda f: f.startswith('Run'), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            raise ValueError('Cannot find ckpt file!')
        save_dir = os.path.join(self.save_dir, dir_names[-1])
        checkpoints = next(os.walk(save_dir))[2]
        checkpoints = filter(lambda f: f.startswith("model"), checkpoints)
        checkpoints = sorted(checkpoints)

        if not checkpoints:
            raise ValueError('Cannot find ckpt file!')
        name, suffix, _ = checkpoints[-1].split('.')
        start_epoch = name.split('_')[1]
        checkpoints = os.path.join(save_dir, name + '.' + suffix)
        return save_dir, checkpoints, int(start_epoch)
    
    def _write_comments(self):
        with open(os.path.join(self.save_dir, 'Comments.txt'), 'w') as txt_file:
            txt_file.write(self.comments)
            
if __name__ == "__main__":
    save_dir = os.path.join(os.getcwd(), "weight")
    saver = Saver(save_dir)
    file = saver._findfilename()
    print(file)
    