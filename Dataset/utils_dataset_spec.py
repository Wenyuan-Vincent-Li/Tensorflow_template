#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:40:17 2018

This script contains the utilities that are specific to certain dataset

@author: wenyuan
"""
import os
import numpy as np
from PIL import Image
import scipy.io

def image_annotation_filename_pairs(image_dir, label_dir, subset = "train"):
    '''
    *Dataset Specific
    Get image and annotation filename pairs for a subset
    '''
    ## Modify this part to specify the subset
    train = [x for x in range(10)]
    val = [x for x in range(10, 15)]
    ####
    
    if subset == "train":
        filename_list = train
    else:
        filename_list = val
    
    image_full_path = list(map(lambda x: os.path.join(image_dir, str(x).zfill(4)) \
                               + '.' + "jpg", filename_list))
    label_full_path = list(map(lambda x: os.path.join(label_dir, str(x).zfill(4)) \
                          + '.' + "mat", filename_list))
    
    filename_pair = list(zip(image_full_path, label_full_path))
    return(filename_pair)

def image_read_and_process(image_file):
    '''
    *Dataset Specific
    Read and pre-process image
    '''
    image = np.array(Image.open(image_file))
    # Add image processing code here
    image = image[:1200, :1200, :]
    
    return(image) #[H, W, C]

def label_read_and_process(label_file):
    '''
    *Dataset Specific
    Read and pre-process label
    '''
    mat = scipy.io.loadmat(label_file, mat_dtype=True, 
                       squeeze_me=True, struct_as_record=False)
    label = mat["ATmask"]
    # Label processing here
    label = _label_process(label)
    return label
    
def _label_process(label):
    '''
    *Dataset Specific
    Pre-process label
    '''
    # 1: benign 2: low-grade 3: high-grade 4: stroma
    if np.isin(3, label):
        return 3
    if np.isin(2, label):
        return 2
    if np.isin(1, label):
        return 1
    return 0
    