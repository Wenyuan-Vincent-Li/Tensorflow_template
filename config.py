#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:54:43 2018
Template
Base Configurations class.
@author: wenyuan
"""

"""

Written by Wenyuan Li
"""

import math
import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes
    
    BATCH_SIZE = 5
    
    

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
    
    DATA_FORMAT = "channels_last"

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 4  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask


    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    # Network Property
    BATCH_NORM_DECAY = 0.999
    BATCH_NORM_EPSILON = 0.001
    
    # Training Property
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MOMENTUM = 0.5
    SAVE_PER_EPOCH = 5
    RESTORE = True
    
    # Input Pipeline
    DATA_DIR = "" # rewrite this as dataset directory.
    # Input image reszing
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    MIN_QUEUE_EXAMPLES = 3
    
    # Summary
    SUMMARY = True
    SUMMARY_GRAPH = False
    SUMMARY_SCALAR = True
    SUMMARY_IMAGE = False
    SUMMARY_TRAIN_VAL = True
    SUMMARY_HISTOGRAM = False
    
    
    
    def __init__(self):
        """Set values of computed attributes."""
        self.MIN_QUEUE_EXAMPLES = int(15 * 0.4)
        # Effective batch size
#        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

#==============================================================================
#         # Input image size
#         self.IMAGE_SHAPE = np.array(
#             [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])
#==============================================================================

        # Compute backbone size from input image size
#==============================================================================
#         self.BACKBONE_SHAPES = np.array(
#             [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
#               int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
#              for stride in self.BACKBONE_STRIDES])
#==============================================================================

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")