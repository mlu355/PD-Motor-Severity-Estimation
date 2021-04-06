import gc
import glob
import os
import json
import math
import pickle
import random

import cv2
import keras
from keras.engine.topology import Layer
from keras.optimizers import *
from keras.models import Model
from keras.layers import Input
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras import losses
from keras import initializers
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import *

###################################################################################

def categorical_ordinal_focal_loss(gamma=2., alpha=.25, beta=0.2):
    """
    Categorical focal loss defined in https://arxiv.org/pdf/2007.08920v1.pdf. 
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
      beta -- weighting factor for ordinal component
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/2007.08920v1.pdf
        Focal loss implementation: https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    Usage:
     model.compile(loss=[categorical_ordinal_focal_loss(gamma=2, alpha=.25, beta=0.2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_ordinal_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as y_pred
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        ordinal_dist = K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))
        weights = K.cast(ordinal_dist/(K.int_shape(y_pred)[1] - 1), dtype='float32')
        focal_loss = alpha * K.pow(1 - y_pred, gamma)
        classes = K.int_shape(y_pred)[1]
        weights_expanded = K.repeat_elements(K.expand_dims(weights, axis=1), rep=classes, axis=1)
        combined_loss = (beta * weights_expanded + focal_loss) * cross_entropy
        return K.sum(combined_loss, axis=1) 

    return categorical_ordinal_focal_loss_fixed

# DD-Net model from https://github.com/fandulu/DD-Net
def build_DD_Net(C):
    M = keras.Input(name='M', shape=(C['frame_l'],C['feat_d']))  
    P = keras.Input(name='P', shape=(C['frame_l'],C['joint_n'],C['joint_d'])) 
    FM = build_FM(C['frame_l'], C['joint_n'], C['joint_d'], C['feat_d'], C['filters'])
    
    x = FM([M,P])
    x = GlobalMaxPool1D()(x)
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = Dense(C['clc_num'], activation='softmax')(x)
    
    ######################Self-supervised part
    model = Model(inputs=[M,P],outputs=x)
    return model

def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.compat.v1.image.resize_nearest_neighbor(x,size=[H,W],align_corners=False) # should not alignment here
    return x

def pose_motion(P,frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l,-1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l/2),-1))(P_diff_fast)
    return P_diff_slow,P_diff_fast
    
def c1D(x,filters,kernel):
    x = Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = Dense(filters,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32,joint_n=22,joint_d=2,feat_d=231,filters=16):   
    M = keras.Input(shape=(frame_l,feat_d))
    P = keras.Input(shape=(frame_l,joint_n,joint_d))
    diff_slow,diff_fast = pose_motion(P,frame_l)
    
    x = c1D(M,filters*2,1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x_d_slow = c1D(diff_slow,filters*2,1) # brehere
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
        
    x_d_fast = c1D(diff_fast,filters*2,1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,3) 
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,1) 
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
   
    x = concatenate([x,x_d_slow,x_d_fast])
    x = block(x,filters*2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)
    
    x = block(x,filters*4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x,filters*8)
    x = SpatialDropout1D(0.1)(x)
    
    return Model(inputs=[M,P],outputs=x)
