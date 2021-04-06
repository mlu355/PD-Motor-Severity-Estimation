import gc
import glob
import os
import json
import random

import numpy as np
import scipy.ndimage.interpolation as inter
import tensorflow as tf
from keras import backend as K
from numpyencoder import NumpyEncoder
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve 
from tqdm import tqdm

###################################################################################

def data_generator(T,C,result="classification"):
    """
    Generates data for training, validation and testing. 
    Processes joint data into features. 
    :param T: joint data with labels
    :param C: preset params 
    :param result: type of data to generate (for classification or regression task)
    :return: featurized data for model input
    """
    X_0 = []
    X_1 = []
    Y = []
    for i in tqdm(range(len(T['pose']))): 
        p = np.copy(T['pose'][i])
        p = zoom(p,target_l=C['frame_l'],joints_num=C['joint_n'],joints_dim=C['joint_d'])

        label = np.zeros(C['clc_num'])
        
        if result == "classification":
            y_label_index = T['label'][i]
            label[y_label_index] = 1 
        elif result == "regression":
            label[0] = T['label'][i]
        
        M = get_CG(p,C)
        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)  
    X_1 = np.stack(X_1) 
    Y = np.stack(Y)
    return X_0,X_1,Y

def save_json(filename, attributes, names):
    """
    Save training parameters and evaluation results to json file.
    :param filename: save filename
    :param attributes: attributes to save
    :param names: name of attributes to save in json file
    """
    with open(filename, "w", encoding="utf8") as outfile:
        d = {}
        for i in range(len(attributes)):
            name = names[i]
            attribute = attributes[i]
            d[name] = attribute
        json.dump(d, outfile, indent=4, cls=NumpyEncoder)
        
def get_predictions(file):
    """
    Returns prediction_list (class probabilities) and predicted_final_classes
    :param file: file produced by save_json()
    :return: prediction_list (class probabilities) and predicted_final_classes
    """
    with open(file) as json_file:
        data = json.load(json_file)
    pred_probs = data['prediction_list'] 
    pred_classes = data['predicted_final_classes']
    return pred_classes, pred_probs
        
def get_predicted_class(preds):
    """
    Get predicted classes for each clip in one video.
    :param preds: predicted class probabilities for one video
    :return: predicted class for one video
    """
    p = np.array(preds)
    pred_classes = []
    for clip in p:
        prediction = np.argmax(clip)
        pred_classes.append(prediction)
    
    return pred_classes

def single_vote(pred):
    """
    Get majority vote of predicted classes for the clips in one video.
    :param preds: list of predicted class for each clip of one video
    :return: majority vote of predicted class for one video
    """
    p = np.array(pred)
    counts = np.bincount(p)
    max_count = 0
    max_index = 0
    for i in range(len(counts)):
        if max_count < counts[i]:
            max_index = i
            max_count = counts[i]
    return max_index

def get_vote(pred_classes):
    """
    Get majority vote of predicted class for list of videos.
    :param preds: list of predicted class for each clip of each video
    :return: list of majority votes of predicted class for each video
    """
    majority_votes = []
    for pred_class in pred_classes:
        vote = single_vote(pred_class)
        majority_votes.append(vote)
    return majority_votes

def total_video_vote(pred):
    """
    Get majority vote of all videos (one class prediction per video)
    :param preds: class probabilities for all clips for given videos
    :return: list of one majority vote prediction for each video
    """
    pred_classes = get_predicted_class(pred)
    return single_vote(pred_classes)

# Rescale to be 64 frames
def zoom(p,target_l=64,joints_num=25,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:,m,n] = medfilt(p_new[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]         
    return p_new

def sampling_frame(p,C):
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.85,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]    
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C['frame_l'],C['joint_n'],C['joint_d'])
    return p

def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)

def get_CG(p,C):
    M = []
    iu = np.triu_indices(C['joint_n'],1,C['joint_n'])
    for f in range(C['frame_l']): 
        d_m = cdist(p[f],p[f],'euclidean')  
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M) 
    M = norm_scale(M)
    return M
    
# Custom Keras loss function to calculate accuracy of regression predictions for classes
def acc(y_true, y_pred):
    rounded = K.cast(tf.keras.backend.round(y_pred), dtype='int32')
    equal = tf.keras.backend.equal(rounded, K.cast(y_true, dtype='int32'))
    equal_int = tf.keras.backend.cast(equal,"int32")
    num_correct = K.sum(equal_int)
    ones = tf.keras.backend.cast(tf.keras.backend.ones_like(rounded), "int32")
    num_total = tf.keras.backend.sum(ones) 
    return num_correct / num_total 