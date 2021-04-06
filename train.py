import glob
import json
import math
import os
import pickle
import random

import argparse
import cv2
import gc
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from tqdm import tqdm
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.python.ops import math_ops

from model import *


parser = argparse.ArgumentParser(description='Calculate auc')
parser.add_argument('--model_dir', type=str, default='/jobs/default/', help='directory with parameters (params.json) and to store output')
parser.add_argument('--seed', type=int, default=1234, help='random seed', nargs='?')
args = parser.parse_args()

# Initialize environment
random.seed(args.seed)
with open(args.model_dir + 'params.json') as json_file:
    data = json.load(json_file)
C = data['config']
print(C)

# K-Fold Cross Validation
def leaveoneout(params):
    test_acc = []
    predicted_final_classes = []
    predicted_classes = []
    prediction_list = []
    true_labels = []
    accumulated_test_list = []
    
    lr = params['lr']
    epochs = params['epochs']
    batch_size = params['batch_size']
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    loss = categorical_ordinal_focal_loss(alpha=alpha, gamma=gamma, beta=beta)

    # Train and evaluate model for each fold
    for j in range(0, C['folds']):
        i = j+1 
        
        Train = pickle.load(open(C['data_dir']+"EPG_train_" + str(i) + ".pkl", "rb"))
        Test = pickle.load(open(C['data_dir']+"EPG_test_" + str(i) + ".pkl", "rb"))
        Test_list = pickle.load(open(C['data_dir']+"EPG_test_list_" + str(i) + ".pkl", "rb"))
        accumulated_test_list.append(Test_list[0])

        X_0,X_1,Y = data_generator(Train,C,result="classification")
        X_test_0,X_test_1,Y_test = data_generator(Test,C,result="classification")
        test_name = Test_list[0]

        lrScheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=5, min_lr=1e-7)

        DD_Net = build_DD_Net(C)
        DD_Net.compile(loss=loss, 
                       optimizer=tf.keras.optimizers.Adam(lr), 
                       metrics=['accuracy'])

        history = DD_Net.fit([X_0, X_1],Y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=True,
                shuffle=True,
                callbacks=[lrScheduler],
                validation_data=([X_test_0,X_test_1],Y_test)
                )

        # Evaluate and store predictions
        print('\n# Evaluate on test data')
        results = DD_Net.evaluate([X_test_0, X_test_1], Y_test, batch_size=len(Y_test))
        print('test loss, test acc:', results)

        # Generate predictions
        print('\n# Generate predictions')
        predictions = DD_Net.predict([X_test_0, X_test_1])
        test_acc.append(results[1])
        pred_classes = get_predicted_class(predictions)
        pred_class = total_video_vote(predictions)
        prediction_list.append(predictions)
        true_labels.append(Test['label'])
        predicted_classes.append(pred_classes)
        predicted_final_classes.append(pred_class)
        test_acc_dict = {}
        for i in range(len(test_acc)):
            test_acc_dict[accumulated_test_list[i]] = test_acc[i]
            print(accumulated_test_list[i] + " accuracy: " + str(test_acc[i]))

        del Train, Test, Test_list, history, X_0, X_1, Y
        #gc.collect()
        #K.clear_session()

        # Save results
        if len(test_acc) != 0:
            average_test_accuracy = sum(test_acc) / len(test_acc)
            print("final average test accuracy:", average_test_accuracy)
        else:
            average_test_accuracy = 0

        if 'run_name' in params:
            jsonfilename = args.model_dir + params['run_name'] + 'results.json'
            if params['run_name'] not in run_names:
                run_names.append(params['run_name'])
        else:
            jsonfilename = args.model_dir + 'results.json'

        prediction_non_np = [x.tolist() for x in prediction_list]
        attributes = [average_test_accuracy, test_acc_dict, predicted_classes, predicted_final_classes, prediction_non_np, true_labels, params]
        names = ["average_test_accuracy", "test_acc_dict", "predicted_classes", "predicted_final_classes", "prediction_list", "true_labels", "params"]
        save_json(jsonfilename, attributes, names)

    return average_test_accuracy, DD_Net

if __name__ == "__main__":
    print("Training...")
    leaveoneout(C)
