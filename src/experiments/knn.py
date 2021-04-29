#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:44:04 2021

@author: qianqian
"""

import os, random, math, time
import pickle
import timeit
import tracemalloc
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import data_primer_modified
import sklearn
import resource

from data_primer_modified import standardizeDataDims
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from scipy import stats
import scipy.signal as scisig
import scipy.stats

DATA_FILE = "./data/data.npy" # make sure you extract data.zip
memory_usage = []

ada = AdaBoostClassifier()
bag = BaggingClassifier()
tree = DecisionTreeClassifier()
grad = GradientBoostingClassifier()


def prepareData(XDrivers, YDrivers, numClasses):
  """
  Normalizes data on a per-driver basis. Normalizes
  every feature column between 0 and 1. Organizes
  stress data into a one-hot vector of num classes
  i.e. 2 classes means the boundary of stressed vs
  not stressed is 0.5. Ideally, we want a range, for
  example a classification of 1-10 means
  numClasses = 10.
  """
  # Constants
  N = 0
  D = 0

  # Normalize the feature data
  numDrivers = len(XDrivers)
  for i in range(numDrivers):
    Xi = XDrivers[i]
    XDrivers[i] = (Xi - Xi.min(axis=0)) / (Xi.max(axis=0) - Xi.min(axis=0))
    N += Xi.shape[0]
    D = Xi.shape[1]

  # Create numClasses one-hot vectors for classification
  yRange = np.arange(numClasses + 1)
  yRange = (yRange - yRange.min(axis=0)) / (yRange.max(axis=0) - yRange.min(axis=0))
  for i in range(numDrivers):
    Yi = YDrivers[i]
    YiC = np.zeros(Yi.shape)
    for j in range(1, numClasses):
      lBound = yRange[j]
      uBound = yRange[j + 1]
      YiC += np.where((lBound < Yi) & (Yi <= uBound), j, 0)
    YDrivers[i] = YiC

  # Combine into combined matrix.
  X = np.zeros((N, D))
  Y = np.zeros((N))
  startIdx = 0
  endIdx = 0
  for i in range(len(XDrivers)):
    endIdx = startIdx + XDrivers[i].shape[0]
    X[startIdx:endIdx] = XDrivers[i]
    Y[startIdx:endIdx] = YDrivers[i]
    startIdx = endIdx

  # Return data
  return XDrivers, X, YDrivers, Y

def getLODOIterData(XDrivers, YDrivers, LODOIdx):
    numDrivers = len(XDrivers)

    # Get testing data from LODO
    XTest = XDrivers[LODOIdx]
    YTest = YDrivers[LODOIdx]
    D = XTest.shape[1]

    # Get training data for everyone except the driver we are leaving out
    NTrain = 0
    for i in range(numDrivers):
        if (i != LODOIdx):
            NTrain += XDrivers[i].shape[0]
    XTrain = np.zeros((NTrain, D))
    YTrain = np.zeros((NTrain,))
    startIdx = 0
    endIdx = 0
    for i in range(numDrivers):
        if (i != LODOIdx):
            endIdx = startIdx + XDrivers[i].shape[0]
            XTrain[startIdx:endIdx] = XDrivers[i]
            YTrain[startIdx:endIdx] = YDrivers[i]
            startIdx = endIdx
    
    # Return the split data
    return XTrain, YTrain, XTest, YTest

def output_current_LODO():
    # Output average results
    print("Average across all LODO folds:")
    print("Average Accuracy: " + str(np.mean(accuracy)))
    print("Average Precision: " + str(np.mean(precision)))
    print("Average Recall: " + str(np.mean(recall)))
    print("Average F1: " + str(np.mean(f_1)))
    print("Average Latency: " + str(np.mean(detection_time)))
    print("Average Memory Usage: " + str(np.mean(memory_usage)))

def output_average_results():
    # Output average results
    print("Average across all LODO folds:")
    print("Average Accuracy: " + str(np.mean(all_acc)))
    print("Average Precision: " + str(np.mean(all_prec)))
    print("Average Recall: " + str(np.mean(all_rec)))
    print("Average F1: " + str(np.mean(all_rec)))
    print("Average Latency: " + str(np.mean(detection_time)))
    print("Average Memory Usage: " + str(np.mean(memory_usage)))
    return


def feature_extraction(X_driver):
    
    
    #n = math.floor(len(X_driver)/25-1)
    #n = math.floor(len(X)/15-1)
    #extracted_feats = np.zeros((n,1)) ##9727 for 40 window, 7781 for 50 window
    # sliding window approach
    lAcc = 0
    rAcc = 0
    for i in range(18):
        start = 0
        end = 50
        mean = []
        std = []
        max_ = []
        min_ = []
        #fdm = []
        
        if (i in [4,5,6]):  
            lAcc += X_driver ** 2
            if (i in [4,5]):
                continue
            
        if (i in [7,8,9]):
            rAcc += X_driver ** 2
            if (i in [7,8]):
                continue
                
        while start + 50 < len(X_driver):
            if i == 6:
                mean.append((lAcc[start:end, i]).mean())
                std.append((lAcc[start:end, i]).std())
                max_.append((lAcc[start:end, i]).max())
                min_.append((lAcc[start:end, i]).min())
            if i == 9:
                mean.append((lAcc[start:end, i]).mean())
                std.append((lAcc[start:end, i]).std())
                max_.append((lAcc[start:end, i]).max())
                min_.append((lAcc[start:end, i]).min())
            # mean, std, max, min
            if (i not in [6,9]): 
                mean.append(X_driver[start:end, i].mean())
                std.append(X_driver[start:end, i].std())
                max_.append(X_driver[start:end, i].max())
                min_.append(X_driver[start:end, i].min())
            #fdm.append((X_driver[end, i]-X_driver[start, i])*1.0/120)
            
            start += 25
            end = start + 50
        if(i == 0):
            extracted_feats = np.array(mean).reshape(len(mean),1)
        else:
            extracted_feats = np.append(extracted_feats, np.array(mean).reshape(len(mean),1),1)
        extracted_feats = np.append(extracted_feats, np.array(std).reshape(len(std),1),1)
        extracted_feats = np.append(extracted_feats, np.array(max_).reshape(len(max_),1),1)
        extracted_feats = np.append(extracted_feats, np.array(min_).reshape(len(min_),1),1)
        #extracted_feats = np.append(extracted_feats, np.array(fdm).reshape(len(fdm),1),1)
        #print("Complete extraction for feature"+str(i))
        
            
    #print(np.shape(extracted_feats))
    return np.array(extracted_feats[:,:])

def new_label(Y_driver,num_classes):
    new_output = []
    # sliding window approach
    start = 0
    end = 50
    while start + 50 < len(Y_driver):
        new_label = Y_driver[start:end].mean()
        
        ####    Two classes label 
        if num_classes == 2:      
            if new_label >= 0.5:
                new_output.append(1)
            else:
                new_output.append(0)
         
        ####    Three classes label 
        if num_classes == 3:
            if new_label >= 1.33:
                new_output.append(2)
            elif new_label >= 0.67:
                new_output.append(1)
            else:
                new_output.append(0)
            
        start += 25
        end = start + 50
    
    #print(np.shape(new_output))
    return np.array(new_output)

def validation(model, X_vals, y_vals,numclasses):
    
    """
    Perform validation and testing.
    """
    # perform testing on sliding windows of size 50 for latency
    start = 0
    # window sizes of 50 samples
    while start + 500 < len(X_vals):
        memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        y = new_label(y_vals[start:start+500],numclasses)
        time_start = time.time()
        X = feature_extraction(X_vals[start:start+500])
        memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        preds = model.predict(X)
        detection_time.append(time.time() - time_start)
        memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        accuracy.append(accuracy_score(y, preds))
        precision.append(precision_score(y, preds, average='macro', zero_division=0))
        recall.append(recall_score(y, preds, average='macro', zero_division=0))
        f_1.append(f1_score(y, preds, average='macro', zero_division=0))
        
        all_acc.append(accuracy_score(y, preds))
        all_prec.append(precision_score(y, preds, average='macro', zero_division=0))
        all_rec.append(recall_score(y, preds, average='macro', zero_division=0))
        all_f1.append(f1_score(y, preds, average='macro', zero_division=0))
        
        start += 250
    return

## K Nearest Neighbors with Python

NumClasses = 3
numDrivers = 13
memory_usage = []
detection_time = []



##### Measure latency and ave acc within the batch:

# all_acc = []
# all_prec = []
# all_rec = []
# all_f1 = []
# accuracy = []
# precision = []
# recall = []
# f_1 = []

# _, XDrivers, _, _, YDrivers = standardizeDataDims()
# XDrivers, X, YDrivers, Y = prepareData(XDrivers, YDrivers, NumClasses)
# X_Drivers = copy.deepcopy(XDrivers)
# Y_Drivers = copy.deepcopy(YDrivers)


# for i in range(numDrivers):
#     X_Drivers[i] = feature_extraction(XDrivers[i])
#     Y_Drivers[i] = new_label(YDrivers[i],NumClasses)


# ## Do LoDo validation
# for i in range(13):
# #for i in (0,1,2,5,7,8,10,11):
    
#     x_test = XDrivers[i]
#     y_test = YDrivers[i]
    
#     if i == 0:
#         x_train = X_Drivers[1]
#         y_train = Y_Drivers[1]
#         for j in range(2,13):
#         #for j in (2,5,7,8,10,11):
#             x_train = np.concatenate((x_train, X_Drivers[j]),axis = 0)
#             y_train = np.concatenate((y_train,Y_Drivers[j]), axis = 0)
    
#     if i != 0:
#         x_train = X_Drivers[0]
#         y_train = Y_Drivers[0]
#         for j in range(1,13):
#         #for j in (1,2,5,7,8,10,11):
#             if j != i:
#                 x_train = np.concatenate((x_train, X_Drivers[j]),axis = 0)
#                 y_train = np.concatenate((y_train,Y_Drivers[j]), axis = 0)
                
    
#     # Get LODO data
#     #x_train, y_train, x_test, y_test = getLODOIterData(XDrivers, YDrivers, i)

#     model = KNeighborsClassifier(11, weights = 'distance')  ### 11 for both 2 classes 20 and 3 classes
#     #model = xgb
    
#     model.fit(x_train,y_train)
#     validation(model, x_test, y_test,NumClasses)
#     output_current_LODO()
#     accuracy.clear()
#     precision.clear()
#     recall.clear()
#     f_1.clear()  
    
# output_average_results()

#### More accurate acc and f1 measurement for knn:

numDriver = 13
accuracy = 0
precision = 0
recall = 0
f1 = 0
#Fea_i = 0
    
for i in range(numDriver):

    x_test = X_Drivers[i]
    y_test = Y_Drivers[i]
    
    if i == 0:
        x_train = X_Drivers[1]
        y_train = Y_Drivers[1]
        for j in range(2,13):
            x_train = np.concatenate((x_train, X_Drivers[j]),axis = 0)
            y_train = np.concatenate((y_train,Y_Drivers[j]), axis = 0)
    
    if i != 0:
        x_train = X_Drivers[0]
        y_train = Y_Drivers[0]
        for j in range(1,13):
        #for j in (1,3,5,7,8,10,11,12):
            if j != i:
                x_train = np.concatenate((x_train, X_Drivers[j]),axis = 0)
                y_train = np.concatenate((y_train,Y_Drivers[j]), axis = 0)

        
    model = KNeighborsClassifier(11,weights = 'distance')### 11 for both 2 classes 20 and 3 classes
    #model = xgb
    
    model.fit(x_train,y_train)
    #Fea_i += model.feature_importances_
    y_pred = model.predict(x_test)
      
    accuracy += accuracy_score(y_test, y_pred)
    precision += precision_score(y_test, y_pred, average='macro', zero_division=1) #zero_division???
    recall += recall_score(y_test, y_pred, average='macro', zero_division=1) #zero_division???
    f1 += f1_score(y_test, y_pred,average='macro', zero_division=1)
    
    print(i)
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='macro'))
        
    figure, axis = plt.subplots(2, 1)

    axis[0].hist(y_pred) 
    axis[1].hist(y_test) 

print("score = " + str(accuracy*1.0/numDriver))
print("Average Validation Precision: " + str(np.mean(precision*1.0/numDriver)))
print("Average Validation Recall: " + str(np.mean(recall*1.0/numDriver)))
print("Average Validation f1: " + str(np.mean(f1*1.0/numDriver)))


    



