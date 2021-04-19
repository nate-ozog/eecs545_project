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

import data_primer
import sklearn

from data_primer import standardizeDataDims
from sklearn.ensemble import AdaBoostClassifier
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
        print("Complete extraction for feature"+str(i))
        
            
    print(np.shape(extracted_feats))
    return np.array(extracted_feats[:,:])

def new_label(Y_driver):
    new_output = []
    # sliding window approach
    start = 0
    end = 50
    while start + 50 < len(Y_driver):
        new_label = Y_driver[start:end].mean()

#####     Three classes label     
#         if new_label >= 1.33:
#             new_output.append(2)
#         elif new_label >= 0.67:
#              new_output.append(1)
#         else:
#             new_output.append(0)
 
####    Two classes label       
        if new_label >= 0.5:
            new_output.append(1)
        else:
            new_output.append(0)
            
        start += 25
        end = start + 50
    
    print(np.shape(new_output))
    return np.array(new_output)

## K Nearest Neighbors with Python

NumClasses = 2
numDrivers = 13

_, XDrivers, _, _, YDrivers = standardizeDataDims()
XDrivers, X, YDrivers, Y = prepareData(XDrivers, YDrivers, NumClasses)
X_Drivers = copy.deepcopy(XDrivers)
Y_Drivers = copy.deepcopy(YDrivers)


for i in range(numDrivers):
    X_Drivers[i] = feature_extraction(XDrivers[i])
    Y_Drivers[i] = new_label(YDrivers[i])
    
## Do LoDo validation
accuracy = []
precision = []
recall = []
f1 = []

ave_accuracy = []
ave_precision = []
ave_recall = []
ave_f1 = []


for i in range(13):
#for i in (0,1,2,5,7,8,10,11):
    
    accuracy = []
    precision = []
    recall = []
    f1 = []

    x_test = X_Drivers[i]
    y_test = Y_Drivers[i]
    if i == 0:
        x_train = X_Drivers[1]
        y_train = Y_Drivers[1]
        for j in range(2,13):
        #for j in (2,5,7,8,10,11):
            x_train = np.concatenate((x_train, X_Drivers[j]),axis = 0)
            y_train = np.concatenate((y_train,Y_Drivers[j]), axis = 0)
    
#     print(np.shape(x_train))       
#     print(np.shape(y_train))
            
    if i != 0:
        x_train = X_Drivers[0]
        y_train = Y_Drivers[0]
        for j in range(1,13):
        #for j in (1,2,5,7,8,10,11):
            if j != i:
                x_train = np.concatenate((x_train, X_Drivers[j]),axis = 0)
                y_train = np.concatenate((y_train,Y_Drivers[j]), axis = 0)
                
    
    # Get LODO data
    #x_train, y_train, x_test, y_test = getLODOIterData(XDrivers, YDrivers, i)
        
#     pca = PCA(n_components = 30)
#     pca.fit(fea_train)
#     pca_train = pca.transform(fea_train)
        
#     pca.fit(fea_test)
#     pca_test = pca.transform(fea_test)

    model = KNeighborsClassifier(218, weights = 'distance')  ### 218 for 2 classes 900 for 3 classes
    
    model.fit(x_train[:,:],y_train)
    
    #y_pred = model.predict(x_test)
    #accuracy.append(accuracy_score(y_test, y_pred))
    #precision.append(precision_score(y_test, y_pred, average='macro', zero_division=0)) #zero_division???
    #recall.append(recall_score(y_test, y_pred, average='macro', zero_division=0)) #zero_division???
    #f1.append(f1_score(y_test, y_pred,average='macro', zero_division=0))

    start = 0
    
    while start < (len(y_test)):  
        step = 10
        y_pred = model.predict(x_test[start:start+step,:])   
        accuracy.append(accuracy_score(y_test[start:start+step], y_pred))
        precision.append(precision_score(y_test[start:start+step], y_pred, average='macro', zero_division=0)) #zero_division???
        recall.append(recall_score(y_test[start:start+step], y_pred, average='macro', zero_division=0)) #zero_division???
        f1.append(f1_score(y_test[start:start+step], y_pred,average='macro', zero_division=0))       
        start = start + step
    
    print("Drive id= " + str(i))
    print("Validation Accuracy = " + str(np.mean(accuracy)))
    print("Validation Precision: " + str(np.mean(precision)))
    print("Validation Recall: " + str(np.mean(recall)))
    print("Validation f1: " + str(np.mean(f1)))
    
    ave_accuracy.append(np.mean(accuracy))
    ave_precision.append(np.mean(precision))
    ave_recall.append(np.mean(recall))
    ave_f1.append(np.mean(f1))
    
#     figure, axis = plt.subplots(2, 1)

#     axis[0].hist(y_pred) 
#     axis[1].hist(y_test) 
    
#     accuracy.clear()
#     precision.clear()
#     recall.clear()
#     f1.clear()    
    
print("Average Accuracy = " + str(np.mean(ave_accuracy)))
print("Average Validation Precision: " + str(np.mean(ave_precision)))
print("Average Validation Recall: " + str(np.mean(ave_recall)))
print("Average Validation f1: " + str(np.mean(ave_f1)))    
    



