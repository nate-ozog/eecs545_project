#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:50:00 2021

@author: qianqian
"""

import os, random, math, time
import pickle
import numpy as np 
import matplotlib.pyplot as plt

import data_primer
from data_primer import standardizeDataDims

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


DATA_FILE = "../../data/data.npy"

def prepareData(XDrivers, YDrivers,numClasses):
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

  # Combine into a large N x D matrix
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
  return X, Y



### Prepare data 
_, XDrivers, _, _, YDrivers = standardizeDataDims()
NumClasses = 10
X, Y = prepareData(XDrivers, YDrivers, NumClasses)
X,Y = shuffle(X, Y)

## K = 1-40

error_rate = []
score_list = []
cv_list = []

error_rate = []
score_list = []
cv_list = []
precision = 0
reacll = 0

nfold = 10
kf = KFold(n_splits=nfold)
       
for i in range(1,2):
    
    error = 0
    score = 0
    
    for trainIdx, testIdx in kf.split(X):
        
        knn = KNeighborsClassifier(i)
        XTrain = X[trainIdx]
        YTrain = Y[trainIdx]
        XTest = X[testIdx]
        YTest = Y[testIdx]
    
        knn.fit(XTrain,YTrain)
        y_pred = knn.predict(XTest)
        
        error += np.mean(y_pred != YTest)
        score += knn.score(XTest, YTest)
        precision += precision_score(YTest, y_pred, average=None)
        reacll += recall_score(YTest, y_pred, average=None)
        print("Baseline Validation Precision: " + str(precision_score(YTest, y_pred, average=None)))
        print("Baseline Validation Recall: " + str(recall_score(YTest, y_pred, average=None)))
    
    #score_list.append(score*1.0/nfold)
    
    # Print the cv accuracy, precision, recall
    print("Average score = " + str(score*1.0/nfold))
    print("Average Validation Precision: " + str(precision*1.0/nfold))
    print("Average Validation Recall: " + str(precision*1.0/nfold))
    
    print("Double check cv socre with the sklearn build in cv function ")
    # Double check cv socre with the sklearn build in cv function
    knn = KNeighborsClassifier(i)
    cv_scores = cross_val_score(knn, X, Y, cv = nfold) 
    cv_scores = np.mean(cv_scores)
    cv_list.append(np.mean(cv_scores))
    print("K nearest neigbor = " + str(i))
    
    # Print the cv result from sklearn.cross_val_socre
    print(str(nfold)+" fold Cross validation score = " + str(cv_scores))
    
    


# Plot the result for different K 
#plt.figure(figsize=(10,6))
#plot1 = plt.figure(1)
#plt.plot(range(1,21),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
#plot1 = plt.figure(1)
#plt.plot(range(1,21),score_list,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
#plt.plot(range(1,20),cv_list,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
#plt.title('Accuracy vs. K Value')
#plt.xlabel('K')
#plt.ylabel('Accuracy')