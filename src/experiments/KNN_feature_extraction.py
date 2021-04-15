#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 03:03:17 2021

@author: qianqian
"""

import os, random, math, time
import pickle
import timeit
import tracemalloc
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import data_primer_with_filter
from data_primer_with_filter import standardizeDataDims

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
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE 

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

def feature_extraction(X):
    
    n = math.floor(len(X)/25)-1
    extracted_feats = np.zeros((n,1)) ##9727 for 40 window, 7781 for 50 window
    # sliding window approach
    
    for i in range(18):
        start = 0
        end = 50
        mean = []
        std = []
        max_ = []
        min_ = []
        while start + 50 < len(X):
            # mean, std, max, min
            mean.append(X[:, [i]][start:end].mean())
            std.append(X[:, [i]][start:end].std())
            max_.append(X[:, [i]][start:end].max())
            min_.append(X[:, [i]][start:end].min())
            start += 25
            end = start + 50
                  
        extracted_feats = np.append(extracted_feats, np.array(mean).reshape(len(mean),1),1)
        extracted_feats = np.append(extracted_feats, np.array(std).reshape(len(std),1),1)
        extracted_feats = np.append(extracted_feats, np.array(max_).reshape(len(max_),1),1)
        extracted_feats = np.append(extracted_feats, np.array(min_).reshape(len(min_),1),1)
    
    print(np.shape(extracted_feats))
    return np.array(extracted_feats)

def new_label(y):
    new_output = []
    # sliding window approach
    start = 0
    end = 50
    
    while start + 50 < len(X):
        new_label = y[start:end].mean()
        if new_label >= 1.5:
            new_output.append(2)
        elif new_label >= 0.5:
            new_output.append(1)
        else:
            new_output.append(0)
        start += 25
        end = start + 50
   
    return np.array(new_output)

_, XDrivers, _, _, YDrivers = standardizeDataDims()
NumClasses = 3
XDrivers, X, YDrivers, Y = prepareData(XDrivers, YDrivers, NumClasses)


### Extact features(mean, std, min, max) using moving time window of size 50
fea_new = feature_extraction(X)
fea_new = fea_new[:,1:]
label_new = new_label(Y)

### Use PCA to reduce the latent
pca = PCA()
pca.fit(fea_new)
true_ev = pca.explained_variance_

select_fea = 30
print(np.sum(true_ev[0:select_fea])/np.sum(true_ev))

pca = PCA(n_components = select_fea)
pca.fit(fea_new)
PCA_fea = pca.transform(fea_new)

#### 10 fold cross validation

# tracemalloc.start()
# start = timeit.default_timer()

nfold = 13
kf = KFold(n_splits=nfold)

#X = fea_new
X_knn = fea_new[:,15:]
Y_knn = label_new

oversample = SMOTE(k_neighbors=2)
X_knn,Y_knn = oversample.fit_resample(X_knn, Y_knn)
scaler = StandardScaler()
X_knn = scaler.fit_transform(X_knn)


for i in range(1):
    
    error = 0
    score = 0
    precision = 0
    recall = 0
    f1 = 0
    
    for trainIdx, testIdx in kf.split(X_knn):
        

        knn = KNeighborsClassifier(1,weights = 'distance')
        XTrain = X_knn[trainIdx]
        YTrain = Y_knn[trainIdx]
        XTest = X_knn[testIdx]
        YTest = Y_knn[testIdx]
        
#         XTrain = X1
#         YTrain = Y1
#         XTest = X2
#         YTest = Y2
    
        knn.fit(XTrain,YTrain)
        y_pred = knn.predict(XTest)
        
        error += np.mean(y_pred != YTest)
        score += knn.score(XTest, YTest)
#         precision += precision_score(YTest, y_pred, average=None)
#         recall += recall_score(YTest, y_pred, average=None)
        precision += precision_score(YTest, y_pred, average='macro', zero_division=0) #zero_division???
        recall += recall_score(YTest, y_pred, average='macro', zero_division=0) #zero_division???
        f1 += f1_score(YTest, y_pred,average='macro')
        
        #print("Baseline Validation Precision: " + str(precision_score(YTest, y_pred, average=None)))
        #print("Baseline Validation Recall: " + str(recall_score(YTest, y_pred, average=None)))
        
   

    # Print the cv accuracy, precision, recall
print("error = " + str(error*1.0/nfold))
print("score = " + str(score*1.0/nfold))
print("Average Validation Precision: " + str(np.mean(precision*1.0/nfold)))
print("Average Validation Recall: " + str(np.mean(recall*1.0/nfold)))
print("Average Validation f1: " + str(np.mean(f1*1.0/nfold)))


#### drop one driver for validation

## drop one Driver
numDrivers = 13

error = 0
score = 0
precision = 0
recall = 0
f1 = 0

oversample = SMOTE(k_neighbors=1)
X_fea,Y_label = oversample.fit_resample(fea_new,label_new)
scaler = StandardScaler()
X_fea = scaler.fit_transform(X_fea)

pca = PCA(n_components = 30)
pca.fit(fea_new)
PCA_fea = pca.transform(fea_new)

oversample = SMOTE(k_neighbors=5)
PCA_fea,PCA_label = oversample.fit_resample(PCA_fea,label_new)
scaler = StandardScaler()
PCA_fea = scaler.fit_transform(PCA_fea)

fea = X_fea
label = Y_label

for i in range(numDrivers):
     
    # Get LODO data
#     #XTrain, YTrain, XTest, YTest = getLODOIterData(XDrivers, YDrivers, i)
    
#     fea_train = feature_extraction(XTrain)
#     fea_train = fea_train[:,1:]
#     new_ytrain = new_label(YTrain)
#     fea_test = feature_extraction(XTest)
#     fea_test = fea_train[:,1:]
#     new_ytest = new_label(YTest)
    
    used_len = 0
    if i == 0:
        used_fea =  np.int(np.floor(used_len/25))
        fea_len =  np.int(np.floor(len(XDrivers[0])/25))
        fea_test = fea[used_fea:(used_fea+fea_len),:]
        new_ytest = label[used_fea:(used_fea+fea_len)]
        fea_train = fea[used_fea+fea_len-1:,:]
        new_ytrain = label[used_fea+fea_len-1:]
    else: 
        for j in range(i):
            used_len +=  len(XDrivers[j])
        fea_len =  np.int(np.floor(len(XDrivers[i])/25))
        used_fea =  np.int(np.floor(used_len/25))
        fea_test = fea[used_fea-1:(used_fea+fea_len),:]
        new_ytest = label[used_fea-1:(used_fea+fea_len)]
        fea_train = np.concatenate([fea[0:used_fea,:], fea[used_fea+fea_len-1:,:]])
        new_ytrain = np.concatenate([label[0:used_fea], label[used_fea+fea_len-1:]])
        
#     pca = PCA(n_components = 30)
#     pca.fit(fea_train)
#     pca_train = pca.transform(fea_train)
        
#     pca.fit(fea_test)
#     pca_test = pca.transform(fea_test)
    
    x_train = fea_train
    y_train = new_ytrain
    x_test = fea_test
    y_test = new_ytest
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    

    knn = KNeighborsClassifier(1,weights = 'distance')
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
        
    error += np.mean(y_pred != y_test)
    score += knn.score(x_test, y_test)
    precision += precision_score(y_test, y_pred, average='macro', zero_division=0) #zero_division???
    recall += recall_score(y_test, y_pred, average='macro', zero_division=0) #zero_division???
    f1 += f1_score(y_test, y_pred,average='macro')
    #print(knn.score(x_test, y_test))
    
print("error = " + str(error*1.0/nfold))
print("score = " + str(score*1.0/nfold))
print("Average Validation Precision: " + str(np.mean(precision*1.0/numDrivers )))
print("Average Validation Recall: " + str(np.mean(recall*1.0/numDrivers )))
print("Average Validation f1: " + str(np.mean(f1*1.0/numDrivers )))  
