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


DATA_FILE = "../../data/data.npy"

def prepareData(XDrivers, YDrivers, NumClasses):
    
###  Refer to Nate-orog's nn.py to prepare the dataset
  """
  Normalizes data on a per-driver basis. Normalizes
  every feature column between 0 and 1. Organizes
  stress data into a one-hot vector of num classes
  i.e. 2 classes means the boundary of stressed vs
  not stressed is 0.5. Ideally, we want a range, for
  example a classification of 1-10 means
  NumClasses = 10.
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

  # Create NumClasses one-hot vectors for classification
  yRange = np.arange(NumClasses + 1)
  yRange = (yRange - yRange.min(axis=0)) / (yRange.max(axis=0) - yRange.min(axis=0))
  for i in range(numDrivers):
    Yi = YDrivers[i]
    YiC = np.zeros(Yi.shape)
    for j in range(1, NumClasses):
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
NumClasses = 3
X, Y = prepareData(XDrivers, YDrivers, NumClasses)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
                

# We'll start with k=1.

knn = KNeighborsClassifier(n_neighbors=1)
np.asarray(y_train).reshape(-1, 1)
knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')
           

pred = knn.predict(X_test)

#Predicting and evavluations 
#Let's evaluate our knn model.

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

## K = 1-40

error_rate = []
score_list = []
cv_list = []

fold = 5
for i in range(1,40):
    

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error = np.mean(pred_i != y_test)
    error_rate.append(error)
    score = knn.score(X_test, y_test)
    score_list.append(score)
    cv_scores = np.mean(cross_val_score(knn, X, Y, cv = fold)) 
    cv_list.append(cv_scores)
    print("K nearest neigbor = " + str(i))
    print("Naive score = " + str(score)) 
    print(str(fold)+" fold Cross validation score = " + str(cv_scores)) 

plt.figure(figsize=(10,6))
#plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plot1 = plt.figure(1)
plt.plot(range(1,20),score_list,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plot2 = plt.figure(2)
plt.plot(range(1,20),cv_list,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

    
## Check K = 1 again
#knn = KNeighborsClassifier(n_neighbors=1)
#knn.fit(X_train,y_train)
#pred = knn.predict(X_test)

#print('WITH K=1')
#print('\n')
#print(confusion_matrix(y_test,pred))
#print('\n')
#print(classification_report(y_test,pred))

## Now compare with K = 16
#knn = KNeighborsClassifier(n_neighbors=16)
#knn.fit(X_train,y_train)
#pred = knn.predict(X_test)

#print('WITH K=16')
#print('\n')
#print(confusion_matrix(y_test,pred))
#print('\n')
#print(classification_report(y_test,pred))






