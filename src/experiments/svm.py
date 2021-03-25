"""
SVM Prelim Experiments Code.
Author: Arman
"""
# imports
import os, random, math, time
import pickle
import numpy as np 
import matplotlib.pyplot as plt

import data_primer
from data_primer import standardizeDataDims

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Constants
DATA_FILE = "../../data/data.npy"
error_rate = []
score_list = []
cv_list = []
accuracy = []
precision = []
recall = []
detection_time = []

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



def validation(num_folds):
    # Prepare data
    _, XDrivers, _, _, YDrivers = standardizeDataDims()
    NumClasses = 10
    X, Y = prepareData(XDrivers, YDrivers, NumClasses)
    X,Y = shuffle(X, Y)

    error_rate = []
    score_list = []
    cv_list = []

    nfold = num_folds
            
    error = 0
    score = 0
        
    kf = KFold(n_splits=nfold)
    for trainIdx, testIdx in kf.split(X):
        # Get data for K-iteration
        XTrain = X[trainIdx]
        YTrain = Y[trainIdx]
        XTest = X[testIdx]
        YTest = Y[testIdx]

        # model
        svm = SVC(C=10, kernel='rbf', cache_size=7000)

        # train, validate and output results
        print("Training...")
        svm.fit(XTrain,YTrain)
        print("Validating...")
        time_start = time.time()
        y_pred = svm.predict(XTest)
        detection_time.append(time.time() - time_start)

        print("Getting results...")
        error.append(np.mean(y_pred != YTest))
        accuracy.append(accuracy_score(YTest, y_pred))
        precision.append(precision_score(YTest, y_pred, average=None))
        recall.append(recall_score(YTest, y_pred, average=None))
        print("Baseline Validation Accuracy: " + str(accuracy_score(YTest, y_pred)))
        print("Baseline Validation Precision: " + str(precision_score(YTest, y_pred, average=None)))
        print("Baseline Validation Recall: " + str(recall_score(YTest, y_pred, average=None)))

    # Print the cv accuracy, precision, recall
    print("Average Validation Accuracy: " + str(np.mean(accuracy)))
    print("Average Validation Precision: " + str(np.mean(precision)))
    print("Average Validation Recall: " + str(np.mean(recall)))
    print("Average Detection Time: " + str(np.mean(detection_time)))


def main():
    num_folds = 5
    validation(num_folds)
    return

if __name__=="__main__":
    main()