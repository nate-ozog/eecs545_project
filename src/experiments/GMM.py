# -*- coding: utf-8 -*-
"""
eecs 545 project, GMM experiments

"""
import numpy as np
import math
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from data_primer_mod import standardizeDataDims
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from memory_profiler import profile
import time
import tqdm
# from resources import*

def prepareData(XDrivers, YDrivers, numClasses):
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

def feature_extraction(X, numCol):
    """
    Feature extraction:
    Mean, Standard Deviation, Maximum, Minimum
    """
    # sliding window approach
    n = math.floor(len(X)/25)-1
    extracted_feats = np.zeros((n,1)) # 9727 for 40 window, 7781 for 50 window
    
    for i in range(numCol):
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

    return np.array(extracted_feats)[:,1:]

def new_label(y):
    """
    New label for a vector of extracted features.
    """
    new_output = []
    # sliding window approach
    start = 0
    end = 50

    while start + 50 < len(y):
        new_label = y[start:end].mean()
        if new_label >= 0.45:
            new_output.append(1)
        # elif new_label >= 0.55:
        #     new_output.append(1)
        # elif new_label >= 1.15:
        #     new_output.append(2)
        # elif new_label >= 0.20:
        #     new_output.append(1)
        else:
            new_output.append(0)
        start += 25
        end = start + 50

    return np.array(new_output)

def pca_reduce(n_comp, X_in):
    pca = PCA(n_components = n_comp)
    return pca.fit_transform(X_in)

@profile    
def NC(numClass, Y_true, pred):
    countArr = np.zeros((numClass, numClass))
    rowSum = [0] * numClass
    Y_pred = np.array([-1] * Y_true.shape[0])
    for i in range(numClass):
        binaryArr = pred == i
        for j in range(len(binaryArr)):
            if binaryArr[j] == True:
                countArr[i, int(Y_true[j])] += 1
        maxClass = np.argmax(countArr[i,:])
        if sum(countArr[i,:]) != 0:
            percent = max(countArr[i,:]) / sum(countArr[i,:]) * 100
        else:
            percent = 0
        rowSum[i] = sum(countArr[i,:])
        Y_pred[binaryArr] = maxClass
        # print("For class {}, it maps to stress level {} with {:.2f} % correctness".format(i, maxClass, percent))
    
    prec = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_true, Y_pred, average='macro')
    accu = accuracy_score(Y_true, Y_pred)
    # print("Precision: ", prec)
    # print("Recall: ", recall)
    # print("Accuracy: ", accu) 
    # print("The pseudo confusion matrix: ", countArr, '\n')  
    # print("The sum of rows: ", rowSum)    
    return prec, recall, accu

def NC_temp(numClass, Y_true, pred):
    countArr = np.zeros((numClass, numClass))
    rowSum = [0] * numClass
    Y_pred = np.array([-1] * Y_true.shape[0])
    for i in range(numClass):
        binaryArr = pred == i
        for j in range(len(binaryArr)):
            if binaryArr[j] == True:
                countArr[i, int(Y_true[j])] += 1
        maxClass = np.argmax(countArr[i,:])
        if sum(countArr[i,:]) != 0:
            percent = max(countArr[i,:]) / sum(countArr[i,:]) * 100
        else:
            percent = 0
        rowSum[i] = sum(countArr[i,:])
        Y_pred[binaryArr] = maxClass
    prec = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_true, Y_pred, average='macro')
    accu = accuracy_score(Y_true, Y_pred)
    return prec, recall, accu

@profile
def GMM(X_tr, X_ts, numClass):
    gmm = GaussianMixture(n_components = numClass, max_iter = 250, n_init = 3, random_state = 147)
    gmm.fit(X_tr)
    pred = gmm.predict(X_ts)
    return pred

def GMM_temp(X_tr, X_ts, numClass):
    gmm = GaussianMixture(n_components = numClass, max_iter = 250, n_init = 3, random_state = 147)
    gmm.fit(X_tr)
    pred = gmm.predict(X_ts)
    return pred

def singleRun(numClass, X_in, Y_in):
    Y_pred = GMM(X_in, X_in, numClass)
    # check the shape of prediction
    print("The shape of prediction: ", Y_pred.shape, "\n")
    return NC(numClass, Y_in, Y_pred)

def kFoldNC(numClass, X_in, Y_in, FOLDS, XDrivers, YDrivers, LODO):
    if not LODO:
        prec_arr, recall_arr, accu_arr = [], [], []
        skf = StratifiedKFold(n_splits = FOLDS)
        for fold, (tr_index, ts_index) in enumerate(skf.split(X_in, Y_in)):
            X_tr, X_ts = X_in[tr_index], X_in[ts_index]
            _, Y_ts = Y_in[tr_index], Y_in[ts_index]
            Y_pred = GMM(X_tr, X_ts, numClass)
            curr_prec, curr_recall, curr_accu = NC(numClass, Y_ts, Y_pred)
            prec_arr.append(curr_prec)
            recall_arr.append(curr_recall)
            accu_arr.append(curr_accu)
    else:
        prec_arr, recall_arr, accu_arr = [], [], []
        numDrivers = len(XDrivers)
        for i in range(numDrivers):
            X_tr, _, X_ts, Y_ts = getLODOIterData(XDrivers, YDrivers, i)
            Y_pred = GMM(X_tr, X_ts, numClass)                                  
            curr_prec, curr_recall, curr_accu = NC(numClass, Y_ts, Y_pred)
            prec_arr.append(curr_prec)
            recall_arr.append(curr_recall)
            accu_arr.append(curr_accu)
            if i == numDrivers - 1:
                numItr = 1000
                startTime = time.time_ns()
                for _ in tqdm.tqdm(range(numItr)):
                    some_i = np.random.randint(len(X_in))
                    end_i = some_i + 50
                    sample_tr = X_tr[some_i:end_i,:] if end_i < len(X_tr) else X_tr[0:50,:]
                    sample_ts = X_ts[some_i:end_i] if end_i < len(X_ts) else X_ts[:50]
                    sample_label_test = Y_ts[some_i:end_i] if end_i < len(Y_ts) else Y_ts[:50]
                    Y_pred_temp = GMM_temp(sample_tr, sample_ts, numClass)
                    NC_temp(numClass, sample_label_test, Y_pred_temp)
                endTime = time.time_ns()
                runtime = (endTime - startTime) / numItr
                print("average running time of GMM & NC is around: ", runtime, '\n')                   
    return np.mean(prec_arr), np.mean(recall_arr), np.mean(accu_arr)


def main():
  _, XDrivers, _, _, YDrivers = standardizeDataDims()
  #####################################
  ##################################### Modify here to change num of classes
  NumClasses = 2
  #####################################
  #####################################
  XDrivers, X, YDrivers, Y = prepareData(XDrivers, YDrivers, NumClasses)   
  
  print("The shape of preprocessed data: ", X.shape, Y.shape, "\n")  
  
  # Use PCA to reduce the dimension of the preprocessed data
  n_comp_pca = 2     
  X_pca = pca_reduce(n_comp_pca , X)
  print("The shape of PCA data: ", X_pca.shape, "\n")
  
  # Use feature extraction to modify (raw or PCA) data & generate new label
  X_fe = feature_extraction(X_pca, n_comp_pca)
  Y = new_label(Y)  
  # check the shape
  print("The shape of feature extraction data: \n", X_fe.shape, Y.shape, "\n")

  # single run
  # singleRun(NumClasses, X_fe, Y)
  
  # K-fold
  FOLDS = 10
  LODO = True
  avgPrec, avgRec, avgAccu = kFoldNC(NumClasses, X_fe, Y, FOLDS, XDrivers, YDrivers, LODO)
  print("avg precision: ", avgPrec, "\navg recall: ", avgRec, "\navg accuracy: ", avgAccu)
  
if __name__=="__main__":
    main()