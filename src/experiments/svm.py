"""
SVM Model Supervised Training, Validation, and Testing Experimental Code.
Author: Arman
"""
# TODO: Performance results and visualizations for paper

# imports
import os, random, math, time
import numpy as np 
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import resource
import data_primer # for data standardization preprocessing

# Constants
DATA_FILE = "../../data/data.npy" # make sure you extract data.zip
OUTPUT_DIR = "../../Evaluation/"

# Validation constant results
all_acc = []
all_prec = []
all_rec = []
all_f1 = []
detection_time = []
memory_usage = []

# for LODO iterations
accuracy = []
precision = []
recall = []
f_1 = []

def prepareData(XDrivers, YDrivers): # taken from Nathan's nn.py code
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

  numClasses = 5 # play around to see what gives the best results

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
  return X, XDrivers, Y, YDrivers

def getLODOIterData(XDrivers, YDrivers, LODOIdx): # from Nathan's rnn.py
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

class SVM():
    def __init__(self, c, kernel_func):
        """
        Initialize the SVM model.
        Input:
            C: The initial C value hyperparameter (regularization parameter)
            Kernel: The kernel function used (linear, poly, rbf, sigmoid)
        """
        self.C = c
        self.kernel = kernel_func
        self.model = SVC(C=c, kernel=kernel_func, decision_function_shape='ovr', random_state=42)
        return
    
    def fit(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)
        return
    
    def predict(self, X):
        # Predict y values from X
        return self.model.predict(X)
    
def training(model, X_trains, y_trains):
    """
    Training SVM model.
    """
    #print(feature_names)
    X = feature_extraction(X_trains) # all 18 normalized features
    y = new_label(y_trains)

    # for each driver, train and save the model
    model.fit(X, y)

def validation(model, X_vals, y_vals):
    """
    Perform validation and testing.
    """
    # perform testing on sliding windows of size 50 for latency
    start = 0
    # window sizes of 500 samples
    while start + 500 < len(X_vals):
        memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        y = new_label(y_vals[start:start+500])
        time_start = time.time()
        X = feature_extraction(X_vals[start:start+500])
        memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        preds = model.predict(X)
        detection_time.append(time.time() - time_start)
        memory_usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        accuracy.append(accuracy_score(y, preds))
        precision.append(precision_score(y, preds, average='macro'))
        recall.append(recall_score(y, preds, average='macro'))
        f_1.append(f1_score(y, preds, average='macro'))
        all_acc.append(accuracy_score(y, preds))
        all_prec.append(precision_score(y, preds, average='macro'))
        all_rec.append(recall_score(y, preds, average='macro'))
        all_f1.append(f1_score(y, preds, average='macro'))
        start += 500
    return

def feature_extraction(X):
    """
    Feature extraction:
    Mean, Standard Deviation, Maximum, Minimum
    """
    # sliding window approach
    for i in range(18):
        start = 0
        end = 50
        mean = []
        std = []
        max_ = []
        min_ = []
        n = 0
        while start + 50 < len(X):
            # mean, std, max, min
            mean.append(X[:, [i]][start:end].mean())
            std.append(X[:, [i]][start:end].std())
            max_.append(X[:, [i]][start:end].max())
            min_.append(X[:, [i]][start:end].min())
            start += 25
            end = start + 50
            n += 1

        extracted_feats = np.zeros((n, 1)) 
        extracted_feats = np.append(extracted_feats, np.array(mean).reshape(len(mean),1),1)
        extracted_feats = np.append(extracted_feats, np.array(std).reshape(len(std),1),1)
        extracted_feats = np.append(extracted_feats, np.array(max_).reshape(len(max_),1),1)
        extracted_feats = np.append(extracted_feats, np.array(min_).reshape(len(min_),1),1)

    return np.array(extracted_feats)

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
        if new_label >= 0.67:
            new_output.append(2)
        elif new_label >= 0.33:
            new_output.append(1)
        else:
            new_output.append(0)
        start += 25
        end = start + 50

    return np.array(new_output)

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

def write_results_to_file():
    # Write performance results to txt file
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    f = open(os.path.join(OUTPUT_DIR, "svm.txt"), "w")
    f.write("Average across all LODO folds\n")
    f.write("Average Accuracy: " + str(np.mean(all_acc)) + "\n")
    f.write("Std Accuracy: " + str(np.std(all_acc)) + "\n")
    f.write("Average Precision: " + str(np.mean(all_prec)) + "\n")
    f.write("Std Precision: " + str(np.std(all_prec)) + "\n")
    f.write("Average Recall: " + str(np.mean(all_rec)) + "\n")
    f.write("Std Recall: " + str(np.std(all_rec)) + "\n")
    f.write("Average F1: " + str(np.mean(all_rec)) + "\n")
    f.write("Std F1: " + str(np.std(all_rec)) + "\n")
    f.write("Average Latency: " + str(np.mean(detection_time)) + "\n")
    f.write("Average Memory Usage: " + str(np.mean(memory_usage)) + "\n") 
    f.close()
    return


def main():
    _, XDrivers, XLabels, _, YDrivers = data_primer.standardizeDataDims()
    X, XDrivers, y, ydrivers = prepareData(XDrivers, YDrivers)

    # Leave One Driver Out (LODO)
    num_drivers = len(XDrivers)
    for i in range(0):
        print("Driver " + str(i) + " left out...")
        svm_model = SVM(10, 'rbf')
        X_train, y_train, X_val, y_val = getLODOIterData(XDrivers, YDrivers, i)
        print("Training...")
        training(svm_model, X_train, y_train)
        print("Validating on Driver " + str(i))
        validation(svm_model, X_val, y_val)
        accuracy.clear()
        precision.clear()
        recall.clear()
        f_1.clear()
    # print results and write same results to txt file
    output_average_results()
    write_results_to_file()
    return

if __name__=="__main__":
    main()