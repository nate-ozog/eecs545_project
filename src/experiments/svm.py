"""
SVM Model Supervised Training, Validation, and Testing Experimental Code.
Author: Arman
"""
# imports
import os, random, math, time
import numpy as np 
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import data_primer # for data standardization preprocessing

# Constants
DATA_FILE = "../../data/data.npy" # make sure you extract data.zip
#OUTPUT_FILE = "../../Evaluation/svm_train.txt"

# Validation constant results
accuracy = []
precision = []
recall = []
detection_time = []
memory_usage = []

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
  return X, Y

class SVM_Model():
    """
    The SVM class model.
    Purpose: performs training, validation, and evaluation of an SVM classifier instance.
    """
    def __init__(self, c, kernel_func):
        """
        Initialize the SVM model.
        Input:
            C: The initial C value hyperparameter (regularization parameter)
            Kernel: The kernel function used (linear, poly, rbf, sigmoid)
        """
        self.C = c
        self.kernel = kernel_func
        self.model = SVC(C=c, kernel=kernel_func, decision_function_shape='ovr')
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
    X = X_trains # all 18 normalized features
    y = y_trains

    # for each driver, train and save the model
    model.fit(X, y)
    #model.save_model(i)

def validation(model, X_vals, y_vals):
    """
    Perform validation and testing.
    """
    #model = model.load_model(i)c
    X = X_vals
    y = y_vals

    # perform testing on sliding windows of size 50 for latency
    start = 0
    while start + 50 < len(X):
        time_start = time.time()
        preds = model.predict(X[start:start + 50])
        detection_time.append(time.time() - time_start)
        accuracy.append(accuracy_score(y[start:start+50], preds))
        precision.append(precision_score(y[start:start+50], preds, average='macro'))
        recall.append(recall_score(y[start:start+50], preds, average='macro'))
    return

def label_dataset(y):
    """
    Label datasets for multi-class classification.
    """
    num_classes = 3
    y_labeled = np.zeros(shape=y.shape)
    y_normalized = (y - y.min()) / (y.max() - y.min())
    for i in range(len(y)):
        if y[i] >= 0.67:
            y_labeled[i] = 2
        elif y[i] >= 0.34:
            y_labeled[i] = 1
        else:
            y_labeled[i] = 0

    return y_labeled

def split_dataset(X, y):
    """
    Split the dataset into training-validation.
    """
    y = label_dataset(y)
    train_idx = math.floor(len(X) * 0.8)
    #val_idx = math.floor(len(X) * 0.8)
    X_train = X[:train_idx]
    X_val = X[train_idx:]
    y_train = y[:train_idx]
    y_val = y[train_idx:]
    return X_train, X_val, y_train, y_val

def data_distribution(y):
    """
    Visualize the class distribution of the data to ensure no heavily skewed classes
    in the data split.
    """
    split_idx = math.floor(len(y) * 0.8)
    y_0 = y[:split_idx].tolist()
    y_1 = y[split_idx:].tolist()
    x = [1, 2, 3]
    width = 0.5

    # Plot the distribution of classes
    plt.figure()
    count_0 = y_0.count(0)
    count_1 = y_0.count(1)
    count_2 = y_0.count(2)
    plt.title("Training Data Class Distribution")
    plt.bar(x, [count_0, count_1, count_2], width)
    plt.savefig('training_dist.pdf', dpi=500)

    plt.figure()
    count_0 = y_1.count(0)
    count_1 = y_1.count(1)
    count_2 = y_1.count(2)
    plt.title("Testing Data Class Distribution")
    plt.bar(x, [count_0, count_1, count_2], width)
    plt.savefig('testing_dist.pdf', dpi=500)

    return

def output_results():
    # Output average results
    print("Average Accuracy: " + str(np.mean(accuracy)))
    print("Average Precision: " + str(np.mean(precision)))
    print("Average Recall: " + str(np.mean(recall)))
    print("Average Latency: " + str(np.mean(detection_time)))
    return


def main():
    svm_model = SVM_Model(10, 'rbf', random_state=42)
    _, XDrivers, XLabels, _, YDrivers = data_primer.standardizeDataDims()
    X, y = prepareData(XDrivers, YDrivers)
    #data_distribution(y)

    X_train, X_val, y_train, y_val = split_dataset(X, y)
    training(svm_model, X_train, y_train)
    validation(svm_model, X_val, y_val)
    output_results()
    return

if __name__=="__main__":
    main()