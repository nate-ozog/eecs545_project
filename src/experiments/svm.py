"""
SVM Model Supervised Training and Validation.
"""
# imports
import os, random, math, time
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as pyplot
import sklearn
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#import data_primer # temp, only for simplicity to import data standardization

# Constants
DATA_FILE = "../../data/data.npy" # make sure you extract data.zip
OUTPUT_FILE = "../../Evaluation/svm_train.txt"
MODEL_PATH = "../models/"

# Validation constant results
accuracy = []
precision = []
recall = []
detection_time = []
memory_usage = []

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
        self.model = SVC(C=c, kernel=kernel_func, gamma=1e-8, decision_function_shape='ovo')
        return
    
    def save_model(self, drive):
        # Save the classifier for the specific drive
        pkl_filename = str(drive) + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, os.path.join(MODEL_PATH, file))
        return
    
    def load_model(self, pkl_filename):
        # Load the classifier for the specific drive
        with open(pkl_filename, 'rb') as file:
            self.model = pickle.load(file)
        return
    
    def fit(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)
        return
    
    def predict(self, X):
        # Predict y values from X
        return self.model.predict(X)
    
def training(model, X_trains, y_trains, feature_names):
    """
    Training all 13 SVM models (one for each driver trip)
    """
    for i in range(0, 1):
        time_start = time.time()
        print("Training on driver " + str(i) + " dataset...")
        #print(feature_names)
        X = X_trains[0] # all 18 features
        y = y_trains[0]

        # normalize the features
        X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        # for each driver, train and save the model
        model.fit(X_normalized, y)
        print("Elapsed time after training driver " + str(i) + ": " + str(time.time() - time_start))
        #model.save_model(i)

def validation(model, X_vals, y_vals):
    """
    Perform validation testing on all 13 SVM models for further hyperparameter tuning (if needed).
    """
    for i in range(0, 1):
        #model = model.load_model(i)c
        X = X_vals[0]
        y = y_vals[0]

        # normalize the features
        X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        time_start = time.time()
        preds = model.predict(X_normalized)
        print("Elapsed validation time: " + str(time.time() - time_start))
        print("Validation Accuracy: " + str(accuracy_score(y, preds)))
        accuracy.append(accuracy_score(y, preds))
        #print("Baseline Validation Precision: " + str(precision_score(y, preds)))
        #print("Baseline Validation Recall: " + str(recall_score(y, preds)))
    return

"""def feature_extraction(X, y):
    extracted_feats = X
    new_output = []
    # sliding window approach
    start = 0
    end = 10
    mean = []
    std = []
    max_ = []
    min_ = []
    while start + 10 < len(X):
        # mean, std, max, min
        mean.append(X[:, [0]][start:end].mean())
        std.append(X[:, [0]][start:end].std())
        max_.append(X[:, [0]][start:end].max())
        min_.append(X[:, [0]][start:end].min())
        new_label = y[start:end].mean()
        if new_label >= 0.67:
            new_output.append(2)
        elif new_label >= 0.33:
            new_output.append(1)
        else:
            new_output.append(0)
        start += 1
        end += 1

    extracted_feats = np.zeros((np.array(mean).shape))
    extracted_feats = np.append(extracted_feats, np.array(mean), 1)
    extracted_feats = np.append(extracted_feats, np.array(std), 1)
    return np.array(extracted_feats), np.array(new_output)"""

def label_dataset(y):
    """
    Label datasets for multi-class classification.
    """
    num_classes = 3
    y_labeled = np.zeros(shape=y.shape)
    y_normalized = (y - y.min()) / (y.max() - y.min())
    #yRange = np.arange(num_classes + 1)
    #yRange = (yRange - yRange.min(axis=0)) / (yRange.max(axis=0) - yRange.min(axis=0))
    for i in range(len(y)):
        if y[i] >= 0.67:
            y_labeled[i] = 2
        elif y[i] >= 0.34:
            y_labeled[i] = 1
        else:
            y_labeled[i] = 0

    return y_labeled

def get_dataset(train_fold, val_fold):
    """
    Refer to data_primer.py in experiments/ for use on how the dataset is created.
    """
    _, XDrivers, XLabels, _, YDrivers = data_primer.standardizeDataDims()
    # for the sake of simplicity, there are 13 drivers indices 0-12
    X_trains = []
    y_trains = []
    X_vals = []
    y_vals = []

    # first, appropriately label the datasets
    for i in range(0, 13):
        YDrivers[i] = label_dataset(YDrivers[i])

    for i in range(0, 13):
        dataset_X = XDrivers[i]
        dataset_Y = YDrivers[i]
        X_train, X_val, y_train, y_val = split_dataset(dataset_X, dataset_Y, train_fold, val_fold)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_vals.append(X_val)
        y_vals.append(y_val)
        
    return X_trains, y_trains, X_vals, y_vals, XLabels

def split_dataset(X, y, train_fold, val_fold):
    """
    Split the dataset into training-validation.
    """
    train_idx = math.floor(len(X) * train_fold)
    val_idx = math.floor(len(X) * val_fold)
    X_train = X[:train_idx]
    X_val = X[train_idx:val_idx]
    y_train = y[:train_idx]
    y_val = y[train_idx:val_idx]
    return X_train, X_val, y_train, y_val

def plot_loss(model):
    """
    Plot the loss of the SVM model over time.
    """
    return


def main():
    svm_model = SVM_Model(10, "rbf")
    # for cross validation
    train = [0.2, 0.4, 0.6, 0.8]
    val = [0.4, 0.6, 0.8, 1.0]
    k = 4

    for i in range(k):
        X_trains, y_trains, X_vals, y_val, feature_names = get_dataset(train[i], val[i])
        training(svm_model, X_trains, y_trains, feature_names)
        validation(svm_model, X_vals, y_val)
    return

if __name__=="__main__":
    main()