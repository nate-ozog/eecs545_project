import os
import numpy as np
import matplotlib.pyplot as pyplot
import sklearn
import scipy.interpolate as interp
from sklearn.preprocessing import StandardScaler, normalize



DATA_PATH = "../../data/data.npy"



# Standardizes driver data to fit into nice 2D numpy matrices
# X:          N x D matrix of data
# XColLables: D sized list of labels for each column of X
# Y:          N sized vector of subjective stress metric
def standardizeDriver(data, labels):
    keys = data.keys()
    Y = data["subData"]
    N = Y.shape[0]
    XCols = []
    XColLables = []
    # Go through all the keys in the dataset
    for k in keys:
        # If the key is not our subject label data then continue
        if (k != "subData"):
            kN = data[k].shape[0]
            # If vector, make N x 1 (needs to be 2D to make later stuff work)
            if data[k].ndim == 1:
                data[k] = data[k].reshape(len(data[k]), 1)
            kD = data[k].shape[1]
            # Interpolate the array to fit subData
            for d in range(kD):
                featureData = data[k].T[d]
                featureDataInterp = interp.interp1d(np.arange(featureData.size), featureData)
                featureDataStretch = featureDataInterp(np.linspace(0, featureData.size - 1, Y.size))
                featureData = featureDataStretch
                featureDataLabel = labels[k][d]
                XCols.append(featureData)
                XColLables.append(featureDataLabel)
    # Go through our resized data and package into 2D array
    X = np.zeros((N, len(XCols)))
    for i in range(X.shape[1]):
        X[:,i] = XCols[i]
    return X, XColLables, Y



# Takes in raw preprocessed data and performs data interpolation on
# every drive to standardize the data into nice 2D numpy matrices
# X:        N x D combined array of all driver data
# XDrivers: List 0->12 drivers data, each is an N[i] x D numpy array
# XLabels:  Column labels for all X/XDrivers[i] data
# Y:        N combined array of all driver stress metric data
# YDrivers: List 0->12 driver stress metric data, each is an N[i] numpy vector
def standardizeDataDims():
    rawData = np.load(DATA_PATH, allow_pickle=True)
    drivers = ["AD1", "BK1", "EK1", "GM1", "GM2", "KSG1", "MT1", "NM1", "NM2", "NM3", "RY1", "RY2", "SJ1"]
    XDrivers = []
    YDrivers = []
    N = 0
    D = 0
    # Standardize all driver data
    for driver in drivers:
        XDriver, XLabels, YDriver = standardizeDriver(rawData[()][driver][0], rawData[()][driver][1])
        XDrivers.append(XDriver)
        YDrivers.append(YDriver)
        N += XDriver.shape[0]
        D = XDriver.shape[1]
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
    return X, XDrivers, XLabels, Y, YDrivers



# Driver code
def main():
    # Read documentation of "standardizeDataDims()" if confused about return values
    X, XDrivers, XLabels, Y, YDrivers = standardizeDataDims()
    print(XLabels)



if __name__=="__main__":
    main()


