import os
import numpy as np
import matplotlib.pyplot as pyplot
import sklearn
import scipy.interpolate as interp
from sklearn.preprocessing import StandardScaler, normalize
from scipy import stats
import scipy.signal as scisig
import scipy.stats


DATA_PATH = "./data.npy"
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}

# Filter of EDA
def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

# Filter of Acc
def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)

def get_net_accel(data):
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))

# Standardizes driver data to fit into nice 2D numpy matrices
# X:          N x D matrix of data
# XColLables: D sized list of labels for each column of X
# Y:          N sized vector of subjective stress metric
def standardizeDriver(data, labels, num):
    keys = data.keys()
    Y = data["subData"]
    
    if num == 0:
        Y = Y[1:11604]
    if num == 1:
        Y = Y[1:12588]
    if num == 2:
        Y = Y[1:13564]
    if num == 3:
        Y = Y[128:12500]
    if num == 4:
        Y = Y[5548:18888]
    if num == 5:
        Y = Y[1:11916]
    if num == 6:
        Y = Y[1:13320]
    if num == 7:
        Y = Y[1:12464]
    if num == 8:
        Y = Y[1:13088]
    if num == 9:
        Y = Y[5488:18572]
    if num == 10:
        Y = Y[1:11656]
    if num == 11:
        Y = Y[1:11628]
    if num == 12:
        Y = Y[5124:18472]
    
    N = Y.shape[0]
    XCols = []
    XColLables = []
    # Go through all the keys in the dataset
    for k in keys:
        # preprocess the data using filter:
        if (k == 'rEDA' or k == 'lEDA'):
            data[k] = butter_lowpass_filter(data[k], 1.0, fs_dict['EDA'], 6)
        
        if ( k == 'rightACC' or k == 'leftACC'):
            kD = data[k].shape[1]
            for d in range(kD):
                data[k][:,d] = filterSignalFIR(data[k][:,d])     
        
        if (k != "subData"):
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
    i = 0
    for driver in drivers:
        XDriver, XLabels, YDriver = standardizeDriver(rawData[()][driver][0], rawData[()][driver][1],i)
        XDrivers.append(XDriver)
        YDrivers.append(YDriver)
        N += XDriver.shape[0]
        D = XDriver.shape[1]
        i = i+1
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



if __name__=="__main__":
    main()


