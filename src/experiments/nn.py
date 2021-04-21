import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from torchsummary import summary
from sklearn.model_selection import KFold
from data_primer_modified import standardizeDataDims



# Seed all RNG sources
torch.manual_seed(545)
random.seed(545)
np.random.seed(545)



# Hyperparameters
learningRate = 0.01
momentum = 0.1
weightDecay = 0.01
dampening = 0.01
epochs = 65536
numClasses = 3
numFeatures = 18
H0 = 32
H1 = 32
latencyTestBatchSize = 500
latencyTestIters = 1000



class Network(torch.nn.Module):
  def __init__(self, D_in, H0, H1, D_out):
    super(Network, self).__init__()
    self.relu = torch.nn.ReLU()
    self.l1 = torch.nn.Linear(D_in, H0)
    self.l2 = torch.nn.Linear(H0, H1)
    self.l3 = torch.nn.Linear(H1, D_out)
  def forward(self, x):
    x = self.l1(x)
    x = self.relu(x)
    x = self.l2(x)
    x = self.relu(x)
    x = self.l3(x)
    return torch.nn.functional.log_softmax(x, dim=1)



def prepareData(XDrivers, YDrivers):
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
    """
    Feature extraction:
    Mean, Standard Deviation, Maximum, Minimum
    """
    # sliding window approach
    n = 0
    start = 0
    while start + 50 < len(X):
        n += 1
        start += 25
    extracted_feats = np.zeros((n, 1))
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

        extracted_feats = np.append(extracted_feats, np.array(mean).reshape(len(mean),1),1)
        extracted_feats = np.append(extracted_feats, np.array(std).reshape(len(std),1),1)
        extracted_feats = np.append(extracted_feats, np.array(max_).reshape(len(max_),1),1)
        extracted_feats = np.append(extracted_feats, np.array(min_).reshape(len(min_),1),1)

    extracted_feats = extracted_feats[:, 1:]
    return np.array(extracted_feats)



def new_label(y):
    """
    New label for a vector of extracted features.
    """
    new_output = []
    # sliding window approach
    start = 0
    end = 50
    # multi-class classification
    while start + 50 < len(y):
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



def run(device, XDrivers, YDrivers):
  numDrivers = len(XDrivers)

  # LODO validation variables
  numCorrect = 0
  numSamples = 0
  PRMat = np.zeros((numClasses, numClasses))

  # Feature extraction.
  for i in range(numDrivers):
    XDrivers[i] = feature_extraction(XDrivers[i])
    YDrivers[i] = new_label(YDrivers[i])

  # LODO cross validation
  avgLossMat = np.zeros((numDrivers, epochs))
  for i in range(numDrivers):
    XTrain, YTrain, XTest, YTest = getLODOIterData(XDrivers, YDrivers, i)

    # Create new neural network and send to device
    net = Network(XTrain.shape[1], H0, H1, numClasses)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=momentum, dampening=dampening, weight_decay=weightDecay)
    lossFunction = torch.nn.CrossEntropyLoss()

    # Send testing data to device
    XTrain = torch.from_numpy(XTrain).float().to(device)
    YTrain = torch.from_numpy(YTrain).long().to(device)

    # Run training
    for j in range(epochs):
      XTrainBatch = torch.autograd.Variable(XTrain)
      YTrainBatch = torch.autograd.Variable(YTrain)
      optimizer.zero_grad()
      netOut = net(XTrainBatch)
      loss = lossFunction(netOut, YTrainBatch)
      loss.backward()
      optimizer.step()
      avgLossMat[i][j] = loss.item()
      if not j % 1024:
        print("Epoch (", j+1, "/", epochs, ") - Avg Loss:", avgLossMat[i][j])

    # Run testing
    XTest = torch.from_numpy(XTest).float().to(device)
    YTest = torch.from_numpy(YTest).long().to(device)
    XTestBatch = torch.autograd.Variable(XTest)
    YTestBatch = torch.autograd.Variable(YTest)
    netOut = net(XTestBatch)
    _, netPreds = netOut.max(1)
    numCorrect += (netPreds == YTestBatch).sum()
    numSamples += netPreds.size(0)

    # Fill our precision/recall matrix
    for j in range(netPreds.shape[0]):
      predicted = netPreds[j].item()
      actual = YTestBatch[j].item()
      PRMat[predicted][actual] += 1
    PRMat = PRMat.astype(int)

    # Print current accuracy after one LODO iteration
    print("LODO Itr", i, "Complete - Current accuracy:", numCorrect.item() / numSamples)

  # After all LODO iterations do data analysis
  valAcc = numCorrect.item() / numSamples
  print("Average Accuracy =", valAcc)

  # Compute average precision, recall, and F1
  prColSum = PRMat.sum(axis=0) + 1
  prRowSum = PRMat.sum(axis=1) + 1
  precisSum = 0.0
  recallSum = 0.0
  for i in range(numClasses):
    precisSum += PRMat[i][i] / prColSum[i]
    recallSum += PRMat[i][i] / prRowSum[i]
  avgPrecis = precisSum / numClasses
  avgRecall = recallSum / numClasses
  avgF1 = 2 * ((avgPrecis * avgRecall) / (avgPrecis + avgRecall))
  print("Average Validation Precision =", avgPrecis)
  print("Average Validation Recall =", avgRecall)
  print("Average Validation F1 =", avgF1)

  # Return
  return



def measureLatency(device, XDrivers, YDrivers):
  # Get latency test data and create the net.
  XLat, _, _, _ = getLODOIterData(XDrivers, YDrivers, 0)
  net = Network(4 * numFeatures, H0, H1, numClasses).to(device)
  optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=momentum, dampening=dampening, weight_decay=weightDecay)
  lossFunction = torch.nn.CrossEntropyLoss()
  XLatPreFE = XLat[0:latencyTestBatchSize]

  # Get average prediction time across a bunch of runs.
  # This has to include feature extraction time for the
  # batch as well, because this would have to be done
  # in real-time as well.
  startTime = time.time()
  for i in range(latencyTestIters):
    XLat = feature_extraction(XLatPreFE)
    XLat = torch.from_numpy(XLat).float().to(device)
    XTestBatch = torch.autograd.Variable(XLat)
    netOut = net(XTestBatch)
    _, netPreds = netOut.max(1)
  endTime = time.time()
  runtime = (endTime - startTime) / latencyTestBatchSize
  print("Average Latency (s) =", runtime)
  return



def main():
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  _, XDrivers, _, _, YDrivers = standardizeDataDims()
  XDrivers, _, YDrivers, _ = prepareData(XDrivers, YDrivers)
  run(device, XDrivers, YDrivers)
  measureLatency(device, XDrivers, YDrivers)
  return



if __name__=="__main__":
    main()
















# def featureImportance(device, XDrivers, XLabels, YDrivers):
#   numDrivers = len(XDrivers)

#   for i in range(numDrivers):
#     XTrain, YTrain, XTest, YTest = getLODOIterData(XDrivers, YDrivers, i)

#     # Create new neural network and send to device
#     net = Network(XTrain.shape[1], H0, H1, numClasses)
#     net = net.to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=momentum, dampening=dampening, weight_decay=weightDecay)
#     lossFunction = torch.nn.CrossEntropyLoss()

#     # Send testing data to device
#     XTrain = torch.from_numpy(XTrain).float().to(device)
#     YTrain = torch.from_numpy(YTrain).long().to(device)

#     # Run training
#     for j in range(epochs):
#       XTrainBatch = torch.autograd.Variable(XTrain)
#       YTrainBatch = torch.autograd.Variable(YTrain)
#       optimizer.zero_grad()
#       netOut = net(XTrainBatch)
#       loss = lossFunction(netOut, YTrainBatch)
#       loss.backward()
#       optimizer.step()
#       if not j % 32:
#         print("In featureImportance() still training...")

#     # Run testing with no randomization.
#     XTestNormal = torch.from_numpy(XTest).float().to(device)
#     YTestNormal = torch.from_numpy(YTest).long().to(device)
#     XTestBatchNormal = torch.autograd.Variable(XTestNormal)
#     YTestBatchNormal = torch.autograd.Variable(YTestNormal)
#     netOut = net(XTestBatchNormal)
#     _, netPreds = netOut.max(1)
#     numCorrectNormal = (netPreds == YTestBatchNormal).sum()
#     numSamplesNormal = netPreds.size(0)
#     valAccNormal = numCorrectNormal.item() / numSamplesNormal

#     # Iterate through all features and get accuracy with values randomized
#     valAccFeatures = np.zeros((numFeatures,))
#     for i in range(numFeatures):
#       N = XTest.shape[0]
#       XTestI = XTest
#       # Randomize the i-th column of XTestI
#       XTestI[:,i] = np.squeeze(np.random.rand(N, 1))
#       XTestI = torch.from_numpy(XTestI).float().to(device)
#       YTestI = torch.from_numpy(YTest).long().to(device)
#       XTestBatchI = torch.autograd.Variable(XTestI)
#       YTestBatchI = torch.autograd.Variable(YTestI)
#       netOut = net(XTestBatchI)
#       _, netPreds = netOut.max(1)
#       numCorrectI = (netPreds == YTestBatchI).sum()
#       numSamplesI = netPreds.size(0)
#       valAccFeatures[i] = numCorrectI.item() / numSamplesI

#     # Just leave after 1 fold, for feature importance that is fine
#     break

#   # Compute the normalized feature importance
#   valAccFeaturesStandardized = np.zeros((numFeatures,))
#   for i in range(numFeatures):
#     valAccFeaturesStandardized[i] = valAccNormal - valAccFeatures[i]
#   vafs = valAccFeaturesStandardized
#   valAccFeaturesErr = 1 - vafs
#   vafs = (vafs.max(axis=0) - vafs) / (vafs.max(axis=0) - vafs.min(axis=0))
#   normalizedFeatureImportance = vafs

#   # Create a bar graph for normalized feature importance
#   plt.bar(np.arange(numFeatures), normalizedFeatureImportance, align='center', alpha=0.5)
#   plt.xticks(np.arange(numFeatures), XLabels, rotation='vertical')
#   plt.ylabel('Normalized Feature Importance')
#   plt.title('Normalized Feature Importance')
#   plt.tight_layout()
#   plt.savefig('../../data/nnNormalizedFeatureImportance.png')
#   plt.close()

#   # Create a bar graph for feature importance
#   plt.bar(np.arange(numFeatures), valAccFeaturesErr, align='center', alpha=0.5)
#   plt.xticks(np.arange(numFeatures), XLabels, rotation='vertical')
#   plt.ylabel('Feature Importance')
#   plt.title('Feature Importance')
#   plt.tight_layout()
#   plt.savefig('../../data/nnFeatureImportance.png')
#   plt.close()

#   # Return
#   return



# def datasetDistribution(Y):
#   dist = np.zeros((numClasses,))
#   for i in range(numClasses):
#     cnt = Y[Y == i]
#     dist[i] = cnt.shape[0]
#   distLables = []
#   for i in range(numClasses):
#     distLables.append(str(i + 1))
#   plt.bar(np.arange(numClasses), dist, align='center', alpha=0.5)
#   plt.xticks(np.arange(numClasses), distLables, rotation='vertical')
#   plt.ylabel('Count')
#   plt.title('Stress Level (Scale of 1 to 3)')
#   plt.tight_layout()
#   plt.savefig('../../data/affectiveROADStressDistribution.png')
#   plt.close()
#   return



# def measureLatency(device, X):
#   # Get a random set of data of latency batch size
#   p = np.random.permutation(latencyTestBatchSize)
#   X = X[p]

#   # Send testing to device
#   X = torch.from_numpy(X).float().to(device)
#   XBatch = torch.autograd.Variable(X)

#   # Create new neural network and send to device
#   net = Network(X.shape[1], H0, H1, numClasses)
#   net = net.to(device)

#   # Time the model average across a bunch of iterations
#   iters = 100000
#   startTime = time.time_ns()
#   for _ in range(iters):
#     netOut = net(XBatch)
#     _, netPreds = netOut.max(1)
#   endTime = time.time_ns()
#   runtime = (endTime - startTime) / iters
#   print(latencyTestBatchSize, "predictions made in", runtime, "nanoseconds")

#   # Return
#   return








