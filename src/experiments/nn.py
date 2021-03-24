import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torchsummary import summary
from sklearn.model_selection import KFold
from data_primer import standardizeDataDims



# Hyperparameters
learningRate = 0.01
momentum = 0.9
epochs = 256
batchSize = 64
nfold = 10
numClasses = 10
numFeatures = 18
latencyTestBatchSize = 50



class Network(torch.nn.Module):
  def __init__(self, D_in, H0, H1, H2, D_out):
    super(Network, self).__init__()
    self.l1 = torch.nn.Linear(D_in, H0)
    self.relu = torch.nn.ReLU()
    self.l2 = torch.nn.Linear(H0, H1)
    self.relu = torch.nn.ReLU()
    self.l3 = torch.nn.Linear(H1, H2)
    self.relu = torch.nn.ReLU()
    self.l4 = torch.nn.Linear(H2, D_out)
  def forward(self, x):
    x = self.l1(x)
    x = self.relu(x)
    x = self.l2(x)
    x = self.relu(x)
    x = self.l3(x)
    x = self.relu(x)
    x = self.l4(x)
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



def run(device, X, Y):
  # Shuffle X and Y data to diversify gradient descent between drivers
  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = Y[p]

  # K-Fold validation variables
  numCorrect = 0
  numSamples = 0
  PRMat = np.zeros((numClasses, numClasses))

  # K-Fold iterations
  nfold = 10
  avgLossMat = np.zeros((nfold, epochs))
  kItr = 0
  kf = KFold(n_splits=nfold)
  for trainIdx, testIdx in kf.split(X):
    # Get data for K-iteration
    XTrain = X[trainIdx]
    YTrain = Y[trainIdx]
    XTest = X[testIdx]
    YTest = Y[testIdx]

    # Create new neural network and send to device
    net = Network(X.shape[1], 128, 256, 128, numClasses)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
    lossFunction = torch.nn.CrossEntropyLoss()

    # Send testing data to device
    XTrain = torch.from_numpy(XTrain).float().to(device)
    YTrain = torch.from_numpy(YTrain).long().to(device)

    # Run training, suffle data after every epoch
    for i in range(epochs):
      p = np.random.permutation(XTrain.shape[0])
      XTrain = XTrain[p]
      YTrain = YTrain[p]
      numLoss = 0
      lossSum = 0
      # Run batches for epoch
      for j in range(0, XTrain.shape[0], batchSize):
        XMiniBatch = torch.autograd.Variable(XTrain[j:j + batchSize])
        YMiniBatch = torch.autograd.Variable(YTrain[j:j + batchSize])
        optimizer.zero_grad()
        netOut = net(XMiniBatch)
        loss = lossFunction(netOut, YMiniBatch)
        loss.backward()
        optimizer.step()
        lossSum += loss.item()
        numLoss += 1
      # Store to our average loss matrix
      avgLossMat[kItr][i] = lossSum / numLoss
      # Print average loss every few iterations
      if i % 16 == 0:
        print("Average loss:", avgLossMat[kItr][i])

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
    for i in range(netPreds.shape[0]):
      predicted = netPreds[i].item()
      actual = YTestBatch[i].item()
      PRMat[predicted][actual] += 1
    PRMat = PRMat.astype(int)

    # Print current accuracy as of this k-iter, update k-iter
    print("Current accuracy:", numCorrect.item() / numSamples)
    kItr += 1

  # After all k-fold validation do data analysis
  valAcc = numCorrect.item() / numSamples
  avgLoss = avgLossMat.mean(axis=0)
  print("Final accuracy:", valAcc)

  # Compute average precision and recall
  prColSum = PRMat.sum(axis=0)
  prRowSum = PRMat.sum(axis=1)
  precisSum = 0.0
  recallSum = 0.0
  for i in range(numClasses):
    precisSum += PRMat[i][i] / prColSum[i]
    recallSum += PRMat[i][i] / prRowSum[i]
  avgPrecis = precisSum / numClasses
  avgRecall = recallSum / numClasses
  print("Average precision:", avgPrecis)
  print("Average recall:", avgRecall)

  # Save loss figure, val-acc, precision, and recall if better than prior solution
  bestValAcc = np.load("../../data/nnBestValAcc.npy")
  if (valAcc > bestValAcc):
    print("Better than prior solution, saving")
    plt.plot(avgLoss)
    plt.xlabel('Epoch #')
    plt.ylabel('Average Loss')
    plt.title('Average Loss over Epochs')
    plt.tight_layout()
    plt.savefig('../../data/nnAvgLoss.png')
    plt.close()
    np.save("../../data/nnBestValAcc.npy", valAcc)
    np.save("../../data/nnBestValAccAvgPrecision.npy", avgPrecis)
    np.save("../../data/nnBestValAccAvgRecall.npy", avgRecall)
  else:
    print("Not better than prior solution")

  # Return
  return



def featureImportance(device, X, XLabels, Y):
  # Shuffle X and Y data to diversify gradient descent between drivers
  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = Y[p]

  # K-Fold iterations
  kf = KFold(n_splits=nfold)
  for trainIdx, testIdx in kf.split(X):
    # Get data for K-iteration
    XTrain = X[trainIdx]
    YTrain = Y[trainIdx]
    XTest = X[testIdx]
    YTest = Y[testIdx]

    # Create new neural network and send to device
    net = Network(X.shape[1], 128, 256, 128, numClasses)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
    lossFunction = torch.nn.CrossEntropyLoss()

    # Send testing data to device
    XTrain = torch.from_numpy(XTrain).float().to(device)
    YTrain = torch.from_numpy(YTrain).long().to(device)

    # Run training, suffle data after every epoch
    for i in range(epochs):
      p = np.random.permutation(XTrain.shape[0])
      XTrain = XTrain[p]
      YTrain = YTrain[p]
      # Run batches for epoch
      for j in range(0, XTrain.shape[0], batchSize):
        XMiniBatch = torch.autograd.Variable(XTrain[j:j + batchSize])
        YMiniBatch = torch.autograd.Variable(YTrain[j:j + batchSize])
        optimizer.zero_grad()
        netOut = net(XMiniBatch)
        loss = lossFunction(netOut, YMiniBatch)
        loss.backward()
        optimizer.step()
      if i % 16 == 0:
        print("In featureImportance() still training...")

    # Run testing with no randomization.
    XTestNormal = torch.from_numpy(XTest).float().to(device)
    YTestNormal = torch.from_numpy(YTest).long().to(device)
    XTestBatchNormal = torch.autograd.Variable(XTestNormal)
    YTestBatchNormal = torch.autograd.Variable(YTestNormal)
    netOut = net(XTestBatchNormal)
    _, netPreds = netOut.max(1)
    numCorrectNormal = (netPreds == YTestBatchNormal).sum()
    numSamplesNormal = netPreds.size(0)
    valAccNormal = numCorrectNormal.item() / numSamplesNormal

    # Iterate through all features and get accuracy with values randomized
    valAccFeatures = np.zeros((numFeatures,))
    for i in range(numFeatures):
      N = XTest.shape[0]
      XTestI = XTest
      # Randomize the i-th column of XTestI
      XTestI[:,i] = np.squeeze(np.random.rand(N, 1))
      XTestI = torch.from_numpy(XTestI).float().to(device)
      YTestI = torch.from_numpy(YTest).long().to(device)
      XTestBatchI = torch.autograd.Variable(XTestI)
      YTestBatchI = torch.autograd.Variable(YTestI)
      netOut = net(XTestBatchI)
      _, netPreds = netOut.max(1)
      numCorrectI = (netPreds == YTestBatchI).sum()
      numSamplesI = netPreds.size(0)
      valAccFeatures[i] = numCorrectI.item() / numSamplesI

    # Just leave after 1 fold, for feature importance that is fine
    break

  # Compute the normalized feature importance
  valAccFeaturesStandardized = np.zeros((numFeatures,))
  for i in range(numFeatures):
    valAccFeaturesStandardized[i] = valAccNormal - valAccFeatures[i]
  vafs = valAccFeaturesStandardized
  valAccFeaturesErr = 1 - vafs
  vafs = (vafs.max(axis=0) - vafs) / (vafs.max(axis=0) - vafs.min(axis=0))
  normalizedFeatureImportance = vafs

  # Create a bar graph for normalized feature importance
  plt.bar(np.arange(numFeatures), normalizedFeatureImportance, align='center', alpha=0.5)
  plt.xticks(np.arange(numFeatures), XLabels, rotation='vertical')
  plt.ylabel('Normalized Feature Importance')
  plt.title('Normalized Feature Importance')
  plt.tight_layout()
  plt.savefig('../../data/nnNormalizedFeatureImportance.png')
  plt.close()

  # Create a bar graph for feature importance
  plt.bar(np.arange(numFeatures), valAccFeaturesErr, align='center', alpha=0.5)
  plt.xticks(np.arange(numFeatures), XLabels, rotation='vertical')
  plt.ylabel('Feature Importance')
  plt.title('Feature Importance')
  plt.tight_layout()
  plt.savefig('../../data/nnFeatureImportance.png')
  plt.close()

  # Return
  return



def datasetDistribution(Y):
  dist = np.zeros((numClasses,))
  for i in range(numClasses):
    cnt = Y[Y == i]
    dist[i] = cnt.shape[0]
  distLables = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
  plt.bar(np.arange(numClasses), dist, align='center', alpha=0.5)
  plt.xticks(np.arange(numClasses), distLables, rotation='vertical')
  plt.ylabel('Count')
  plt.title('Stress Level (Scale of 1 to 10)')
  plt.tight_layout()
  plt.savefig('../../data/affectiveROADStressDistribution.png')
  plt.close()
  return



def measureLatency(device, X, Y):
  # Get a random set of data of latency batch size
  p = np.random.permutation(latencyTestBatchSize)
  X = X[p]
  Y = Y[p]

  # Send testing to device
  X = torch.from_numpy(X).float().to(device)
  Y = torch.from_numpy(Y).long().to(device)
  XBatch = torch.autograd.Variable(X)
  YBatch = torch.autograd.Variable(Y)

  # Create new neural network and send to device
  net = Network(X.shape[1], 128, 256, 128, numClasses)
  net = net.to(device)
  optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=momentum)
  lossFunction = torch.nn.CrossEntropyLoss()

  # Time the model average across a bunch of iterations
  iters = 100000
  startTime = time.time_ns()
  for _ in range(iters):
    netOut = net(XBatch)
    _, netPreds = netOut.max(1)
  endTime = time.time_ns()
  runtime = (endTime - startTime) / iters
  print(latencyTestBatchSize, "predictions made in", runtime, "nanoseconds")

  # Return
  return



def main():
  np.random.seed(545)
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  _, XDrivers, XLabels, _, YDrivers = standardizeDataDims()
  X, Y = prepareData(XDrivers, YDrivers)
  datasetDistribution(Y)
  run(device, X, Y)
  featureImportance(device, X, XLabels, Y)
  measureLatency(device, X, Y)
  return



if __name__=="__main__":
    main()


