import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from torchsummary import summary
from sklearn.model_selection import KFold
from data_primer_modified import standardizeDataDims



# Seed all RNG sources
torch.manual_seed(545)
random.seed(545)
np.random.seed(545)



# Hyperparameters
numFeatures = 18
numClasses = 2
epochs = 100
minibatchSize = 5000
sequenceLen = 50
numLayers = 2
hiddenSize = 16
learningRate = 0.005
weightDecay = 0.01
latencyTestBatchSize = 500
latencyTestIters = 1000



class Network(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = torch.nn.LSTM(numFeatures, hiddenSize, batch_first=True, num_layers=numLayers)
    self.linear = torch.nn.Linear(hiddenSize, numClasses)
    self.h = torch.zeros(numLayers, 1, hiddenSize)
    self.c = torch.zeros(numLayers, 1, hiddenSize)

  def forward(self, x):
    x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
    x = self.linear(x)
    x = x[:, -1, :]
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

  # Get number of drivers for LODO validation iterations.
  numDrivers = len(XDrivers)

  # Counters for validation accuracy.
  numCorrect = 0
  numSamples = 0
  PRMat = np.zeros((numClasses, numClasses))

  # LODO iterations.
  for i in range(numDrivers):
    # Get LODO data
    XTrain, YTrain, XTest, YTest = getLODOIterData(XDrivers, YDrivers, i)

    # Prepare RNN Training data.
    trainBatchSize = XTrain.shape[0] - sequenceLen + 1
    XTrainRNN = np.zeros((trainBatchSize, sequenceLen, numFeatures))
    YTrainRNN = np.zeros((trainBatchSize,))
    for k in range(0, trainBatchSize):
      XTrainRNN[k] = XTrain[k:k + sequenceLen]
      YTrainRNN[k] = YTrain[k:k + sequenceLen][-1]

    # Prepare RNN Testing data.
    testBatchSize = XTest.shape[0] - sequenceLen + 1
    XTestRNN = np.zeros((testBatchSize, sequenceLen, numFeatures))
    YTestRNN = np.zeros((testBatchSize,))
    for k in range(0, testBatchSize):
      XTestRNN[k] = XTest[k:k + sequenceLen]
      YTestRNN[k] = YTest[k:k + sequenceLen][-1]

    # Send data to device.
    XTrain = torch.from_numpy(XTrainRNN).float().to(device)
    YTrain = torch.from_numpy(YTrainRNN).long().to(device)
    XTest = torch.from_numpy(XTestRNN).float().to(device)
    YTest = torch.from_numpy(YTestRNN).long().to(device)

    # Create RNN LSTM.
    net = Network().to(device)
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay=weightDecay)

    for j in range(epochs):
      for k in range(0, trainBatchSize, minibatchSize):
        # Get torch tensors of data.
        XTrainBatch = torch.autograd.Variable(XTrain[k:k + minibatchSize])
        YTrainBatch = torch.autograd.Variable(YTrain[k:k + minibatchSize])

        # Set hidden states to zeros.
        net.h = torch.zeros(numLayers, XTrainBatch.shape[0], hiddenSize).to(device)
        net.c = torch.zeros(numLayers, XTrainBatch.shape[0], hiddenSize).to(device)

        # Run training step for minibatch.
        optimizer.zero_grad()
        netOut = net(XTrainBatch)
        loss = lossFunction(netOut, YTrainBatch)
        loss.backward()
        optimizer.step()

      # Print update messages.
      if j % 10 == 0:
        print("Training Epoch (", j, "/", epochs, "):", loss.item())


    for j in range(0, testBatchSize, minibatchSize):
      # Get torch tensors of data.
      XTestBatch = torch.autograd.Variable(XTest[j:j + minibatchSize])
      YTestBatch = torch.autograd.Variable(YTest[j:j + minibatchSize])

      # Set hidden states to zeros.
      net.h = torch.zeros(numLayers, XTestBatch.shape[0], hiddenSize).to(device)
      net.c = torch.zeros(numLayers, XTestBatch.shape[0], hiddenSize).to(device)

      # Run testing minibatch and collect results.
      netOut = net(XTestBatch)
      _, netPreds = netOut.max(1)
      numCorrect += ((netPreds == YTestBatch).sum()).item()
      numSamples += netPreds.size(0)

      # Fill our precision/recall matrix
      for k in range(netPreds.shape[0]):
        predicted = netPreds[k].item()
        actual = YTestBatch[k].item()
        PRMat[predicted][actual] += 1
      PRMat = PRMat.astype(int)

    # Print current results.
    print("LODO Iter", i, "accuracy:", numCorrect / numSamples)

  # After all LODO iterations do data analysis
  valAcc = numCorrect / numSamples
  print("Average Accuracy =", valAcc)

  # Compute average precision and recall
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
  net = Network().to(device)
  lossFunction = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay=weightDecay)
  testBatchSize = XLat.shape[0] - sequenceLen + 1
  XLatRNN = np.zeros((testBatchSize, sequenceLen, numFeatures))
  for k in range(0, testBatchSize):
    XLatRNN[k] = XLat[k:k + sequenceLen]
  XLatPre = XLatRNN[0:latencyTestBatchSize]

  # Get average prediction time across a bunch of runs.
  # This has to include feature extraction time for the
  # batch as well, because this would have to be done
  # in real-time as well.
  startTime = time.time()
  for i in range(latencyTestIters):
    XLat = torch.from_numpy(XLatPre).float().to(device)
    XTestBatch = torch.autograd.Variable(XLat)
    net.h = torch.zeros(numLayers, XTestBatch.shape[0], hiddenSize).to(device)
    net.c = torch.zeros(numLayers, XTestBatch.shape[0], hiddenSize).to(device)
    netOut = net(XTestBatch)
    _, netPreds = netOut.max(1)
  endTime = time.time()
  runtime = (endTime - startTime) / latencyTestIters
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


