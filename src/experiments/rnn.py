import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torchsummary import summary
from sklearn.model_selection import KFold
from data_primer import standardizeDataDims



# Hyperparameters
numFeatures = 18
numClasses = 3
epochs = 100
minibatchSize = 1000 # Large values will make GPU run out of memory
sequenceLen = 100    # Sample rate means approx. 50 seq size is 1s of real time data
numLayers = 2        # LSTM hidden layers
hiddenSize = 5       # LSTM hidden layer size
learningRate = 0.001



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



def run(device, XDrivers, YDrivers):

  # Get number of drivers for LODO validation iterations.
  numDrivers = len(XDrivers)

  # Counters for validation accuracy.
  numCorrect = 0
  numSamples = 0

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
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, )

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
        print("Training Epoch (", j, "/", epochs, ")")


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

    # Print current results.
    print("LODO Iter", i, "accuracy:", numCorrect / numSamples)

  # Return
  return



def main():
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  _, XDrivers, XLabels, _, YDrivers = standardizeDataDims()
  XDrivers, X, YDrivers, Y = prepareData(XDrivers, YDrivers)
  run(device, XDrivers, YDrivers)
  return



if __name__=="__main__":
    main()


