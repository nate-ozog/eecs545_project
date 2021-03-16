import torch
import numpy as np
from sklearn.model_selection import KFold
from data_primer import standardizeDataDims



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



def prepareData(XDrivers, YDrivers, NumClasses):
  """
  Normalizes data on a per-driver basis. Normalizes
  every feature column between 0 and 1. Organizes
  stress data into a one-hot vector of num classes
  i.e. 2 classes means the boundary of stressed vs
  not stressed is 0.5. Ideally, we want a range, for
  example a classification of 1-10 means
  NumClasses = 10.
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

  # Create NumClasses one-hot vectors for classification
  yRange = np.arange(NumClasses + 1)
  yRange = (yRange - yRange.min(axis=0)) / (yRange.max(axis=0) - yRange.min(axis=0))
  for i in range(numDrivers):
    Yi = YDrivers[i]
    YiC = np.zeros(Yi.shape)
    for j in range(1, NumClasses):
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



def run(device, X, Y, NumClasses):
  # Shuffle X and Y data to diversify gradient descent between drivers
  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = Y[p]

  # Hyperparameters
  learningRate = 1e-2
  momentum = 0.9
  epochs = 128
  batchSize = 64

  # K-Fold validation variables
  numCorrect = 0
  numSamples = 0

  # K-Fold iterations
  kf = KFold(n_splits=10)
  for trainIdx, testIdx in kf.split(X):
    # Get data for K-iteration
    XTrain = X[trainIdx]
    YTrain = Y[trainIdx]
    XTest = X[testIdx]
    YTest = Y[testIdx]

    # Create new neural network and send to device
    net = Network(X.shape[1], 128, 256, 128, NumClasses)
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
      for j in range(0, XTrain.shape[0], batchSize):
        XMiniBatch = torch.autograd.Variable(XTrain[j:j + batchSize])
        YMiniBatch = torch.autograd.Variable(YTrain[j:j + batchSize])
        optimizer.zero_grad()
        netOut = net(XMiniBatch)
        loss = lossFunction(netOut, YMiniBatch)
        loss.backward()
        optimizer.step()
      print(loss.item())

    # Run testing
    XTest = torch.from_numpy(XTest).float().to(device)
    YTest = torch.from_numpy(YTest).long().to(device)
    XTestBatch = torch.autograd.Variable(XTest)
    YTestBatch = torch.autograd.Variable(YTest)
    netOut = net(XTestBatch)
    _, netPreds = netOut.max(1)
    numCorrect += (netPreds == YTestBatch).sum()
    numSamples += netPreds.size(0)
    print("Current Acc:", numCorrect.item() / numSamples)

  return



def main():
  np.random.seed(545)
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  _, XDrivers, _, _, YDrivers = standardizeDataDims()
  NumClasses = 3
  X, Y = prepareData(XDrivers, YDrivers, NumClasses)
  run(device, X, Y, NumClasses)
  return



if __name__=="__main__":
    main()


