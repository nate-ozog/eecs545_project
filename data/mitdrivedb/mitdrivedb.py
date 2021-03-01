import numpy as np
import wfdb
import glob
import os



# USAGE:
# data = np.load("../mitdrivedb_data.npy")
# labels = np.load("../mitdrivedb_labels.npy")



# Filters the data and updates the matrix as needed
def filterData(signals, labels, filterLabels):
  # Remove the marker datapoint
  remIdx = labels.index("marker") if "marker" in labels else -1
  if (remIdx != -1):
    del labels[remIdx]
    signals = np.delete(signals, remIdx, 1)
  # Filter out only data that has these labels
  include = False
  if labels == filterLabels:
    include = True
  return include, signals



# Preprocesses all files in the database
def preprocess():
  filterLabels = ['ECG', 'EMG', 'foot GSR', 'hand GSR', 'HR', 'RESP']
  labels = np.array(filterLabels)
  D = len(filterLabels)
  data = np.zeros((0, D))
  os.chdir(".")
  for file in glob.glob("*.dat"):
    signals, fields = wfdb.rdsamp(file)
    include, signals = filterData(signals, fields["sig_name"], filterLabels)
    if include:
      data = np.concatenate((data, signals))
  np.save("../mitdrivedb_data.npy", data)
  np.save("../mitdrivedb_labels.npy", labels)
  return



# Driver code
def main():
  preprocess()



if __name__ == '__main__':
    main()


