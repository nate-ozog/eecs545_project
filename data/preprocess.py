import numpy as np
import glob
import os



# USAGE:
# Make sure to unzip data.zip and have a file called data.npy
#
# Load in the data:
# data = np.load("data.npy", allow_pickle=True)
#
# --- HOW TO GET DATA ---
# DRIVERS: ["AD1", "BK1", "EK1", "GM1", "GM2", "KSG1", "MT1", "NM1", "NM2", "NM3", "RY1", "RY2", "SJ1"]
#
# Get data list for given driver:
# data[()]["<DRIVER>"][0]
#
# Get data list labels for driver:
# data[()]["<DRIVER>"][1]
#
# Get dictionary keys for the data
# data[()]["<DRIVER>"][0].keys() --- OR --- data[()]["<DRIVER>"][1].keys()



def preprocessDriver(driver):
  # Combine the data into a dictionary.
  dataDict = {}
  dataLabelsDict = {}

  # Compute relative paths for data
  bioPath = "./Database/Bioharness/Bio_" + driver + ".csv"
  subPath = "./Database/Subj_metric/SM_" + driver + ".csv"
  os.chdir("./Database/E4/")
  for file in glob.glob("*-" + driver):
    erDirName = file
  os.chdir("../..")
  erLeftPath = "./Database/E4/" + erDirName + "/Left/"
  erRightPath = "./Database/E4/" + erDirName + "/Right/"

  # (N x D), D = 4, [HR, BR, Posture, Activity]
  bioData = np.genfromtxt(bioPath, dtype=float, delimiter=';')
  bioData = bioData[:,1:]
  bioData = bioData[1:,:]
  dataDict["bioData"] = bioData
  dataLabelsDict["bioData"] = ["HR", "BR", "Posture", "Activity"]

  # (N x D), D = 1, [subj_stress_metric]
  subData = np.genfromtxt(subPath, dtype=float, delimiter=';')
  subData = subData[1:]
  dataDict["subData"] = subData
  dataLabelsDict["subData"] = ["Subjective Stress Metric"]

  # (N x D), D = 3, [xAccelerometer, yAccelerometer, zAccelerometer]
  leftACC = np.genfromtxt(erLeftPath + "ACC.csv", dtype=float, delimiter=',')
  leftACC = leftACC[2:,:]
  rightACC = np.genfromtxt(erRightPath + "ACC.csv", dtype=float, delimiter=',')
  rightACC = rightACC[2:,:]
  dataDict["leftACC"] = leftACC
  dataLabelsDict["leftACC"] = ["xAccelerometer", "yAccelerometer", "zAccelerometer"]
  dataDict["rightACC"] = rightACC
  dataLabelsDict["rightACC"] = ["xAccelerometer", "yAccelerometer", "zAccelerometer"]

  # (N x D), D = 1, [BVP(photoplethysmograph)]
  leftBVP = np.genfromtxt(erLeftPath + "BVP.csv", dtype=float, delimiter=',')
  leftBVP = leftBVP[2:]
  rightBVP = np.genfromtxt(erRightPath + "BVP.csv", dtype=float, delimiter=',')
  rightBVP = rightBVP[2:]
  dataDict["leftBVP"] = leftBVP
  dataLabelsDict["leftBVP"] = ["BVP (photoplethysmograph)"]
  dataDict["rightBVP"] = rightBVP
  dataLabelsDict["rightBVP"] = ["BVP (photoplethysmograph)"]

  # (N x D), D = 1, [EDA(electrodermal_activity)]
  leftEDA = np.genfromtxt(erLeftPath + "EDA.csv", dtype=float, delimiter=',')
  leftEDA = leftEDA[2:]
  rightEDA = np.genfromtxt(erRightPath + "EDA.csv", dtype=float, delimiter=',')
  rightEDA = rightEDA[2:]
  dataDict["leftEDA"] = leftEDA
  dataLabelsDict["leftEDA"] = ["EDA (Electrodermal Activity)"]
  dataDict["rightEDA"] = rightEDA
  dataLabelsDict["rightEDA"] = ["EDA (Electrodermal Activity)"]

  # (N x D), D = 1, [HR]
  leftHR = np.genfromtxt(erLeftPath + "HR.csv", dtype=float, delimiter=',')
  leftHR = leftHR[2:]
  rightHR = np.genfromtxt(erRightPath + "HR.csv", dtype=float, delimiter=',')
  rightHR = rightHR[2:]
  dataDict["leftHR"] = leftHR
  dataLabelsDict["leftHR"] = ["HR"]
  dataDict["rightHR"] = rightHR
  dataLabelsDict["rightHR"] = ["HR"]

  # # (N x D), D = 2, [init_time, end_time] (time between heart beats)
  # # This data is not very useful so we will probably ignore it
  # leftIBI = np.genfromtxt(erLeftPath + "IBI.csv", dtype=float, delimiter=',')
  # leftIBI = leftIBI[2:]
  # rightIBI = np.genfromtxt(erRightPath + "IBI.csv", dtype=float, delimiter=',')
  # rightIBI = rightIBI[2:]

  # (N x D), D = 1, [celcius]
  leftTEMP = np.genfromtxt(erLeftPath + "TEMP.csv", dtype=float, delimiter=',')
  leftTEMP = leftTEMP[2:]
  rightTEMP = np.genfromtxt(erRightPath + "TEMP.csv", dtype=float, delimiter=',')
  rightTEMP = rightTEMP[2:]
  dataDict["leftTEMP"] = leftTEMP
  dataLabelsDict["leftTEMP"] = ["Temp (Celcius)"]
  dataDict["rightTEMP"] = rightTEMP
  dataLabelsDict["rightTEMP"] = ["Temp (Celcius)"]

  # Return the data and labels.
  return dataDict, dataLabelsDict



def preprocess():
  os.chdir(".")
  drivers = ["AD1", "BK1", "EK1", "GM1", "GM2", "KSG1", "MT1", "NM1", "NM2", "NM3", "RY1", "RY2", "SJ1"]
  data = {}
  for d in drivers:
    print("Preprocessing Driver: " + d)
    driverData, driverLabels = preprocessDriver(d)
    data[d] = [driverData, driverLabels]
  np.save('data.npy', data)
  return



def main():
  preprocess()



if __name__ == '__main__':
    main()


