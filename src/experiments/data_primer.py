"""
Experimenting/playing around with the data and features.
Goal: Familiarize with obtaining the data, cleaning, and preprocessing/feature extracting for training models.
"""
import os
import numpy as np 
import matplotlib.pyplot as pyplot
import sklearn
from sklearn.preprocessing import StandardScaler, normalize

DATA_PATH = "../../data/data.npy"

def load_data(file):
    # Load the .npy dataset
    return np.load(file, allow_pickle=True)

def get_driver_data(data):
    """
    Get all relevant data features/labels for driver.
    Input:
        file: data.npy
        driver_id: Unique driver label (i.e, AD1)
    Output:
        feature_names: ordered features of the data
        feature_values: values of the respective feature_names
    """
    feature_names = data[()]["AD1"][1]
    feature_values = data[()]["AD1"][0]

    print(feature_names) # individual features grouped into categories
    print(feature_values['bioData']) 
    print(feature_values['bioData'][:, 0]) # HR column of bioData
    #print(feature_values['rightBVP'])
    print(feature_values['subData']) # subjective stress metric

    return feature_names['bioData'], feature_values['bioData']

def standardize_data(feature_labels, feature_values):
    """
    Standardize the data, features of different scales.
    Input:
        feature_labels:
        feature_values:
    Output:
        feature_labels: Same as input feature_labels
        feature_values: Standardized features across appropriate axis 
    """
    #scaler = StandardScaler()
    #feature_values = scaler.fit_transform(feature_values)
    feature_values = normalize(feature_values, axis=0) # normalize along columns

    print("Standardized features: ")
    print(feature_values)


def main():
    data = load_data(DATA_PATH)
    print("Dataset loaded!")

    feature_names, feature_values = get_driver_data(data)

    standardize_data(feature_names, feature_values)
    return

if __name__=="__main__":
    main()