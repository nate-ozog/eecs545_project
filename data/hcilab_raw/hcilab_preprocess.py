import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# constants
RAW_FILE_DIR = "/"
PREPROCESSED_FILE_DIR = "../hcilab_preprocessed/"

def process_dataset(dataset):
    particip_num = int(file.split('.')[0].split('_')[1]) # get the participant number of the csv file

    # TODO

    return

def main():
    # check if output dir exists
    if not os.path.isdir(PREPROCESSED_FILE_DIR):
        os.makedirs(PREPROCESSED_FILE_DIR)
    # preprocess each raw dataset
    for file in os.listdir(RAW_FILE_DIR):
        if file.endswith('.csv'):
            process_dataset(file)
    return

if __name__=="__main__":
    main()