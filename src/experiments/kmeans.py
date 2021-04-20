import pandas as pd
import numpy as np
import seaborn as sns
import time
import tqdm
import copy
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(5,5)})
sns.color_palette("mako", as_cmap=True)
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from data_primer_modified import standardizeDataDims as data
from data_primer import standardizeDataDims as data
import resource

SEED = 1
NUM_DRIVERS = 13
TEST_LATENCY_AND_RAM = True
N_CLUSTERS = 7
OUTFILE_NAME = 'kmeans_results.csv'

def main():
    lat_arr = []
    ram_arr = []
    prec_arr = []
    rec_arr = []
    f1_arr = []
    acc_arr = []
    classes_arr = []
    fe_arr = []

    for num_classes in [2, 3]:
        for fe_extract in [True, False]:
            lat, ram_usage, precision, recall, f1, accuracy = run_experiment(num_classes, fe_extract)
            lat_arr.append(lat)
            ram_arr.append(ram_usage)
            prec_arr.append(precision)
            rec_arr.append(recall)
            f1_arr.append(f1)
            acc_arr.append(accuracy)
            classes_arr.append(num_classes)
            fe_arr.append(fe_extract)

    pd.DataFrame({
        'number of classes' : classes_arr,
        'feature extraction used' : fe_arr,
        'accuracy' : acc_arr,
        'precision' : prec_arr,
        'recall' : rec_arr,
        'f1': f1_arr,
        '50 prediction latency (ns)' : lat_arr,
        'RAM usage' : ram_arr
    }).to_csv(OUTFILE_NAME, index=False)



def run_experiment(num_classes, fe_extract):
    print('Pulling data...')
    _, XDrivers, _, _, YDrivers = data()
    XDrivers, _, YDrivers, _ = prepareData(XDrivers, YDrivers, num_classes)
    X_Drivers = copy.deepcopy(XDrivers)
    Y_Drivers = copy.deepcopy(YDrivers)

    if fe_extract:
        print('Feature extraction...')
        for i in range(NUM_DRIVERS):
            X_Drivers[i] = feature_extraction(XDrivers[i])
            Y_Drivers[i] = new_label(YDrivers[i], num_classes)
    
    print('Making elbow graph...')
    make_elbow_curve_graph(X_Drivers, fe=fe_extract)

    print('Running LODO:')
    return run_lodo(X_Drivers, Y_Drivers)


def run_lodo(X_Drivers, Y_Drivers):
    precision = []
    recall = []
    f1 = []
    accuracy = []

    lat, res_usage = None, None

    for iteration in tqdm.tqdm(range(NUM_DRIVERS)):
        # Get LODO data
        X_train, y_train, X_test, y_test = getLODOIterData(X_Drivers, Y_Drivers, iteration)

        # kmeans = MiniBatchKMeans(n_clusters=NUM_CLASSES, init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
        kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
        kmeans.fit(X_train)
        # Now that the model is fit, assign each centroid's label to be the dominant class (we're really just hoping this lines up with y)
        centroid_to_class_counters = defaultdict(lambda : Counter())
        for i in range(len(X_train)):
            centroid_label = kmeans.labels_[i]
            true_label = y_train[i]
            centroid_to_class_counters[centroid_label][true_label] += 1
        centroid_to_label = {}
        for centroid, counter in centroid_to_class_counters.items():
            centroid_to_label[centroid] = counter.most_common(1)[0][0]

        centroid_preds = kmeans.predict(X_test)
        y_pred = np.array([centroid_to_label[centroid] for centroid in centroid_preds])
        y_true = y_test

        precision.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        recall.append(recall_score(y_true, y_pred, average='macro'))
        f1.append(f1_score(y_true, y_pred, average='macro'))
        accuracy.append(accuracy_score(y_true, y_pred))

        # Do timing and mem usage on last iteration
        if iteration == NUM_DRIVERS - 1 and TEST_LATENCY_AND_RAM:
            print("Running latency and RAM usage benchmark...")
            predict_func = lambda val : [centroid_to_label[pred] for pred in kmeans.predict(val)]
            lat, res_usage = latency(predict_func, X_test, iters=100000)

    return lat, res_usage, np.mean(precision), np.mean(recall), np.mean(f1), np.mean(accuracy)

def latency(predict_func, X_in, iters=100000):
    memory_usage = []
    get_mem = lambda : resource.getrusage(resource.RUSAGE_SELF)
    startTime = time.time_ns()
    for _ in tqdm.tqdm(range(iters)):
        some_i = iters % len(X_in)
        end_i = some_i + 50
        sample = X_in[some_i:end_i] if end_i < len(X_in) else X_in[:50]
        memory_usage.append(get_mem())
        _ = predict_func(sample)
        memory_usage.append(get_mem())
    endTime = time.time_ns()
    runtime = (endTime - startTime) / iters
    return runtime, np.mean(memory_usage)

def make_elbow_curve_graph(X_Drivers, fe=True):
    scores = []
    clusters = []
    all_X = np.concatenate(X_Drivers, axis=0)
    for i in tqdm.tqdm(range(1, 30)):
        kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
        kmeans.fit(all_X)
        score = kmeans.score(all_X)
        clusters.append(i)
        scores.append(-score)
    df = pd.DataFrame({
        '# Clusters' : clusters,
        'Loss' : scores
    })
    sns.lineplot(data=df, x='# Clusters', y='Loss', color='blue')
    plt.savefig('fe_elbow_curve.png' if fe else 'elbow_curve.png')


def prepareData(XDrivers, YDrivers, numClasses):
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

def feature_extraction(X_driver):
    
    lAcc = 0
    rAcc = 0
    for i in range(18):
        start = 0
        end = 50
        mean = []
        std = []
        max_ = []
        min_ = []
        #fdm = []
        
        if (i in [4,5,6]):  
            lAcc += X_driver ** 2
            if (i in [4,5]):
                continue
            
        if (i in [7,8,9]):
            rAcc += X_driver ** 2
            if (i in [7,8]):
                continue
                
        while start + 50 < len(X_driver):
            if i == 6:
                mean.append((lAcc[start:end, i]).mean())
                std.append((lAcc[start:end, i]).std())
                max_.append((lAcc[start:end, i]).max())
                min_.append((lAcc[start:end, i]).min())
            if i == 9:
                mean.append((lAcc[start:end, i]).mean())
                std.append((lAcc[start:end, i]).std())
                max_.append((lAcc[start:end, i]).max())
                min_.append((lAcc[start:end, i]).min())
            # mean, std, max, min
            if (i not in [6,9]): 
                mean.append(X_driver[start:end, i].mean())
                std.append(X_driver[start:end, i].std())
                max_.append(X_driver[start:end, i].max())
                min_.append(X_driver[start:end, i].min())
            
            start += 25
            end = start + 50
        if(i == 0):
            extracted_feats = np.array(mean).reshape(len(mean),1)
        else:
            extracted_feats = np.append(extracted_feats, np.array(mean).reshape(len(mean),1),1)
        extracted_feats = np.append(extracted_feats, np.array(std).reshape(len(std),1),1)
        extracted_feats = np.append(extracted_feats, np.array(max_).reshape(len(max_),1),1)
        extracted_feats = np.append(extracted_feats, np.array(min_).reshape(len(min_),1),1)
        
            
    #print(np.shape(extracted_feats))
    return np.array(extracted_feats[:,:])

def new_label(Y_driver,num_classes):
    new_output = []
    # sliding window approach
    start = 0
    end = 50
    while start + 50 < len(Y_driver):
        new_label = Y_driver[start:end].mean()
        
        ####    Two classes label 
        if num_classes == 2:      
            if new_label >= 0.5:
                new_output.append(1)
            else:
                new_output.append(0)
         
        ####    Three classes label 
        if num_classes == 3:
            if new_label >= 1.33:
                new_output.append(2)
            elif new_label >= 0.67:
                new_output.append(1)
            else:
                new_output.append(0)
 
        start += 25
        end = start + 50

    return np.array(new_output)

if __name__=="__main__":
    main()