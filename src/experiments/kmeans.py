import pandas as pd
import numpy as np
import time
import tqdm
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os, os.path
import errno
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from data_primer_modified import standardizeDataDims as data
from data_primer import standardizeDataDims as data

DIR = Path(os.path.abspath('')).resolve()
ROOT = DIR.parent.parent
EVALUATION = ROOT/"Evaluation"
OUTDIR = EVALUATION/"kmeans"

SEED = 1
NUM_DRIVERS = 13
TEST_LATENCY_AND_RAM = False
PCA_COMPONENTS = 2

# 1st dict is FE choice
# 2nd dict is PCA choice
N_CLUSTERS_DICT = {
    True : {
        True : 7,   # FE + PCA
        False : 8,  # FE
    },
    False : {
        True : 5,   # PCA
        False : 12  #
    }
}

def main():
    try_mkdir(str(OUTDIR))
    run_graph_experiment()
    run_full_experiment()
    make_pairwise_graphs()
    make_pairwise_grid()
    make_elbow_grid()

def run_graph_experiment():
    for fe_extract in [True, False]:
        for use_pca in [True, False]:
            graph_experiment(fe_extract, use_pca)

def make_pairwise_grid():
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    fig.suptitle('Cluster & Label Comparison')
    plot_index = 0
    for num_classes in [2, 3]:
        for fe_extract in [True, False]:

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

            X = np.concatenate(X_Drivers)
            y = np.concatenate(Y_Drivers)

            print('PCA...')
            pca = PCA(n_components=2)
            embedded_X = pca.fit_transform(X)

            kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS_DICT[fe_extract][True], init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
            kmeans.fit(embedded_X)

            df = pd.DataFrame({
                'pca0' : embedded_X[:, 0],
                'pca1' : embedded_X[:, 1],
                'Cluster' : kmeans.labels_,
                'Label' : [int(item) for item in y]
            })

            ax = axes[plot_index]
            ax[0].set_ylabel(f"{num_classes}-Class {'w/FE' if fe_extract else ''}", rotation='vertical', size='large')
            ax[1].set_ylabel(' ')
            ax[0].set_xlabel(' ')
            ax[1].set_xlabel(' ')

            sns.scatterplot(ax=ax[0], data=df, x='pca0', y='pca1', hue='Cluster', palette='mako', linewidth=0.05, edgecolors=None)
            sns.scatterplot(ax=ax[1], data=df, x='pca0', y='pca1', hue='Label', palette='mako', linewidth=0.05, edgecolors=None)
            ax[0].get_legend().remove()
            ax[1].get_legend().remove()
            plot_index += 1

    axes[0][0].set_title('Cluster')
    axes[0][1].set_title('Label')
    plt.savefig(f"{str(OUTDIR)}/all_scatter.png", bbox_inches='tight')
    plt.close()

def make_pairwise_graphs():
    for num_classes in [2, 3]:
        for fe_extract in [True, False]:

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

            X = np.concatenate(X_Drivers)
            y = np.concatenate(Y_Drivers)

            print('PCA...')
            pca = PCA(n_components=2)
            embedded_X = pca.fit_transform(X)

            kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS_DICT[fe_extract][True], init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
            kmeans.fit(embedded_X)

            df = pd.DataFrame({
                'pca0' : embedded_X[:, 0],
                'pca1' : embedded_X[:, 1],
                'Cluster' : kmeans.labels_,
                'Label' : [int(item) for item in y]
            })
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
            fig.suptitle('Cluster & Label Comparison')

            sns.scatterplot(ax=axes[0], data=df, x='pca0', y='pca1', hue='Cluster', palette="mako")
            axes[0].set_title('Cluster')

            sns.scatterplot(ax=axes[1], data=df, x='pca0', y='pca1', hue='Label', palette="mako")
            axes[1].set_title('Label')
            plt.savefig(f"{str(OUTDIR)}/{str(num_classes)}_{'fe_' if fe_extract else ''}scatter.png")
            plt.close()

def run_full_experiment():
    lat_arr = []
    prec_arr = []
    rec_arr = []
    f1_arr = []
    acc_arr = []

    classes_arr = []
    fe_arr = []
    clusters_arr = []
    pca_arr = []

    for num_classes in [2, 3]:
        for fe_extract in [True, False]:
            for use_pca in [True, False]:
                elbow_analysis_choice = N_CLUSTERS_DICT[fe_extract][use_pca]
                for num_clusters in [elbow_analysis_choice, num_classes]:
                    lat, precision, recall, f1, accuracy = run_experiment(num_classes, fe_extract, use_pca, num_clusters)

                    lat_arr.append(lat)
                    prec_arr.append(precision)
                    rec_arr.append(recall)
                    f1_arr.append(f1)
                    acc_arr.append(accuracy)

                    classes_arr.append(num_classes)
                    clusters_arr.append(num_clusters)
                    fe_arr.append(fe_extract)
                    pca_arr.append(use_pca)

    pd.DataFrame({
        'number of classes' : classes_arr,
        'number of clusters' : clusters_arr,
        'feature extraction used' : fe_arr,
        'pca used' : pca_arr,
        'accuracy' : acc_arr,
        'precision' : prec_arr,
        'recall' : rec_arr,
        'f1': f1_arr,
        '500 prediction latency (ms)' : lat_arr,
    }).to_csv(str(OUTDIR/'kmeans_results.csv'), index=False)

def make_elbow_grid():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.suptitle('PCA')
    fig.text(0.04, 0.5, 'FE', va='center', rotation='vertical', size='large')
    plot_index = 0
    for use_pca in [True, False]:
        for fe_extract in [True, False]:
            print('Pulling data...')
            _, XDrivers, _, _, YDrivers = data()
            # Num classes is arbitrary here, we don't use the y assignments
            XDrivers, _, YDrivers, _ = prepareData(XDrivers, YDrivers, 2)
            X_Drivers = copy.deepcopy(XDrivers)

            if fe_extract:
                print('Feature extraction...')
                for i in range(NUM_DRIVERS):
                    X_Drivers[i] = feature_extraction(XDrivers[i])

            X_Drivers = np.concatenate(X_Drivers)

            if use_pca:
                pca = PCA(n_components=PCA_COMPONENTS)
                X_Drivers = pca.fit_transform(X_Drivers)
            
            print('Making elbow graph...')
            scores = []
            clusters = []
            for i in tqdm.tqdm(range(1, 30)):
                kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
                kmeans.fit(X_Drivers)
                score = kmeans.score(X_Drivers)
                clusters.append(i)
                scores.append(-score)
            df = pd.DataFrame({
                '# Clusters' : clusters,
                'Loss' : scores
            })

            m, n = plot_index % 2, int(plot_index/2)
            ax = axes[m][n]

            if m == 0:
                ax.set_title("Yes" if use_pca else "No", fontsize='large')
            else:
                ax.set_title(' ')
            if n == 0:
                ax.set_ylabel("Yes" if fe_extract else "No", fontsize='large')
            else:
                ax.set_ylabel(' ')


            sns.lineplot(ax=ax, data=df, x='# Clusters', y='Loss', color='blue')
            plot_index += 1

    plt.savefig(f"{str(OUTDIR)}/all_elbows.png", bbox_inches='tight')
    plt.close()

def graph_experiment(fe_extract, use_pca):
    print('Pulling data...')
    _, XDrivers, _, _, YDrivers = data()
    # Num classes is arbitrary here, we don't use the y assignments
    XDrivers, _, YDrivers, _ = prepareData(XDrivers, YDrivers, 2)
    X_Drivers = copy.deepcopy(XDrivers)

    if fe_extract:
        print('Feature extraction...')
        for i in range(NUM_DRIVERS):
            X_Drivers[i] = feature_extraction(XDrivers[i])

    X_Drivers = np.concatenate(X_Drivers)

    if use_pca:
        pca = PCA(n_components=PCA_COMPONENTS)
        X_Drivers = pca.fit_transform(X_Drivers)
    
    print('Making elbow graph...')
    make_elbow_curve_graph(X_Drivers, fe_extract, use_pca)


def run_experiment(num_classes, fe_extract, use_pca, n_clusters):
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

    lat_test_set = XDrivers[0]

    print('Running LODO:')
    return run_lodo(X_Drivers, Y_Drivers, lat_test_set, fe_extract, use_pca, n_clusters)

def run_lodo(X_Drivers, Y_Drivers, lat_test_set, fe_extract, use_pca, n_clusters):
    precision = []
    recall = []
    f1 = []
    accuracy = []

    lat = None

    for iteration in tqdm.tqdm(range(NUM_DRIVERS)):
        # Get LODO data
        X_train, y_train, X_test, y_test = getLODOIterData(X_Drivers, Y_Drivers, iteration)

        if use_pca:
            pca = PCA(n_components=PCA_COMPONENTS)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
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
            lat = latency(predict_func, lat_test_set, fe_extract, iters=100000)

    return lat, np.mean(precision), np.mean(recall), np.mean(f1), np.mean(accuracy)

def latency(predict_func, X_in, fe_extract, iters=100000):
    X = copy.deepcopy(X_in)
    startTime = time.time_ns() * 1e-6

    for _ in tqdm.tqdm(range(iters)):

        some_i = iters % len(X)
        end_i = some_i + 500
        sample = X_in[some_i:end_i] if end_i < len(X) else X[:500]

        if fe_extract:
            sample = feature_extraction(sample)

        _ = predict_func(sample)
    endTime = time.time_ns() * 1e-6
    runtime = (endTime - startTime) / iters
    print('Runtime (ms):', runtime)
    return runtime

def make_elbow_curve_graph(X_Drivers, fe, use_pca):
    scores = []
    clusters = []
    for i in tqdm.tqdm(range(1, 30)):
        kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=1000, batch_size=10000, compute_labels=True, random_state=SEED)
        kmeans.fit(X_Drivers)
        score = kmeans.score(X_Drivers)
        clusters.append(i)
        scores.append(-score)
    df = pd.DataFrame({
        '# Clusters' : clusters,
        'Loss' : scores
    })
    sns.lineplot(data=df, x='# Clusters', y='Loss', color='blue')
    plt.savefig(f"{str(OUTDIR)}/{'fe_' if fe else ''}{'pca_' if use_pca else ''}elbow.png")
    plt.close()

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

def try_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

if __name__=="__main__":
    main()