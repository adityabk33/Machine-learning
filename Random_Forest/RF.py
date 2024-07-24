import pandas as pd
import numpy as np
from decision_tree_cancer import DecisionTreeCancer
from collections import Counter 
import matplotlib.pyplot as plt
import math
import random

data_c = pd.read_csv("hw3_cancer.csv", header=None, skiprows=1)
data_c_split = data_c[0].str.split(expand=True)
data_cancer = data_c_split.sample(frac=1).reset_index(drop=True)
data_cancer_numeric = data_cancer.apply(pd.to_numeric, errors='coerce')

def k_split(data, k):
    instances = len(data)
    size = instances // k
    rem = instances % k
    k_partitions = []
    ind_0 = np.argwhere(data.iloc[:,-1] == 0).flatten()
    ind_1 = np.argwhere(data.iloc[:,-1] == 1).flatten()
    size_0 = math.floor(size * len(ind_0) / (len(ind_0) + len(ind_1)))
    size_1 = math.floor(size * len(ind_1) / (len(ind_0) + len(ind_1)))
    start_0 = 0
    start_1 = 0
    for i in range(k):
        if i < rem:
            end_0 = start_0 + size_0 + 1
            end_1 = start_1 + size_1 + 1
        elif i == k-1:
            end_0 = start_0 + size_0 + 2
            end_1 = start_1 + size_1 + 2
        else:
            end_0 = start_0 + size_0
            end_1 = start_1 + size_1
        part_ind_0 = ind_0[start_0:end_0]
        part_ind_1 = ind_1[start_1:end_1]
        part_ind = np.concatenate((part_ind_0, part_ind_1))
        print(len(part_ind))
        k_partitions.append(data.iloc[part_ind])
        start_0 = end_0
        start_1 = end_1
    return k_partitions   



def train_test(parts, i_test):
    train  =pd.concat([parts[i] for i in range(len(parts)) if i != i_test])
    size = len(train)
    train_boot = train.sample(n = size, replace = True)
    test = parts[i_test]
    return train_boot, test

def k_fold(partitions, i):
    train, test = train_test(partitions, i)
    X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1] 
    X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]
    return X_train, X_test, Y_train, Y_test

def run_RandomForest(data, n):
    partitions = k_split(data, 5)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    for j in range(5):
        X_train, X_test, Y_train, Y_test = k_fold(partitions,j)
        prediction = RandomForest(n, X_train, X_test, Y_train, Y_test)
        Y_test.reset_index(drop=True, inplace=True)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(prediction)):
            if prediction[i] == 1 and Y_test.iloc[i] == 1:
                tp += 1
            if prediction[i] == 0 and Y_test.iloc[i] == 0:
                tn += 1
            if prediction[i] == 0 and Y_test.iloc[i] == 1:
                fn += 1 
            if prediction[i] == 1 and Y_test.iloc[i] == 0:
                fp += 1
        acc = (tp + tn) / len(prediction)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        fs = 2*(pre * rec/(pre + rec))
        accuracy.append(acc)
        precision.append(pre)
        recall.append(rec)
        fscore.append(fs)
    avg_accuracy = sum(accuracy) / len(accuracy)
    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)
    avg_fscore = sum(fscore) / len(fscore)
    return avg_accuracy, avg_precision, avg_recall, avg_fscore
    
    
def RandomForest(n_trees, X_train, X_test, Y_train, Y_test):
    prediciton_trees = []
    for i in range(n_trees):
        #I can do my bootstraping here too, preferably.
        clf = DecisionTreeCancer(X_train, Y_train)
        predictions = clf.make_prediction(X_test)
        prediciton_trees.append(predictions)
    
    final_prediction = []
    for i in range(len(prediciton_trees[0])):
        y_values = [sublist[i] for sublist in prediciton_trees]
        count = Counter(y_values)
        majority = count.most_common(1)[0][0]
        final_prediction.append(majority)
    #print(final_prediction)
    return final_prediction

def make_plot(data):
    n_trees = [1, 5, 10, 20, 30, 40, 50]
    avg_accuracy = []
    avg_precision = []
    avg_recall = []
    avg_fscore = []
    for i in n_trees:
        avg_acc, avg_pre, avg_rec, avg_fs = run_RandomForest(data, i)
        avg_accuracy.append(avg_acc)
        avg_precision.append(avg_pre)
        avg_recall.append(avg_rec)
        avg_fscore.append(avg_fs)
    

    plt.scatter(n_trees, avg_accuracy, color='b', label='Accuracy')
    plt.xlabel('number of trees')
    plt.ylabel('Accuracy')
    plt.show()
    plt.scatter(n_trees, avg_precision, color='g', label='Precision')
    plt.xlabel('number of trees')
    plt.ylabel('Precision')
    plt.show()
    plt.scatter(n_trees, avg_recall, color='r', label='Recall')
    plt.xlabel('number of trees')
    plt.ylabel('Recall')
    plt.show()
    plt.scatter(n_trees, avg_fscore, color='y', label='F1 Score')
    plt.xlabel('number of trees')
    plt.ylabel('FScore')
    plt.show()
    
make_plot(data_cancer_numeric)
#n_trees = [i for i in range(1, 50, 5)]
#avg_accuracy = []
#avg_precision = []
#avg_recall = []
#avg_fscore = []
#for i in n_trees:
#   avg_acc, avg_pre, avg_rec, avg_fs = run_RandomForest(data_house_votes, i)
#    avg_accuracy.append(avg_acc)
#    avg_precision.append(avg_pre)
#    avg_recall.append(avg_rec)
#    avg_fscore.append(avg_fs)

#print(avg_accuracy)
  
