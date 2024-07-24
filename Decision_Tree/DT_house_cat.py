from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statistics
import matplotlib.pyplot as plt
import csv

# I will define my children nodes here. 
class Child_Node:
    
    def __init__(self, feature = None, yes = None, no = None, oth = None, value = None):
        self.feature = feature
        self.yes = yes
        self.no = no
        self.oth = oth
        self.value = value

#I will define my decision tree here. 

class DecisionTree:
    
    def __init__(self, X, Y):
        
        self.root= None
        self.n_features = X.shape[1]
        self.root = self.tree(X, Y)
        
        
    def information_gain(self, X, Y):
        value_p, counts_p = np.unique(Y, return_counts=True)
        t_elements_p = len(Y)
        pb_p = counts_p/t_elements_p
        ent_p = -np.sum([p * np.log(p) for p in pb_p])
    
    
        ind_o = np.argwhere(X == 0).flatten()
        ind_n = np.argwhere(X == 1).flatten()
        ind_y = np.argwhere(X == 2).flatten()
        
        if len(ind_y) > 0:
            value_y, counts_y = np.unique(Y.iloc[ind_y], return_counts=True)
            t_elements_y = len(Y.iloc[ind_y])
            pb_y = counts_y/t_elements_y
            ent_y = -np.sum([p * np.log(p) for p in pb_y])
        else:
            ent_y = 0
        
        if len(ind_n) > 0:
            value_n, counts_n = np.unique(Y.iloc[ind_n], return_counts=True)
            t_elements_n = len(Y.iloc[ind_n])
            pb_n = counts_n/t_elements_n
            ent_n = -np.sum([p * np.log(p) for p in pb_n])
        else:
            ent_n = 0
    
        if len(ind_o) > 0:
            value_o, counts_o = np.unique(Y.iloc[ind_o], return_counts=True)
            t_elements_o = len(Y.iloc[ind_o])
            pb_o = counts_o/t_elements_o
            ent_o = -np.sum([p * np.log(p) for p in pb_o])
        else:
            ent_o = 0
    
        t_n = len(Y)
        n_y, n_n, n_o = len(ind_y), len(ind_n), len(ind_o) 
    
        ent_c = (n_y/t_n) * ent_y + (n_n/t_n) * ent_n + (n_o/t_n) * ent_o
    
        info_gain = ent_p - ent_c
    
        return info_gain
    
    def split(self, X, Y, feature_indices):
        
        self.X = X
        self.Y = Y
        self.feature_indices = feature_indices
        min_gain = -1
        index_split = None
    
    
        for index in feature_indices:
            
            X_col = X.iloc[:,index]
        
            gain = self.information_gain(X_col, Y)
        
            if gain > min_gain:
                min_gain = gain
                index_split = index
            
        return index_split
    
    def tree(self, X, Y, i = 0):
        
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(Y))
    
        if (i >= 50 or n_labels < 2 or n_samples <= 10):
            
            leaf = Counter(Y).most_common(1)[0][0]
            return Child_Node(value = leaf)
        
    
        feature_indices = list(range(16))
        
        
    
        feat_best = self.split(X, Y, feature_indices)
    
        ind_o = np.argwhere(X.iloc[:,feat_best] == 0).flatten()
        ind_n = np.argwhere(X.iloc[:,feat_best] == 1).flatten()
        ind_y = np.argwhere(X.iloc[:,feat_best] == 2).flatten()
        
        if len(ind_y) and len(ind_n) and len(ind_o):
            
            X_y , Y_y = X.iloc[ind_y,:],  Y.iloc[ind_y]
            yes = self.tree(X_y , Y_y, i+1)
            X_n , Y_n = X.iloc[ind_n,:],  Y.iloc[ind_n]
            no = self.tree(X_n , Y_n, i+1)
            X_o , Y_o = X.iloc[ind_o,:],  Y.iloc[ind_o]
            oth = self.tree(X_o , Y_o, i+1)

            return Child_Node(feat_best, yes = yes, no = no, oth = oth)
        
        
        if len(ind_y) == 0 and len(ind_n) and len(ind_o):
            
            X_n , Y_n = X.iloc[ind_n,:],  Y.iloc[ind_n]
            no = self.tree(X_n, Y_n, i+1)
            X_o , Y_o = X.iloc[ind_o,:],  Y.iloc[ind_o]
            oth = self.tree(X_o , Y_o, i+1)
            
            return Child_Node(feat_best,no = no, oth = oth)
        
        if len(ind_n) == 0 and len(ind_y) and len(ind_o):
            
            X_y , Y_y = X.iloc[ind_y,:],  Y.iloc[ind_y]
            yes = self.tree(X_y , Y_y,i+1)
            X_o , Y_o = X.iloc[ind_o,:],  Y.iloc[ind_o]
            oth = self.tree(X_o , Y_o, i+1)
            
            return Child_Node(feat_best,yes = yes,oth = oth)
        
        if len(ind_o) == 0 and len(ind_n) and len(ind_y):
            
            X_y , Y_y = X.iloc[ind_y,:],  Y.iloc[ind_y]
            yes = self.tree(X_y , Y_y, i+1)
            X_n , Y_n = X.iloc[ind_n,:],  Y.iloc[ind_n]
            no = self.tree(X_n , Y_n, i+1)
            
            return Child_Node(feat_best,yes = yes,no = no)
        
        if len(ind_y) == 0 and len(ind_n) == 0 and len(ind_o) != 0:
            
            X_o , Y_o = X.iloc[ind_o,:],  Y.iloc[ind_o]
            oth = self.tree(X_o , Y_o, i+1)
            
            return Child_Node(feat_best,oth = oth)
        
        if len(ind_y) == 0 and len(ind_o) == 0 and len(ind_n) != 0:
            
            X_n , Y_n = X.iloc[ind_n,:],  Y.iloc[ind_n]
            no = self.tree(X_n , Y_n, i+1)
            
            return Child_Node(feat_best,no = no)
        
        if len(ind_n) == 0 and len(ind_o) == 0 and len(ind_y) != 0:
            
            X_y , Y_y = X.iloc[ind_y,:],  Y.iloc[ind_y]
            yes = self.tree(X_y , Y_y, i+1)
        
            return Child_Node(feat_best,yes = yes)
        
        
    
    def make_prediction(self, X):
        prediction = []
        for i in range(X.shape[0]):
            x = X.iloc[i,:]
            value = self.traverse(x, self.root)
            prediction.append(value)
        return prediction
    
    def traverse(self,x, child):
    
        if child.value is not None:
            return child.value

        if x[child.feature] == 2 and child.yes is not None:
            return self.traverse(x, child.yes) 
        
        elif x[child.feature] == 1 and child.no is not None: 
            return self.traverse(x, child.no) 
        
        elif x[child.feature] == 0 and child.oth is not None:
            return self.traverse(x, child.oth)

data = pd.read_csv("house_votes_84.csv")

X, Y = data.iloc[:,:-1], data.iloc[:,-1]

#For training data: 

accuracies_train = []
for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)
    clf = DecisionTree(X_train, Y_train)
    predictions = clf.make_prediction(X_train)
    acc = np.sum(predictions == Y_train) / len(Y_train)
    accuracies_train.append(acc)
    
    
plt.hist(accuracies_train, bins=8, edgecolor='black')
plt.xlabel('Accuracies')
plt.ylabel('Frequency')
plt.title('Histogram for the training data')
plt.show()
print(np.mean(accuracies_train))
print(np.std(accuracies_train))

#For testing data:

accuracies_test = []
for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)
    clf = DecisionTree(X_train, Y_train)
    predictions = clf.make_prediction(X_test)
    acc = np.sum(predictions == Y_test) / len(Y_test)
    accuracies_test.append(acc)   
    
plt.hist(accuracies_test, bins=9, edgecolor='black')
plt.xlabel('Accuracies')
plt.ylabel('Frequency')
plt.title('Histogram for the testing data')
plt.show()
print(np.mean(accuracies_test))
print(np.std(accuracies_test))
