# Importing libraries

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statistics
import matplotlib.pyplot as plt
import csv
import random
import math



# I will define my children nodes here. 

class Child_Node:
    
    def __init__(self, feature = None, threshold = None, yes = None, no = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.yes = yes
        self.no = no
        self.value = value

#I will define my decision tree here. 

class DecisionTreeCancer:
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.root= None
        self.n_features = X.shape[1]
        self.root = self.tree(X, Y)
        
        
    def information_gain(self, X, Y, threshold):
        value_p, counts_p = np.unique(Y, return_counts=True)
        t_elements_p = len(Y)
        pb_p = counts_p/t_elements_p
        ent_p = -np.sum([p * np.log(p) for p in pb_p if p > 0])
    
    
        ind_y = np.argwhere(X <= threshold).flatten()
        ind_n = np.argwhere(X > threshold).flatten()
        
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
        t_n = len(Y)
        n_y, n_n = len(ind_y), len(ind_n)
    
        ent_c = (n_y/t_n) * ent_y + (n_n/t_n) * ent_n
    
        info_gain = ent_p - ent_c
    
        return info_gain
    
    def split(self, X, Y, feature_indices):
        
        min_gain = -1
        index_split = None
        index_thres = None
    
    
        for index in feature_indices:
            X_col = X.iloc[:,index]
            threshold = np.mean(X_col)
            gain = self.information_gain(X_col, Y, threshold)
        
            if gain > min_gain:
                min_gain = gain
                index_split = index
                index_thres = threshold
                    
        return index_split, index_thres
    
    def tree(self, X, Y, i = 0):
        
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(Y))
    
        if i > 15 or n_samples < 20:
            counter = Counter(Y)
            most_common_elements = counter.most_common(1)

            if len(most_common_elements) > 1:
                leaf = random.choice(most_common_elements)[0]
            else:
                leaf = most_common_elements[0][0]
                
            return Child_Node(value = leaf)

        indices = list(range(self.n_features))
        m = math.floor(np.sqrt(self.n_features))
        feature_indices = random.sample(indices,m)
        feat_best, thres_best = self.split(X, Y, feature_indices)
    
        ind_n = np.argwhere(X.iloc[:,feat_best] > thres_best).flatten()
        ind_y = np.argwhere(X.iloc[:,feat_best] <= thres_best).flatten()
        
        if len(ind_y) and len(ind_n):
            
            X_y , Y_y = X.iloc[ind_y,:],  Y.iloc[ind_y]
            yes_l = self.tree(X_y , Y_y, i+1)
            X_n , Y_n = X.iloc[ind_n,:],  Y.iloc[ind_n]
            no_l = self.tree(X_n , Y_n, i+1)

            return Child_Node(feat_best, thres_best, yes = yes_l, no = no_l)
        
        
        if len(ind_y) == 0 and len(ind_n) != 0:
            
            #X_y, Y_y = None, None
            #yes_l = self.tree(X_y, Y_y,i+1)
            X_n , Y_n = X.iloc[ind_n,:],  Y.iloc[ind_n]
            no_l = self.tree(X_n, Y_n, i+1)
            
            #return Child_Node(feat_best, thres_best, no = no)
            return Child_Node(feat_best, thres_best, yes = None, no = no_l)
        else:
            
            X_y , Y_y = X.iloc[ind_y,:],  Y.iloc[ind_y]
            yes_l = self.tree(X_y , Y_y,i+1)
            #X_n, Y_n = None, None
            #no_l = self.tree(X_n, Y_n,i+1)
            
            #return Child_Node(feat_best,thres_best, yes = yes)
            return Child_Node(feat_best,thres_best, yes = yes_l, no = None)
        
        
    
    def make_prediction(self, X):
        prediction = []
        for x in X:
            value = self.traverse(x, self.root)
            prediction.append(value)
        return prediction
    
    def traverse(self,x, child):
    
        if child.value is not None:
            return child.value

        if x.iloc[child.feature] <= child.threshold:
            return self.traverse(x, child.yes)
        return self.traverse(x, child.no) 
        








