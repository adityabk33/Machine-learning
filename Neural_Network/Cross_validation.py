#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math


# In[3]:


class CV_wine:
    def __init__(self, data):
        
        self.data = data
        
        
    def k_fold(self, partitions, i):
        
        self.partitions = partitions
        self.i = i
        train, test = self.train_test(self.partitions, i)
        X_train, Y_train = train[:,1:], train[:,0] 
        X_test, Y_test = test[:,1:], test[:,0]
    
        return X_train, X_test, Y_train, Y_test

    def train_test(self, parts, i_test):
        
        self.parts = parts
        self.i_test = i_test
        train  =np.concatenate([self.parts[i] for i in range(len(self.parts)) if i != self.i_test])
        test = self.parts[self.i_test]
        #size = len(train)
        #train_boot = train.sample(n = size, replace = True)
        #test = parts[i_test]
        
        return train, test
    
    def k_split(self, k):
        
        self.k = k
        
        instances = len(self.data)
        size = instances // self.k
        rem = instances % self.k
        k_partitions = []
        ind_1 = np.argwhere(self.data[:,0] == 1).flatten()
        ind_2 = np.argwhere(self.data[:,0] == 2).flatten()
        ind_3 = np.argwhere(self.data[:,0] == 3).flatten()

        size_1 = math.floor(size * len(ind_1) / (len(ind_1) + len(ind_2) + len(ind_3)))
        size_2 = math.floor(size * len(ind_2) / (len(ind_1) + len(ind_2) + len(ind_3)))
        size_3 = math.floor(size * len(ind_3) / (len(ind_1) + len(ind_2) + len(ind_3)))
    
        ind = np.arange(instances)
        np.random.shuffle(ind)
    
        start_1 = 0
        start_2 = 0
        start_3 = 0
        for i in range(self.k):
            if i < rem:
                end_1 = start_1 + size_1 + 1
                end_2 = start_2 + size_2 + 1
                end_3 = start_3 + size_3 + 1
            elif i == k-1:
                end_1 = start_1 + size_1 + 3
                end_2 = start_2 + size_2 + 3
                end_3 = start_3 + size_3 + 3
            else:
                end_1 = start_1 + size_1
                end_2 = start_2 + size_2
                end_3 = start_3 + size_3
        
            part_ind_1 = ind_1[start_1:end_1]
            part_ind_2 = ind_2[start_2:end_2]
            part_ind_3 = ind_3[start_3:end_3]
            part_ind = np.concatenate((part_ind_1, part_ind_2, part_ind_3))
            k_partitions.append(self.data[part_ind])
    
            start_1 = end_1
            start_2 = end_2
            start_3 = end_3
        return k_partitions


# In[4]:


class CV_house:
    def __init__(self, data):
        
        self.data = data
        
        
    def k_fold(self, partitions, i):
        
        self.partitions = partitions
        self.i = i
        train, test = self.train_test(self.partitions, i)
        X_train, Y_train = train[:,1:], train[:,0] 
        X_test, Y_test = test[:,1:], test[:,0]
    
        return X_train, X_test, Y_train, Y_test

    def train_test(self, parts, i_test):
        
        self.parts = parts
        self.i_test = i_test
        train  =np.concatenate([self.parts[i] for i in range(len(self.parts)) if i != self.i_test])
        test = self.parts[self.i_test]
        #size = len(train)
        #train_boot = train.sample(n = size, replace = True)
        #test = parts[i_test]
        
        return train, test
    
    def k_split(self, k):
        
        self.k = k
        
        instances = len(self.data)
        size = instances // self.k
        rem = instances % self.k
        k_partitions = []
        ind_0 = np.argwhere(self.data[:,0] == 0).flatten()
        ind_1 = np.argwhere(self.data[:,0] == 1).flatten()
        

        size_0 = math.floor(size * len(ind_0) / (len(ind_0) + len(ind_1)))
        size_1 = math.floor(size * len(ind_1) / (len(ind_0) + len(ind_1)))
        
    
        ind = np.arange(instances)
        np.random.shuffle(ind)
    
        start_0 = 0
        start_1 = 0
                            
        for i in range(self.k):
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
            k_partitions.append(self.data[part_ind])
                            
            start_0 = end_0
            start_1 = end_1
    
        return k_partitions
    


# In[7]:






# In[ ]:




