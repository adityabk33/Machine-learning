#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets
import random


# In[2]:


class CV_digits:
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
        
        return train, test
    
    def k_split(self, k):
        
        self.k = k
        
        instances = len(self.data)
        size = instances // self.k
        rem = instances % self.k
        k_partitions = []
        ind_0 = np.argwhere(self.data[:,0] == 0).flatten()
        ind_1 = np.argwhere(self.data[:,0] == 1).flatten()
        ind_2 = np.argwhere(self.data[:,0] == 2).flatten()
        ind_3 = np.argwhere(self.data[:,0] == 3).flatten()
        ind_4 = np.argwhere(self.data[:,0] == 4).flatten()
        ind_5 = np.argwhere(self.data[:,0] == 5).flatten()
        ind_6 = np.argwhere(self.data[:,0] == 6).flatten()
        ind_7 = np.argwhere(self.data[:,0] == 7).flatten()
        ind_8 = np.argwhere(self.data[:,0] == 8).flatten()
        ind_9 = np.argwhere(self.data[:,0] == 9).flatten()
        
        L = len(ind_0)+len(ind_1)+len(ind_2)+len(ind_3)+len(ind_4)+len(ind_5)+len(ind_6)+len(ind_7)+ len(ind_8)+len(ind_9)

        size_0 = math.floor(size * len(ind_0) / L)
        size_1 = math.floor(size * len(ind_1) / L)
        size_2 = math.floor(size * len(ind_2) / L)
        size_3 = math.floor(size * len(ind_3) / L)
        size_4 = math.floor(size * len(ind_4) / L)
        size_5 = math.floor(size * len(ind_5) / L)
        size_6 = math.floor(size * len(ind_6) / L)
        size_7 = math.floor(size * len(ind_7) / L)
        size_8 = math.floor(size * len(ind_8) / L)
        size_9 = math.floor(size * len(ind_9) / L)
                        
                            
                            
        ind = np.arange(instances)
        np.random.shuffle(ind)
    
        start_0 = 0
        start_1 = 0
        start_2 = 0
        start_3 = 0
        start_4 = 0
        start_5 = 0
        start_6 = 0
        start_7 = 0
        start_8 = 0
        start_9 = 0                          
                            
                            
        for i in range(self.k):
            if i < rem:
                end_0 = start_0 + size_0 + 1
                end_1 = start_1 + size_1 + 1
                end_2 = start_2 + size_2 + 1
                end_3 = start_3 + size_3 + 1
                end_4 = start_4 + size_4 + 1
                end_5 = start_5 + size_5 + 1
                end_6 = start_6 + size_6 + 1
                end_7 = start_7 + size_7 + 1
                end_8 = start_8 + size_8 + 1
                end_9 = start_9 + size_9 + 1
                
            elif i == k-1:
                end_0 = start_0 + size_0 + 2
                end_1 = start_1 + size_1 + 2
                end_2 = start_2 + size_2 + 2
                end_3 = start_3 + size_3 + 2
                end_4 = start_4 + size_4 + 2
                end_5 = start_5 + size_5 + 2
                end_6 = start_6 + size_6 + 2
                end_7 = start_7 + size_7 + 2
                end_8 = start_8 + size_8 + 2
                end_9 = start_9 + size_9 + 2
            else:
                end_0 = start_0 + size_0 
                end_1 = start_1 + size_1 
                end_2 = start_2 + size_2 
                end_3 = start_3 + size_3 
                end_4 = start_4 + size_4 
                end_5 = start_5 + size_5 
                end_6 = start_6 + size_6 
                end_7 = start_7 + size_7 
                end_8 = start_8 + size_8 
                end_9 = start_9 + size_9 
        
            part_ind_0 = ind_0[start_0:end_0]
            part_ind_1 = ind_1[start_1:end_1]
            part_ind_2 = ind_2[start_2:end_2]
            part_ind_3 = ind_3[start_3:end_3]
            part_ind_4 = ind_4[start_4:end_4]
            part_ind_5 = ind_5[start_5:end_5]
            part_ind_6 = ind_6[start_6:end_6]
            part_ind_7 = ind_7[start_7:end_7]
            part_ind_8 = ind_8[start_8:end_8]
            part_ind_9 = ind_9[start_9:end_9]
            part_ind = np.concatenate((part_ind_0, part_ind_1, part_ind_2, part_ind_3, part_ind_4, part_ind_5, part_ind_6, part_ind_7, part_ind_8, part_ind_9))
            #part_ind1 = random.shuffle(part_ind)
            k_partitions.append(self.data[part_ind])
                            
            start_0 = end_0
            start_1 = end_1
            start_2 = end_2
            start_3 = end_3
            start_4 = end_4
            start_5 = end_5
            start_6 = end_6
            start_7 = end_7
            start_8 = end_8
            start_9 = end_9
        return k_partitions



# In[3]:


class NN3:
    def __init__(self, data):
        self.data = data
        
    def hard_code(self, i):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        y = self.Y_train[i].astype(int)
        y_hard_code = Y_hard_code[y,:]
        
        return y_hard_code
    
    def activation(self, z):
        
        return 1 / (1 + np.exp(-z))
    
    def initial_params_new(self, nn1, nn2, nn3):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        p, q = self.X_train.shape
        w1 = np.random.normal(0, 1, (nn1, q))
        w2 = np.random.normal(0, 1, (nn2, nn1+1))
        w3 = np.random.normal(0, 1, (nn3, nn2+1))
        w4 = np.random.normal(0, 1, (Y_unique_length, nn3+1))
        
        return w1, w2, w3, w4
    
    def forward(self, w1, w2, w3, w4, x):
        a1 = x
        z2 = np.dot(w1, a1.T)
        a2 = self.activation(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = np.dot(w2, a2.T)
        a3 = self.activation(z3)
        a3 = np.insert(a3, 0, 1)
        z4 = np.dot(w3, a3.T)
        a4 = self.activation(z4)
        a4 = np.insert(a4, 0, 1)
        z5 =  np.dot(w4, a4.T)
        a5 = self.activation(z5)
        return a1, a2, a3, a4, a5
    
    def backward(self, a2, a3, a4,a5, w1, w2, w3,w4, x, y):
        a1 = x
        a1 = a1.reshape(1,-1)
        a2 = a2.reshape(1,-1)
        a3 = a3.reshape(1,-1)
        a4 = a4.reshape(1,-1)
        a5 = a5.reshape(1,-1)
        y = y.reshape(1,-1)
        d5 = a5 - y
        d5 = d5.reshape(-1,1)
        
        dp4 = np.dot(w4[:, 1:a4.size].T, d5)
        d4 = np.zeros((dp4.size,1))
        for i in range(dp4.size):
            N = a4[:,i+1]*(1-a4[:,i+1])
            d4[i] = dp4[i]*N
            
        dp3 = np.dot(w3[:, 1:a3.size].T, d4)
        d3 = np.zeros((dp3.size,1))
        for i in range(dp3.size):
            N = a3[:,i+1]*(1-a3[:,i+1])
            d3[i] = dp3[i]*N
        dp2 = np.dot(w2[:,1:a2.size].T, d3)
        d2 =np.zeros((dp2.size,1))
        for i in range(dp2.size):
            N = a2[:,i+1]*(1-a2[:,i+1])
            d2[i] = dp2[i]*N 
            
        grad4 = np.dot(d5, a4)
        grad3 = np.dot(d4, a3)
        grad2 = np.dot(d3, a2)
        grad1 = np.dot(d2, a1)
        
        return  grad4, grad3, grad2, grad1, d5, d4, d3, d2
    
    
    def update_parameters(self, w1, w2, w3, w4, alpha, lamb):
        grad4_array = []
        grad3_array = []
        grad2_array = []
        grad1_array = []
        m, n = self.X_train.shape
        for i in range(m):
            x = self.X_train[i]
            y = self.hard_code(i)
            a1, a2, a3, a4, a5 = self.forward(w1, w2, w3, w4, x)
            grad4, grad3, grad2, grad1, d5, d4, d3, d2 = self.backward(a2, a3, a4, a5, w1, w2, w3, w4, x, y)
            
            grad4_array.append(grad4)
            grad3_array.append(grad3)
            grad2_array.append(grad2)
            grad1_array.append(grad1)
            
        grad4_stack = np.stack(grad4_array)
        grad3_stack = np.stack(grad3_array)
        grad2_stack = np.stack(grad2_array)
        grad1_stack = np.stack(grad1_array)
        
        avg_grad4 = np.mean(grad4_stack, axis = 0)
        avg_grad3 = np.mean(grad3_stack, axis = 0)
        avg_grad2 = np.mean(grad2_stack, axis = 0)
        avg_grad1 = np.mean(grad1_stack, axis = 0)
    
        
        reg4 = [[0]+[element * lamb/ m for element in row[1:]] for row in w4]
        reg4 = np.array(reg4)
        reg_avg_grad4 = avg_grad4 + reg4
        
        reg3 = [[0]+[element * lamb/ m for element in row[1:]] for row in w3]
        reg3 = np.array(reg3)
        reg_avg_grad3 = avg_grad3 + reg3
    
        reg2 = [[0]+[element * lamb/ m for element in row[1:]] for row in w2]
        reg2 = np.array(reg2)
        reg_avg_grad2 = avg_grad2 + reg2
    
        reg1 = [[0]+[element * lamb/ m for element in row[1:]] for row in w1]
        reg1 = np.array(reg1)
        reg_avg_grad1 = avg_grad1 + reg1
    
        w4n = w4 - alpha*reg_avg_grad4
        w3n = w3 - alpha*reg_avg_grad3
        w2n = w2 - alpha*reg_avg_grad2
        w1n = w1 - alpha*reg_avg_grad1
        
        return w1n, w2n, w3n, w4n, reg_avg_grad1, reg_avg_grad2, reg_avg_grad3, reg_avg_grad4
    
    def predict(self, X_train, Y_train, X_test, nn1, nn2, nn3, alpha, lamb, max_itr, min_deriv):
        
        
        self.X_train = X_train
        self.Y_train = Y_train
        Xm_test, Xn_test = X_test.shape
    
        prediction = []
    
        w1, w2, w3, w4 = self.initial_params_new(nn1, nn2, nn3)
        
        i = 0
        while i < max_itr:
            w1n, w2n, w3n, w4n, avg_grad1, avg_grad2, avg_grad3, avg_grad4 = self.update_parameters(w1, w2, w3, w4, alpha, lamb)
            w1, w2, w3, w4 = w1n, w2n, w3n, w4n
            i += 1
        
        #Breaking if the gradient is smaller that the required precision(min_deriv)
            if avg_grad1.all() < min_deriv and avg_grad2.all() < min_deriv and avg_grad3.all() < min_deriv and avg_grad4.all() < min_deriv:
                break
            
            
        Y_unique_length = len(np.unique(self.data[:,0]))    
        for j in range(Xm_test):
            x = X_test[j]
            a1, a2, a3, a4, a5 = self.forward(w1, w2, w3, w4, x)
            arg_max = np.argmax(a5)
            predict = arg_max
                
            prediction.append(predict)
        
        return prediction   
    
    def run_NN3(self, nn1, nn2, nn3, step_size, Lambda, max_iteration, min_derivative):
    
        Y_unique_length = len(np.unique(self.data[:,0]))
    
            
        CV = CV_digits(self.data)
        
        partitions = CV.k_split(10)
        accuracy = []
        fscore = []
    
    
        for j in range(10):
            X_train, X_test, Y_train, Y_test = CV.k_fold(partitions,j)
            #Normalising the data coulmn wise to be between 0 and 1
            #X_train = (X_train - np.min(X_train, axis = 0))/(np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

            #Adding a bias coulmn to the training data
            X_train_b = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        
            #Normalising the testing data
            #X_test = (X_test - np.min(X_test, axis = 0))/(np.max(X_test, axis = 0) - np.min(X_test, axis = 0))
        
            #Adding a bias coulmn to the testing data
            X_test_b = np.hstack((np.ones((X_test.shape[0],1)),X_test))

            prediction = self.predict(X_train_b, Y_train, X_test_b, nn1, nn2, nn3, step_size, Lambda, max_iteration, min_derivative)
            
            M = np.zeros((Y_unique_length,Y_unique_length))
                
            for i in range(len(prediction)):
                M[Y_test[i].astype(int),prediction[i].astype(int)] += 1
                    
            P = []
            R = []
            A = []
            for i in range(Y_unique_length):
                p = M[i,i]/np.sum(M[:,i]) if M[i,i] != 0 else 0
                r = M[i,i]/np.sum(M[:,i]) if M[i,i] != 0 else 0
                P.append(p)
                R.append(r)
                A.append(M[i,i])
    
            acc = np.sum(A)/len(prediction)
            pre = np.mean(P)
            rec = np.mean(R)
            fs = 2*100*(pre * rec/(pre + rec)) if pre != 0 and rec != 0 else 0
            accuracy.append(acc)
            fscore.append(fs)
                
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_fscore = sum(fscore) / len(fscore)
    
        return avg_accuracy, avg_fscore


# In[4]:


class NN2:
    def __init__(self, data):
        self.data = data
        
    def hard_code(self, i):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        y = self.Y_train[i].astype(int)
        y_hard_code = Y_hard_code[y,:]
        
        return y_hard_code
    
    def activation(self, z):
        
        return 1 / (1 + np.exp(-z))
    
    def initial_params_new(self, nn1, nn2):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        p, q = self.X_train.shape
        w1 = np.random.normal(0, 1, (nn1, q))
        w2 = np.random.normal(0, 1, (nn2, nn1+1))
        w3 = np.random.normal(0, 1, (Y_unique_length, nn2+1))
        
        return w1, w2, w3
    
    def forward(self, w1, w2, w3, x):
        a1 = x
        z2 = np.dot(w1, a1.T)
        a2 = self.activation(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = np.dot(w2, a2.T)
        a3 = self.activation(z3)
        a3 = np.insert(a3, 0, 1)
        z4 = np.dot(w3, a3.T)
        a4 = self.activation(z4)
        return a1, a2, a3, a4
    
    def backward(self, a2, a3, a4, w1, w2, w3, x, y):
        a1 = x
        a1 = a1.reshape(1,-1)
        a2 = a2.reshape(1,-1)
        a3 = a3.reshape(1,-1)
        a4 = a4.reshape(1,-1)
        y = y.reshape(1,-1)
        d4 = a4 - y
        d4 = d4.reshape(-1,1)
        dp3 = np.dot(w3[:, 1:a3.size].T, d4)
        d3 = np.zeros((dp3.size,1))
        for i in range(dp3.size):
            N = a3[:,i+1]*(1-a3[:,i+1])
            d3[i] = dp3[i]*N
        dp2 = np.dot(w2[:,1:a2.size].T, d3)
        d2 =np.zeros((dp2.size,1))
        for i in range(dp2.size):
            N = a2[:,i+1]*(1-a2[:,i+1])
            d2[i] = dp2[i]*N 
        grad3 = np.dot(d4, a3)
        grad2 = np.dot(d3, a2)
        grad1 = np.dot(d2, a1)
        
        return grad3, grad2, grad1, d4, d3, d2
    
    
    def update_parameters(self, w1, w2, w3, alpha, lamb):
        grad3_array = []
        grad2_array = []
        grad1_array = []
        m, n = self.X_train.shape
        for i in range(m):
            x = self.X_train[i]
            y = self.hard_code(i)
            a1, a2, a3, a4 = self.forward(w1, w2, w3, x)
            grad3, grad2, grad1, d4, d3, d2 = self.backward(a2, a3, a4, w1, w2, w3, x, y)
            grad3_array.append(grad3)
            grad2_array.append(grad2)
            grad1_array.append(grad1)
        grad3_stack = np.stack(grad3_array)
        grad2_stack = np.stack(grad2_array)
        grad1_stack = np.stack(grad1_array)
        avg_grad3 = np.mean(grad3_stack, axis = 0)
        avg_grad2 = np.mean(grad2_stack, axis = 0)
        avg_grad1 = np.mean(grad1_stack, axis = 0)
    
        reg3 = [[0]+[element * lamb/ m for element in row[1:]] for row in w3]
        reg3 = np.array(reg3)
        reg_avg_grad3 = avg_grad3 + reg3
    
        reg2 = [[0]+[element * lamb/ m for element in row[1:]] for row in w2]
        reg2 = np.array(reg2)
        reg_avg_grad2 = avg_grad2 + reg2
    
        reg1 = [[0]+[element * lamb/ m for element in row[1:]] for row in w1]
        reg1 = np.array(reg1)
        reg_avg_grad1 = avg_grad1 + reg1
    
        w3n = w3 - alpha*reg_avg_grad3
        w2n = w2 - alpha*reg_avg_grad2
        w1n = w1 - alpha*reg_avg_grad1
        
        return w1n, w2n, w3n, reg_avg_grad1, reg_avg_grad2, reg_avg_grad3
    
    def predict(self, X_train, Y_train, X_test, nn1, nn2, alpha, lamb, max_itr, min_deriv):
        
        
        self.X_train = X_train
        self.Y_train = Y_train
        Xm_test, Xn_test = X_test.shape
    
        prediction = []
    
        w1, w2, w3 = self.initial_params_new(nn1, nn2)
        i = 0
        while i < max_itr:
            w1n, w2n, w3n, avg_grad1, avg_grad2, avg_grad3 = self.update_parameters(w1, w2, w3, alpha, lamb)
            w1, w2, w3 = w1n, w2n, w3n
            i += 1
        
        #Breaking if the gradient is smaller that the required precision(min_deriv)
            if avg_grad1.all() < min_deriv and avg_grad2.all() < min_deriv and avg_grad3.all() < min_deriv:
                break
            
            
        Y_unique_length = len(np.unique(self.data[:,0]))    
        
        for j in range(Xm_test):
            x = X_test[j]
            a1, a2, a3, a4 = self.forward(w1, w2, w3, x)
            arg_max = np.argmax(a4)
            predict = arg_max
            prediction.append(predict)
        
        return prediction   
    
    def run_NN2(self, nn1, nn2, step_size, Lambda, max_iteration, min_derivative):
    
        Y_unique_length = len(np.unique(self.data[:,0]))
    
        CV = CV_digits(self.data)
        
        partitions = CV.k_split(10)
        accuracy = []
        fscore = []
    
    
        for j in range(10):
            X_train, X_test, Y_train, Y_test = CV.k_fold(partitions,j)
            #Normalising the data coulmn wise to be between 0 and 1
            #X_train = (X_train - np.min(X_train, axis = 0))/(np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

            #Adding a bias coulmn to the training data
            X_train_b = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        
            #Normalising the testing data
            #X_test = (X_test - np.min(X_test, axis = 0))/(np.max(X_test, axis = 0) - np.min(X_test, axis = 0))
        
            #Adding a bias coulmn to the testing data
            X_test_b = np.hstack((np.ones((X_test.shape[0],1)),X_test))

            prediction = self.predict(X_train_b, Y_train, X_test_b, nn1, nn2, step_size, Lambda, max_iteration, min_derivative)
            
            M = np.zeros((Y_unique_length,Y_unique_length))
                
            for i in range(len(prediction)):
                M[Y_test[i].astype(int),prediction[i].astype(int)] += 1
                    
            P = []
            R = []
            A = []
            for i in range(Y_unique_length):
                p = M[i,i]/np.sum(M[:,i]) if M[i,i] != 0 else 0
                r = M[i,i]/np.sum(M[:,i]) if M[i,i] != 0 else 0
                P.append(p)
                R.append(r)
                A.append(M[i,i])
    
            acc = np.sum(A)/len(prediction)
            pre = np.mean(P)
            rec = np.mean(R)
            fs = 2*100*(pre * rec/(pre + rec)) if pre != 0 and rec != 0 else 0
            accuracy.append(acc)
            fscore.append(fs)
                
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_fscore = sum(fscore) / len(fscore)

    
        return avg_accuracy, avg_fscore


# In[13]:


class NN1:
    def __init__(self, data):
        self.data = data
        
    def hard_code(self, i):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        y = self.Y_train[i].astype(int)
        y_hard_code = Y_hard_code[y,:]
    
        
        return y_hard_code
    
    def activation(self, z):
        
        return 1 / (1 + np.exp(-z))
    
    def initial_params_new(self, nn):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        p, q = self.X_train.shape
        
        w1 = np.random.normal(0, 1, (nn, q))
        w2 = np.random.normal(0, 1, (Y_unique_length, nn+1))
        
        return w1, w2
    
    def forward(self, w1, w2, x):
        a1 = x
        z2 = np.dot(w1, a1.T)
        a2 = self.activation(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = np.dot(w2, a2.T)
        a3 = self.activation(z3)
        return a1, a2, a3
    
    def backward(self, a2, a3, w1, w2, x, y):
        a1 = x
        d3 = a3 - y
        d3 = d3.reshape(-1,1)
        #because w2 is a 1d array
        #might need to uncomment of the .txt
        #w2 = w2.reshape(1,-1)
        dp2 = np.dot(w2[:,1:a2.size].T, d3)
        d2 = np.zeros((dp2.size,1))
        for i in range(dp2.size):
            N = a2[i+1]*(1-a2[i+1])
            d2[i] = dp2[i]*N
        grad2 = a2.T*d3
        grad1 = a1.T*d2
        return grad2, grad1, d3, d2
    
    
    def update_parameters(self, w1, w2, alpha, lamb):
        grad2_array = []
        grad1_array = []
        m, n = self.X_train.shape

        for i in range(m):
            x = self.X_train[i]
            y = self.hard_code(i)
            a1, a2, a3 = self.forward(w1, w2, x)
            
            grad2, grad1, d3, d2 = self.backward(a2, a3, w1, w2, x, y)

            grad2_array.append(grad2)
            grad1_array.append(grad1)
       
        grad2_stack = np.stack(grad2_array)
        grad1_stack = np.stack(grad1_array)
        
        avg_grad2 = np.mean(grad2_stack, axis = 0)
        avg_grad1 = np.mean(grad1_stack, axis = 0)
    
       
        reg2 = [[0]+[element * lamb/ m for element in row[1:]] for row in w2]
        reg2 = np.array(reg2)
        reg_avg_grad2 = avg_grad2 + reg2
    
        reg1 = [[0]+[element * lamb/ m for element in row[1:]] for row in w1]
        reg1 = np.array(reg1)
        reg_avg_grad1 = avg_grad1 + reg1
    
       
        w2n = w2 - alpha*reg_avg_grad2
        w1n = w1 - alpha*reg_avg_grad1
        
        return w1n, w2n, reg_avg_grad1, reg_avg_grad2

    
    def cost_function(self, data, nn, alpha, lamb):
        
        i = 0
        np.random.shuffle(data)
        split_idx = int(0.7*len(data))
        train = data[:split_idx]
        x_train, y_train = train[:,1:], train[:,0]
        test = data[split_idx:]
        x_test, y_test = test[:,1:], test[:,0]
        
        Y_unique_length = len(np.unique(train[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        
        p,q = x_train.shape
        
        w1 = np.random.normal(0, 1, (nn, q))
        w2 = np.random.normal(0, 1, (Y_unique_length, nn+1))
        
        Cost = []
        
        for i in range(len(train)):
            x = x_train[i]
            if Y_unique_length == 3:
                y = Y_hard_code[y_train[i].astype(int)-1,:]
            else:
                y = Y_hard_code[y_train[i].astype(int),:]
            
            a1, a2, a3 = self.forward(w1, w2, x)
            grad2, grad1, d3, d2 = self.backward(a2, a3, w1, w2, x, y)
            
            reg2 = [[0]+[element * lamb for element in row[1:]] for row in w2]
            reg2 = np.array(reg2)
            reg_avg_grad2 = grad2 + reg2
    
            reg1 = [[0]+[element * lamb for element in row[1:]] for row in w1]
            reg1 = np.array(reg1)
            reg_avg_grad1 = grad1 + reg1
            
            w2n = w2 - alpha*reg_avg_grad2
            w1n = w1 - alpha*reg_avg_grad1
            
            w1 = w1n
            w2 = w2n
            J = []
        
            for i in range(len(test)):
                
                x = x_test[i]
                
                if Y_unique_length == 3:
                    y = Y_hard_code[y_train[i].astype(int)-1,:]
                else:
                    y = Y_hard_code[y_train[i].astype(int),:]
                    
                a1, a2, a3 = self.forward(w1, w2, x)
                j = -y*np.log(a3) - (1-y)*np.log(1-a3)
                J.append(np.sum(j))
                
            L = lamb/(2*len(test))
            S = L*(sum(sum(w**2 for w in row[1:])for row in w1) + sum(sum(w**2 for w in row[1:])for row in w2))    
            CJ = np.sum(J)/len(test) + S
            
            Cost.append(CJ)
            
        return Cost
    
    def predict(self, X_train, Y_train, X_test, nn, alpha, lamb, max_itr, min_deriv):
        
        
        self.X_train = X_train
        self.Y_train = Y_train
        Xm_test, Xn_test = X_test.shape
    
        prediction = []
    
        w1, w2= self.initial_params_new(nn)
        i = 0
        while i < max_itr:
            w1n, w2n, avg_grad1, avg_grad2 = self.update_parameters(w1, w2, alpha, lamb)
            w1, w2 = w1n, w2n
            i += 1
        
        #Breaking if the gradient is smaller that the required precision(min_deriv)
            if avg_grad1.all() < min_deriv and avg_grad2.all() < min_deriv:
                break
            
        Y_unique_length = len(np.unique(self.data[:,0]))    
        for j in range(Xm_test):
            x = X_test[j]
            a1, a2, a3 = self.forward(w1, w2, x)
            arg_max = np.argmax(a3)
            predict = arg_max
                
            prediction.append(predict)
        
        return prediction
    
    def run_NN1(self, nn, step_size, Lambda, max_iteration, min_derivative):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
    
        CV = CV_digits(self.data)
        partitions = CV.k_split(10)
        accuracy = []
        fscore = []
    
    
        for j in range(10):
            X_train, X_test, Y_train, Y_test = CV.k_fold(partitions,j)
            #Normalising the data coulmn wise to be between 0 and 1
            #X_train = (X_train - np.min(X_train, axis = 0))/(np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

            #Adding a bias coulmn to the training data
            X_train_b = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        
            #Normalising the testing data
            #X_test = (X_test - np.min(X_test, axis = 0))/(np.max(X_test, axis = 0) - np.min(X_test, axis = 0))
        
            #Adding a bias coulmn to the testing data
            X_test_b = np.hstack((np.ones((X_test.shape[0],1)),X_test))

            prediction = self.predict(X_train_b, Y_train, X_test_b, nn, step_size, Lambda, max_iteration, min_derivative)
    
            M = np.zeros((Y_unique_length,Y_unique_length))
                
            for i in range(len(prediction)):
                M[Y_test[i].astype(int),prediction[i].astype(int)] += 1
                    
            P = []
            R = []
            A = []
            for i in range(Y_unique_length):
                p = M[i,i]/np.sum(M[:,i]) if M[i,i] != 0 else 0
                r = M[i,i]/np.sum(M[:,i]) if M[i,i] != 0 else 0
                P.append(p)
                R.append(r)
                A.append(M[i,i])
    
            acc = np.sum(A)/len(prediction)
            pre = np.mean(P)
            rec = np.mean(R)
            fs = 2*100*(pre * rec/(pre + rec)) if pre != 0 and rec != 0 else 0
            accuracy.append(acc)
            fscore.append(fs)
                
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_fscore = sum(fscore) / len(fscore)
    
        return avg_accuracy, avg_fscore
    
        


# In[6]:





# In[11]:


digits = datasets.load_digits(return_X_y = True)
digits_datasets_X = digits[0]
digits_datasets_y = digits[1]
digits_datasets_y  = digits_datasets_y.reshape(-1,1)

N = len(digits_datasets_X)

data_digits = np.concatenate((digits_datasets_y,digits_datasets_X), axis=1)


# In[2]:


nn1 = 10
nn2 = 0
nn3 = 0
step_size = 1
Lambda = 0.01
max_iteration = 500
min_derivative = 1e-10

clf = NN1(data_digits)
acc, fs = clf.run_NN1(nn1, step_size, Lambda, max_iteration, min_derivative)


# In[ ]:





# In[1]:


m = int(0.7 *data_digits.shape[0])
CJ = clf.cost_function(data_digits, nn1, step_size, Lambda)
x = np.arange(1,m+1)
plt.plot(x, CJ)
plt.xlabel("The number of instances")
plt.ylabel("Performance")
plt.show()


# In[ ]:





# In[ ]:




