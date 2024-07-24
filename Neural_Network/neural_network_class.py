#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Cross_validation import CV_wine, CV_house

#data_wine = pd.read_csv("hw3_wine.csv", header=None, skiprows=1)
#data_wine = data_wine[0].str.split(expand=True)
#data_wine = data_wine.apply(pd.to_numeric)
#data_wine = data_wine.values
#np.random.shuffle(data_wine)


# In[ ]:





# In[1]:


class NN3:
    def __init__(self, data):
        self.data = data
        
    def hard_code(self, i):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        y = self.Y_train[i].astype(int)
        
        if Y_unique_length == 3:
            y_hard_code = Y_hard_code[y-1,:]
        else:
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
            
            if Y_unique_length == 3:
                predict = arg_max+1
            else:
                predict = arg_max
                
            prediction.append(predict)
        
        return prediction   
    
    def run_NN3(self, nn1, nn2, nn3, step_size, Lambda, max_iteration, min_derivative):
    
        Y_unique_length = len(np.unique(self.data[:,0]))
    
        if Y_unique_length == 3:
            CV = CV_wine(self.data)
        else:
            CV = CV_house(self.data)
    
        partitions = CV.k_split(6)
        accuracy = []
        fscore = []
    
    
        for j in range(6):
            X_train, X_test, Y_train, Y_test = CV.k_fold(partitions,j)
            #Normalising the data coulmn wise to be between 0 and 1
            X_train = (X_train - np.min(X_train, axis = 0))/(np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

            #Adding a bias coulmn to the training data
            X_train_b = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        
            #Normalising the testing data
            X_test = (X_test - np.min(X_test, axis = 0))/(np.max(X_test, axis = 0) - np.min(X_test, axis = 0))
        
            #Adding a bias coulmn to the testing data
            X_test_b = np.hstack((np.ones((X_test.shape[0],1)),X_test))

            prediction = self.predict(X_train_b, Y_train, X_test_b, nn1, nn2, nn3, step_size, Lambda, max_iteration, min_derivative)
            print("prediction is:", prediction)
            print("Y_test is:", Y_test)
            if len(np.unique(Y_test)) == 3:
                m11, m12, m13 = 0, 0, 0
                m21, m22, m23 = 0, 0, 0
                m31, m32, m33 = 0, 0, 0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and Y_test[i] == 1:
                        m11 += 1
                    if prediction[i] == 2 and Y_test[i] == 2:
                        m22 += 1
                    if prediction[i] == 3 and Y_test[i] == 3:
                        m33 += 1
                    if prediction[i] == 1 and Y_test[i] == 2:
                        m21 += 1
                    if prediction[i] == 1 and Y_test[i] == 3:
                        m31 += 1
                    if prediction[i] == 2 and Y_test[i] == 1:
                        m12 += 1
                    if prediction[i] == 2 and Y_test[i] == 3:
                        m32 += 1
                    if prediction[i] == 3 and Y_test[i] == 1:
                        m13 += 1
                    if prediction[i] == 3 and Y_test[i] == 2:
                        m23 += 1
                acc = (m11 + m22 + m33)*100 / len(prediction)
                p1 = m11 / (m11 + m21 + m31) if m11 != 0 else 0
                p2 = m22 / (m22 + m12 + m32) if m22 != 0 else 0
                p3 = m33 / (m33 + m13 + m23) if m33 != 0 else 0
                pre = (p1 + p2 + p3)*100 / 3
                r1 = m11 / (m11 + m12 + m13) if m11 != 0 else 0
                r2 = m22 / (m22 + m21 + m23) if m22 != 0 else 0
                r3 = m33 / (m33 + m31 + m32) if m33 != 0 else 0
                rec = (r1 + r2 + r3)*100 / 3
                f1 = 2*(p1 * r1/(p1 + r1)) if p1 != 0 and r1 != 0 else 0
                f2 = 2*(p2 * r2/(p2 + r2)) if p2 != 0 and r2 != 0 else 0
                f3 = 2*(p3 * r3/(p3 + r3)) if p3 != 0 and r3 != 0 else 0
                fs = (f1 + f2 + f3)*100/3
                fscore.append(fs)
                accuracy.append(acc)
                #avg_accuracy = sum(accuracy) / len(accuracy)
                #avg_fscore = sum(fscore) / len(fscore)
            else:
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and Y_test[i] == 1:
                        tp += 1
                    if prediction[i] == 0 and Y_test[i] == 0:
                        tn += 1
                    if prediction[i] == 0 and Y_test[i] == 1:
                        fn += 1 
                    if prediction[i] == 1 and Y_test[i] == 0:
                        fp += 1
                acc = 100*(tp + tn) / len(prediction)
                pre = tp / (tp + fp) if tp != 0 else 0
                rec = tp / (tp + fn) if tn != 0 else 0
                fs = 2*100*(pre * rec/(pre + rec)) if pre != 0 and rec != 0 else 0
                accuracy.append(acc)
                fscore.append(fs)
                
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_fscore = sum(fscore) / len(fscore)
    
        return avg_accuracy, avg_fscore


# In[2]:


class NN2:
    def __init__(self, data):
        self.data = data
        
    def hard_code(self, i):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        y = self.Y_train[i].astype(int)
        
        if Y_unique_length == 3:
            y_hard_code = Y_hard_code[y-1,:]
        else:
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
            if Y_unique_length == 3:
                predict = arg_max+1
            else:
                predict = arg_max
                
            prediction.append(predict)
        
        return prediction   
    
    def run_NN2(self, nn1, nn2, step_size, Lambda, max_iteration, min_derivative):
    
        Y_unique_length = len(np.unique(self.data[:,0]))
    
        if Y_unique_length == 3:
            CV = CV_wine(self.data)
        else:
            CV = CV_house(self.data)
    
        partitions = CV.k_split(6)
        accuracy = []
        fscore = []
    
    
        for j in range(6):
            X_train, X_test, Y_train, Y_test = CV.k_fold(partitions,j)
            #Normalising the data coulmn wise to be between 0 and 1
            X_train = (X_train - np.min(X_train, axis = 0))/(np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

            #Adding a bias coulmn to the training data
            X_train_b = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        
            #Normalising the testing data
            X_test = (X_test - np.min(X_test, axis = 0))/(np.max(X_test, axis = 0) - np.min(X_test, axis = 0))
        
            #Adding a bias coulmn to the testing data
            X_test_b = np.hstack((np.ones((X_test.shape[0],1)),X_test))

            prediction = self.predict(X_train_b, Y_train, X_test_b, nn1, nn2, step_size, Lambda, max_iteration, min_derivative)
            print("prediction is:", prediction)
            print("Y_test is:", Y_test)
            if len(np.unique(Y_test)) == 3:
                m11, m12, m13 = 0, 0, 0
                m21, m22, m23 = 0, 0, 0
                m31, m32, m33 = 0, 0, 0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and Y_test[i] == 1:
                        m11 += 1
                    if prediction[i] == 2 and Y_test[i] == 2:
                        m22 += 1
                    if prediction[i] == 3 and Y_test[i] == 3:
                        m33 += 1
                    if prediction[i] == 1 and Y_test[i] == 2:
                        m21 += 1
                    if prediction[i] == 1 and Y_test[i] == 3:
                        m31 += 1
                    if prediction[i] == 2 and Y_test[i] == 1:
                        m12 += 1
                    if prediction[i] == 2 and Y_test[i] == 3:
                        m32 += 1
                    if prediction[i] == 3 and Y_test[i] == 1:
                        m13 += 1
                    if prediction[i] == 3 and Y_test[i] == 2:
                        m23 += 1
                acc = (m11 + m22 + m33)*100 / len(prediction)
                p1 = m11 / (m11 + m21 + m31) if m11 != 0 else 0
                p2 = m22 / (m22 + m12 + m32) if m22 != 0 else 0
                p3 = m33 / (m33 + m13 + m23) if m33 != 0 else 0
                pre = (p1 + p2 + p3)*100 / 3
                r1 = m11 / (m11 + m12 + m13) if m11 != 0 else 0
                r2 = m22 / (m22 + m21 + m23) if m22 != 0 else 0
                r3 = m33 / (m33 + m31 + m32) if m33 != 0 else 0
                rec = (r1 + r2 + r3)*100 / 3
                f1 = 2*(p1 * r1/(p1 + r1)) if p1 != 0 and r1 != 0 else 0
                f2 = 2*(p2 * r2/(p2 + r2)) if p2 != 0 and r2 != 0 else 0
                f3 = 2*(p3 * r3/(p3 + r3)) if p3 != 0 and r3 != 0 else 0
                fs = (f1 + f2 + f3)*100/3
                fscore.append(fs)
                accuracy.append(acc)
                #avg_accuracy = sum(accuracy) / len(accuracy)
                #avg_fscore = sum(fscore) / len(fscore)
            else:
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and Y_test[i] == 1:
                        tp += 1
                    if prediction[i] == 0 and Y_test[i] == 0:
                        tn += 1
                    if prediction[i] == 0 and Y_test[i] == 1:
                        fn += 1 
                    if prediction[i] == 1 and Y_test[i] == 0:
                        fp += 1
                acc = 100*(tp + tn) / len(prediction)
                pre = tp / (tp + fp) if tp != 0 else 0
                rec = tp / (tp + fn) if tn != 0 else 0
                fs = 2*100*(pre * rec/(pre + rec)) if pre != 0 and rec != 0 else 0
                accuracy.append(acc)
                fscore.append(fs)
                
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_fscore = sum(fscore) / len(fscore)

    
        return avg_accuracy, avg_fscore


# In[3]:


class NN1:
    def __init__(self, data):
        self.data = data
        
    def hard_code(self, i):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
        Y_hard_code = np.eye(Y_unique_length)
        y = self.Y_train[i].astype(int)
        if Y_unique_length == 3:
            y_hard_code = Y_hard_code[y-1,:]
        else:
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
            if Y_unique_length == 3:
                predict = arg_max+1
            else:
                predict = arg_max
                
            prediction.append(predict)
        
        return prediction
    
    def run_NN1(self, nn, step_size, Lambda, max_iteration, min_derivative):
        
        Y_unique_length = len(np.unique(self.data[:,0]))
    
        if Y_unique_length == 3:
            CV = CV_wine(self.data)
        else:
            CV = CV_house(self.data)
    
        partitions = CV.k_split(6)
        accuracy = []
        fscore = []
    
    
        for j in range(6):
            X_train, X_test, Y_train, Y_test = CV.k_fold(partitions,j)
            #Normalising the data coulmn wise to be between 0 and 1
            X_train = (X_train - np.min(X_train, axis = 0))/(np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

            #Adding a bias coulmn to the training data
            X_train_b = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        
            #Normalising the testing data
            X_test = (X_test - np.min(X_test, axis = 0))/(np.max(X_test, axis = 0) - np.min(X_test, axis = 0))
        
            #Adding a bias coulmn to the testing data
            X_test_b = np.hstack((np.ones((X_test.shape[0],1)),X_test))

            prediction = self.predict(X_train_b, Y_train, X_test_b, nn, step_size, Lambda, max_iteration, min_derivative)
            print("prediction is:", prediction)
            print("Y_test is:", Y_test)
        
            if len(np.unique(Y_test)) == 3:
                m11, m12, m13 = 0, 0, 0
                m21, m22, m23 = 0, 0, 0
                m31, m32, m33 = 0, 0, 0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and Y_test[i] == 1:
                        m11 += 1
                    if prediction[i] == 2 and Y_test[i] == 2:
                        m22 += 1
                    if prediction[i] == 3 and Y_test[i] == 3:
                        m33 += 1
                    if prediction[i] == 1 and Y_test[i] == 2:
                        m21 += 1
                    if prediction[i] == 1 and Y_test[i] == 3:
                        m31 += 1
                    if prediction[i] == 2 and Y_test[i] == 1:
                        m12 += 1
                    if prediction[i] == 2 and Y_test[i] == 3:
                        m32 += 1
                    if prediction[i] == 3 and Y_test[i] == 1:
                        m13 += 1
                    if prediction[i] == 3 and Y_test[i] == 2:
                        m23 += 1
                acc = (m11 + m22 + m33)*100 / len(prediction)
                p1 = m11 / (m11 + m21 + m31) if m11 != 0 else 0
                p2 = m22 / (m22 + m12 + m32) if m22 != 0 else 0
                p3 = m33 / (m33 + m13 + m23) if m33 != 0 else 0
                pre = (p1 + p2 + p3)*100 / 3
                r1 = m11 / (m11 + m12 + m13) if m11 != 0 else 0
                r2 = m22 / (m22 + m21 + m23) if m22 != 0 else 0
                r3 = m33 / (m33 + m31 + m32) if m33 != 0 else 0
                rec = (r1 + r2 + r3)*100 / 3
                f1 = 2*(p1 * r1/(p1 + r1)) if p1 != 0 and r1 != 0 else 0
                f2 = 2*(p2 * r2/(p2 + r2)) if p2 != 0 and r2 != 0 else 0
                f3 = 2*(p3 * r3/(p3 + r3)) if p3 != 0 and r3 != 0 else 0
                fs = (f1 + f2 + f3)*100/3
                fscore.append(fs)
                accuracy.append(acc)
                #avg_accuracy = sum(accuracy) / len(accuracy)
                #avg_fscore = sum(fscore) / len(fscore)
            else:
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for i in range(len(prediction)):
                    if prediction[i] == 1 and Y_test[i] == 1:
                        tp += 1
                    if prediction[i] == 0 and Y_test[i] == 0:
                        tn += 1
                    if prediction[i] == 0 and Y_test[i] == 1:
                        fn += 1 
                    if prediction[i] == 1 and Y_test[i] == 0:
                        fp += 1
                acc = 100*(tp + tn) / len(prediction)
                pre = tp / (tp + fp) if tp != 0 else 0
                rec = tp / (tp + fn) if tn != 0 else 0
                fs = 2*100*(pre * rec/(pre + rec)) if pre != 0 and rec != 0 else 0
                accuracy.append(acc)
                fscore.append(fs)
                
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_fscore = sum(fscore) / len(fscore)
    
        return avg_accuracy, avg_fscore
    
        


# In[14]:


#import pandas as pd
#import numpy as np
#df = pd.read_csv("house_votes_84.csv")
#cols = list(df.columns)
#cols = [cols[-1]] + cols[:-1]

#df = df[cols]
#data_house_votes = df.values

#data_wine = pd.read_csv("hw3_wine.csv", header=None, skiprows=1)
#data_wine = data_wine[0].str.split(expand=True)
#data_wine = data_wine.apply(pd.to_numeric)
#data_wine = data_wine.values
#np.random.shuffle(data_wine)


# In[ ]:






# In[ ]:





# In[ ]:




