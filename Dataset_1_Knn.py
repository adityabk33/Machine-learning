#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import statistics
from statistics import mode
from sklearn.model_selection import train_test_split


# In[2]:


digits = datasets.load_digits(return_X_y = True)
digits_datasets_X = digits[0]
digits_datasets_y = digits[1]
N = len(digits_datasets_X)


# In[12]:





# In[3]:


class KNN:
    def __init__(self, k, X, Y):
        self.k_train = k
        self.X_train = X
        self.Y_train = Y
        
    def result(self, Z):
        self.Z_test = Z
        prediction =[]
        for z_train in self.Z_test:
            distances = [np.sqrt(np.sum((z_train-x_train)**2)) for x_train in self.X_train]
            knn_ids = np.argsort(distances)[:self.k_train]
            knn_labels = [self.Y_train[i] for i in knn_ids]
            most_common = mode(knn_labels)
            prediction.append(most_common)
        return prediction


# In[ ]:





# In[ ]:





# In[ ]:


kodd = [x for x in range(52) if x%2==1]

accuracy_avg_testing = []
accuracy_std_testing = []
fs_avg_testing = []
fs_std_testing = []

for i in kodd:
    accuracy = []
    fscore = []
    
    for j in range(50):
        
        X_train, X_test, Y_train, Y_test = train_test_split(digits_datasets_X, digits_datasets_y, test_size=0.2, stratify = digits_datasets_y)

        KNN_classifier = KNN(k=i, X = X_train, Y = Y_train)
        prediction = KNN_classifier.result(X_test)
        M = np.zeros((10,10))
                
        for i in range(len(prediction)):
            M[Y_test[i].astype(int),prediction[i].astype(int)] += 1        
        P = []
        R = []
        A = []
        for i in range(10):
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
        #accuracy = np.sum(species == Y_test) / len(Y_test)
        #avg_acc.append(accuracy)

    accuracy_avg_testing.append(np.mean(accuracy))
    accuracy_std_testing.append(np.std(accuracy))
    fs_avg_testing.append(np.mean(fscore))
    fs_std_testing.append(np.std(fscore))
    
plt.errorbar(kodd, accuracy_avg_testing,
            yerr=accuracy_std_testing,
            fmt='-o',markersize=4,capsize=3)
plt.xlabel('kodd')
plt.ylabel('Accuracy')

plt.title('Accuracy vs. kodd for the testing set')

plt.show()

plt.errorbar(kodd, fs_avg_testing,
            yerr=fs_std_testing,
            fmt='-o',markersize=4,capsize=3)
plt.xlabel('kodd')
plt.ylabel('Fscore')

plt.title('Fscore vs. kodd for the testing set')

plt.show()


# In[ ]:





# In[ ]:




