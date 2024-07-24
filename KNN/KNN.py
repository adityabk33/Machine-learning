# Import libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from statistics import mode
from sklearn.model_selection import train_test_split
import statistics
from matplotlib import pyplot as plt

class KNN:
    def __init__(self, k, X, Y):
        self.k_train = k
        self.X_train = X
        self.Y_train = Y
        
    def result(self, Z):
        self.Z_train = Z
        prediction =[]
        for z_train in self.Z_train:
            distances = [np.sqrt(np.sum((z_train-x_train)**2)) for x_train in self.X_train]
            knn_ids = np.argsort(distances)[:self.k_train]
            knn_labels = [self.Y_train[i] for i in knn_ids]
            most_common = mode(knn_labels)
            prediction.append(most_common)
        return prediction

#Loading the dataset.

iris = datasets.load_iris()
X, Y = iris.data, iris.target


#For odd values of k
    
kodd = [x for x in range(52) if x%2==1]

#Looping over 20 times while changing the split for the training set.

accuracy_avg_training = []
accuracy_std_training = []

for i in kodd:
    avg_acc = []
    for j in range(20):
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)
        
        #Normalising
        
        min_vals_train = np.min(X_train, axis=0)
        max_vals_train = np.max(X_train, axis=0)
        X_train_n = (X_train - min_vals_train) / (max_vals_train - min_vals_train)
        
        min_vals_test = np.min(X_test, axis=0)
        max_vals_test = np.max(X_test, axis=0)
        X_test_n = (X_test - min_vals_test) / (max_vals_test - min_vals_test)

        KNN_classifier = KNN(k=i, X = X_train_n, Y = Y_train)
        species = KNN_classifier.result(X_train_n)
        accuracy = np.sum(species == Y_train) / len(Y_train)
        avg_acc.append(accuracy)

    accuracy_avg_training.append(np.mean(avg_acc))
    accuracy_std_training.append(np.std(avg_acc))
    
    
plt.errorbar(kodd, accuracy_avg_training,
            yerr=accuracy_std_training,
            fmt='-o',markersize=4,capsize=3)
plt.xlabel('kodd')
plt.ylabel('Accuracy')

plt.title('Accuracy vs. kodd for the training set')

plt.show()


#Looping over 20 times while changing the split for the testing set.

accuracy_avg_testing = []
accuracy_std_testing = []

for i in kodd:
    avg_acc = []
    for j in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)
        
        #Normalising
        
        min_vals_test = np.min(X_train, axis=0)
        max_vals_test = np.max(X_train, axis=0)
        X_train_n = (X_train - min_vals_train) / (max_vals_train - min_vals_train)
        
        min_vals_test = np.min(X_test, axis=0)
        max_vals_test = np.max(X_test, axis=0)
        X_test_n = (X_test - min_vals_test) / (max_vals_test - min_vals_test)
        
        KNN_classifier = KNN(k=i, X = X_train_n, Y = Y_train)
        species = KNN_classifier.result(X_test_n)
        accuracy = np.sum(species == Y_test) / len(Y_test)
        avg_acc.append(accuracy)

    accuracy_avg_testing.append(np.mean(avg_acc))
    accuracy_std_testing.append(np.std(avg_acc))
    

plt.errorbar(kodd, accuracy_avg_testing,
            yerr=accuracy_std_testing,
            fmt='-o',markersize=4,capsize=3)
plt.xlabel('kodd')
plt.ylabel('Accuracy')

plt.title('Accuracy vs. kodd for the testing set')

plt.show()


#Now we work with the training set after removing the normalization

accuracy_avg_testing1 = []
accuracy_std_testing1 = []

for i in kodd:
    avg_acc = []
    for j in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)
        KNN_classifier = KNN(k=i, X = X_train, Y = Y_train)
        species = KNN_classifier.result(X_test)
        accuracy = np.sum(species == Y_test) / len(Y_test)
        avg_acc.append(accuracy)
    accuracy_avg_testing1.append(np.mean(avg_acc))
    accuracy_std_testing1.append(np.std(avg_acc))
    
plt.errorbar(kodd, accuracy_avg_testing1,
            yerr=accuracy_std_testing1,
            fmt='-o',markersize=4,capsize=3)
plt.xlabel('kodd')
plt.ylabel('Accuracy')

plt.title('Accuracy vs. kodd for the testing set after removing normalisation')

plt.show()





# In[ ]:
