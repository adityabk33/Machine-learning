
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from neural_network_class import NN3, NN2, NN1

data_wine = pd.read_csv("hw3_wine.csv", header=None, skiprows=1)
data_wine = data_wine[0].str.split(expand=True)
data_wine = data_wine.apply(pd.to_numeric)
data_wine = data_wine.values
np.random.shuffle(data_wine)

df = pd.read_csv("house_votes_84.csv")
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]
data_house_votes = df.values

def layers(data, L,nn1,nn2,nn3,step_size, Lambda, max_iteration, min_derivative):
    
    if L == 1:
        clf = NN1(data)
        acc, fs = clf.run_NN1(nn1, step_size, Lambda, max_iteration, min_derivative)
        
    if L == 2:
        clf = NN2(data)
        acc, fs = clf.run_NN2(nn1,nn2, step_size, Lambda, max_iteration, min_derivative)
        
    if L == 3:
        clf = NN3(data)
        acc, fs = clf.run_NN3(nn1, nn2, nn3, step_size, Lambda, max_iteration, min_derivative)
        
    print("My accuracy is:", acc)
    print("My Fscore is:", fs)

# Change the data between data_wine or data_house_votes.
# I am using k=6 folds for both the data sets
# Change the number of neurons by changing the value of L by choosing between 1 2 or 3. nn1, nn2, nn3 are the
#number of neurons in each layer. 
# If you are using one layer then input nn1 = number of neurons, nn2, nn3 = 0.
# If you are using two layes then nn1 = number of neurons, nn2 = number of neurons, n3= 0.
# If you are using three layers then nn1,nn2,nn3 = number of neurons you want.
# keep step_size  = 0.1 and max_itr = 500 and min_derivative = 1e-10.
# choose Lambda(regularization parmeter) as you want.
data = data_wine
clf1 = NN1(data)
m = int(0.7 * data.shape[0])
L = 1
nn1 = 10
nn2 = 0
nn3 = 0
step_size = 0.1
Lambda = 0
max_iteration = 500
min_derivative = 1e-10

layers(data,L, nn1, nn2, nn3, step_size, Lambda, max_iteration, min_derivative)


CJ = clf1.cost_function(data, nn1, step_size, Lambda)
x = np.arange(1,m+1)
plt.plot(x, CJ)
plt.xlabel("The number of instances")
plt.ylabel("Performance")
plt.show()




