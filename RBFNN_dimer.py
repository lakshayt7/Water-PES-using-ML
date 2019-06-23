import math
import sklearn
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
n = 1000
Input = np.array(genfromtxt('C:\\Users\\lakshay\\Desktop\\ML in Catalalysis\\train_data-1\\input_1.csv', delimiter=','))
Output = np.array(genfromtxt('C:\\Users\\lakshay\\Desktop\\ML in Catalalysis\\train_data-1\\Dipole_Output.csv', delimiter=','))
count = 0
for i in range(n):
    if(Output[i][0] == 9 and Output[i][1] == 9 and Output[i][2] == 9):
        count+=1
for i in range(n-count):
    if(Output[i][0] == 9 and Output[i][1] == 9 and Output[i][2] == 9):
        Input = np.delete(Input, i, 0)
        Output = np.delete(Output, i, 0)
Input = normalize(Input, 'l2', 0)
Output = normalize(Output, 'l2', 0)
n = n - count
NClust = 12
iter =100
Clust =  KMeans(NClust, max_iter = iter).fit(Input)
Centers = Clust.cluster_centers_
M = 0
for i in range(NClust):
    for j in range(NClust):
        M = max(M, np.linalg.norm(Centers[i] - Centers[j]))
sigma = M/math.sqrt(2*NClust)
