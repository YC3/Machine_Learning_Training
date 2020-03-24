import numpy as np
import pandas as pd
import operator
import math
import matplotlib.pyplot as plt 
import sys, getopt
sys.path.append('../')
import auxfuns 

'''
Parameters:
-k, --k: number of centroids

Examples:
python3 kMeans.py -k 2

''' 


def dist_euc(v1, v2):
    
    return np.sqrt(sum(np.power(v1 - v2, 2)))


def starting_cent(data, k):
    
    d = data.shape[1]  # dimension
    cents = np.zeros((k, d))
    for i in range(d):
        minv = min(data.iloc[:, i])
        maxv = max(data.iloc[:, i])

        cents[:, i] = minv + np.random.rand(k)*float(maxv - minv)
    
    return cents


def kMeans(data, k, distFUN = dist_euc):
    
    m = data.shape[0]
    class_labels = np.zeros((m, 2))
    cents = starting_cent(data, k)
    
    change_class = True
    
    while change_class:
        
        change_class = False
        
        for i in range(m):
            closest_dist = np.inf
            closest_cent = -1
            for c in range(k):
                d = distFUN(data.iloc[i, :], cents[c, :])
                if d < closest_dist:
                    closest_dist = d
                    closest_cent = c
            if class_labels[i, 0] != closest_cent:
                change_class = True
            class_labels[i, :] = (closest_cent, closest_dist)
        
        # reconpute centroids
        for c in range(k):
            cents[c, :] = np.mean(data.loc[class_labels[:, 0] == c, :], axis = 0)
    
    return cents, class_labels
    

    
    
if __name__ == '__main__':

    # get user input
    opts,args = getopt.getopt(sys.argv[1:], '-k:', ['k='])
    for opt_name, opt_value in opts:
        if opt_name in ('-k', '--k'):
            k = int(opt_value)

    # read in data
    data = pd.read_csv('../Toydata/mnist_27_training.csv', index_col = 0).iloc[0:20, 1:]

    # clustering
    centroids, clusters = kMeans(data, k, distFUN = dist_euc)

    print('Clustering result:')
    print(clusters)
    
