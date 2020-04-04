import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt 
import sys, getopt
import pickle
import auxfuns



'''
Author: YC3

Description:
Useful auxiliary functions.
''' 



## scale data to [0, 1] range
def scaling01(data):

    dmin = data.min()
    dmax = data.max()
    return (data - dmin)/(dmax - dmin)


## calculate error rate given prediction and true class labels
def cat_error_rate(label_true, label_pred):

    return sum(label_true != label_pred)/len(label_true)


## split dataset into training and testing sets
def trainTest_split(data, p_train = 0.8):

    # generate random indices
    nrow = data.shape[0]
    np.random.seed(2016)
    idx = np.random.choice(np.arange(nrow), size = nrow, replace = False)

    dataR = data.iloc[idx, :]
    
    # do separation accordingto specified proportion
    train_set = dataR.iloc[0:round(nrow*p_train), :]
    test_set = dataR.iloc[round(nrow*p_train):, :]
    
    return train_set, test_set


## split numeric values into categorical ones
def feature_cut(values, n_class):

    # create bins
    maxv = max(values)
    minv = min(values)
    inc = (maxv - minv)/n_class
    
    bins = np.array([minv])
    for i in range(n_class):
        bins = np.append(bins, minv + inc)
        minv = minv + inc

    bins = np.round(bins, 2)

    # split
    lab = 0
    cat_values = values.copy()
    
    for i in range(len(bins) - 1):
        idx = [True if bins[i] <= x < bins[i+1] else False for x in values]
        cat_values[idx] = str(lab)
        lab += 1
    
    idx = [True if x == bins[-1] else False for x in values]     
    cat_values[idx] = str(lab)
    
    return cat_values


## save and load trees
def tree_save(tree, filename):
    fw = open(filename, 'wb')
    pickle.dump(tree, fw)
    fw.close()

def tree_load(filename):
    fr = open(filename, "rb")
    return pickle.load(fr)


## coding binary class labels as 0s and 1s
def coding_label_bi(data, lab_1):
    data.iloc[:, 0] = [1 if i == lab_1 else 0 for i in data.iloc[:, 0]]
    return(data)



