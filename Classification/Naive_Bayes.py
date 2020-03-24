import numpy as np
import pandas as pd
import operator
import math
import matplotlib.pyplot as plt 
import sys, getopt
sys.path.append('../')
import auxfuns 


'''
Author: YC3

Description:
Naive Bayes for document classification.

Example:
python3 Naive_Bayes.py

''' 


def NB_train(data):
    
    n = data.shape[0]
    m = data.shape[1] - 1
    
    # p(class=1)
    labels = data.iloc[:, 0]
    p = sum(labels)/float(n)
        
    # laplace smoothing
    p1_num, p1_denom = np.ones(m), 2.0
    p0_num, p0_denom = np.ones(m), 2.0
    
    for i in range(n):
        if labels[i] == 1:
            p1_num += data.iloc[i, 1:]
            p1_denom += sum(data.iloc[i, 1:])
        else:
            p0_num += data.iloc[i, 1:]
            p0_denom += sum(data.iloc[i, 1:])

    # use log() to avoid underflow
    p1 = np.log(p1_num/p1_denom)
    p0 = np.log(p0_num/p0_denom)

    return p0, p1, p  


def NB_classifier(example, p0, p1, p):
    
    p_1 = sum(example * p1) + np.log(p)
    p_0 = sum(example * p0) + np.log(1.0 - p)
    
    if p_1 > p_0:
        return 1
    else:
        return 0
    

if __name__ == '__main__':

    # Read in data
    data = pd.read_csv('../Toydata/document.csv', index_col = 0)
    train_set = data[0:6]
    test_set = data[6:8]
    
    # estimat parameters
    p0, p1, p = NB_train(train_set)
    
    # classification
    pred = []
    for i in range(len(test_set)):
        lab = NB_classifier(test_set.iloc[i, 1:], p0, p1, p)
        pred = pred + [lab]

    print('Predicting the labels of 2 new docs.')
    print('The true labels are: ', list(test_set.iloc[:, 0]))
    print('The predicted labels are: ', list(pred))
    
