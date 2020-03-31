import numpy as np
import pandas as pd
import operator
from numpy.random import random 
import matplotlib.pyplot as plt 
import sys, getopt
import pickle
import auxfuns




'''
run with:

python3 AdaBoost.py
'''


def stump(data_v, cutoff, side):

    res = np.ones(len(data_v))

    if side == '+':       
        res[data_v <= cutoff] = -1
    if side == '-':
        res[data_v > cutoff] = -1
    
    return res
        
    
def sel_feature(data_m, label_v, sample_weight_v):
    
    n, m = data_m.shape
    sel_stump = {}
    error_min = np.inf
    
    for i in range(m):   
        data_v = data_m.iloc[:, i]
        minv = data_v.min()
        maxv = data_v.max()
        steps = [-1] + [minv + s*(maxv - minv)/10 for s in list(range(11))]

        for j in range(len(steps)):
            for x in ['+', '-']:
                pred = stump(data_v, steps[j], x)
                a = [0 if pred[k] == label_v[k] else 1 for k in range(len(pred))]
                error_iter = np.mat(np.array(sample_weight_v).reshape(n, 1)).T*np.mat(np.array(a).reshape(n, 1))

                if error_iter < error_min:
                    error_min = error_iter
                    sel_pred = pred.copy()
                    sel_stump['sel_feature'] = i
                    sel_stump['sel_cutoff'] = steps[j]
                    sel_stump['sel_side'] = x

    return error_min, sel_pred, sel_stump



def AdaBoost_train(data_m, label_v, iters):
    
    n, m = data_m.shape
    sample_weight_v = np.ones(n)/n
    stumps = []
    labels_pred = np.zeros(n)
    
    for i in range(iters):
        error, pred, sel_stump = sel_feature(data_m, label_v, sample_weight_v)
        a = 0.5*np.log((1 - error)/(error + 1e-10))
        sel_stump['a'] = a
        stumps.append(sel_stump)

        sample_weight_v = np.multiply(sample_weight_v, np.exp(np.multiply(-1*a*np.mat(label_v), pred)))
        sample_weight_v/sample_weight_v.sum()

        labels_pred = labels_pred + a*pred
        error2 = np.multiply(np.sign(labels_pred) != label_v, np.ones(n))
        error_rate = np.mean(error2)
        
        if error_rate == 0:
            break
            
    return stumps



def AdaBoost_classifier(data_m, stump_v):
            
    n = data_m.shape[0]
    
    labels_pred = np.zeros(n)
    for i in range(len(stump_v)):

        model = stump_v[i]
        pred = stump(data_m.iloc[:, model['sel_feature']], model['sel_cutoff'], model['sel_side'])
        labels_pred = labels_pred + model['a']*pred
    
    return np.sign(labels_pred)
                
                
            
        
if __name__ == '__main__':


    # read in data
    train_set = pd.read_csv('../Toydata/mnist_27_training.csv', index_col = 0)
    test_set = pd.read_csv('../Toydata/mnist_27_testing.csv', index_col = 0)

    train_labels = [1 if x == 7 else -1 for x in train_set.y]
    test_labels = [1 if x == 7 else -1 for x in test_set.y]

    
    stump_v = AdaBoost_train(train_set.iloc[:, 1:3], train_labels, 20)
    pred = AdaBoost_classifier(test_set.iloc[:, 1:3], stump_v)

    diff = test_labels - pred
    error_rate = 1 - np.array(diff == 0).reshape(len(test_labels)).sum()/diff.shape[1]

    print('Error rate: ', round(error_rate, 3))
