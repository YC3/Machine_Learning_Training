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

Parameters:
-m, --method: 'batch', 'sgd1' or 'sgd2'
-a, --alpha: step in gradient descent (< 0 values) or gradient ascent (> 0 values)
-i, --iternum: number of iterations in 'sgd2'

Examples:
1. batch GD:
python3 Logistic_regression.py -m batch -a 0.01 -i 150

2. SGD:
python3 Logistic_regression.py -m sgd1 -a 0.01 -i 150

3. SGD with decreasing step length and randomization:
python3 Logistic_regression.py -m sgd2 -a 0.01 -i 150

''' 


def sigmoid(input):

     # to avoid overflow
    if input.shape[0] == 1 and input < 0:
        return np.exp(input)/(1 + np.exp(input))

    return 1.0/(1 + np.exp(-input))


# calculate the gradient of entire dataset
def gradDescent(data, a = 0.01, iter_num = 500):
    
    lab_v = np.mat(data.iloc[:, 0]).transpose()
    
    b = np.ones((data.shape[0], 1))
    data_m = np.mat(np.append(b, data.iloc[:, 1:], axis = 1))

    m, n = data_m.shape
    
    weights_v = np.ones((n, 1))
    
    for k in range(iter_num):
        pred_v = sigmoid(data_m*weights_v)
        error_v = pred_v - lab_v
        
        weights_v = weights_v - a*data_m.transpose()*error_v
        
    return weights_v


# calculate the gradient of only 1 example
def StocGD_1(data, a = 0.01, iter_num = 500):
    
    lab_v = np.mat(data.iloc[:, 0]).transpose()
    
    b = np.ones((data.shape[0], 1))
    data_m = np.mat(np.append(b, data.iloc[:, 1:], axis = 1))

    m, n = data_m.shape

    weights_v = np.ones((n, 1))

    for i in range(iter_num):
        
        for x in range(m):
            pred = sigmoid(sum(data_m[x]*weights_v))
            error = pred - lab_v[x]

            weights_v = weights_v - a*data_m[x].transpose()*error
        
    return weights_v


# decreasing alpha each iteration, randomized
def StocGD_2(data, a = 0.01, iter_num = 500):
    
    lab_v = np.mat(data.iloc[:, 0]).transpose()
    
    b = np.ones((data.shape[0], 1))
    data_m = np.mat(np.append(b, data.iloc[:, 1:], axis = 1))

    m, n = data_m.shape
    
    weights_v = np.ones((n, 1))

    for i in range(iter_num):
        
        id_list = list(range(m))
        for x in range(m):
            
            a = 4/(1.0 + i + x) + a    # alpha decrease as iter_num increases
            
            rand_id = np.random.randint(0, len(id_list))
            
            pred = sigmoid(sum(data_m[rand_id]*weights_v))
            error = pred - lab_v[rand_id]

            weights_v = weights_v - a*data_m[rand_id].transpose()*error
            del(id_list[rand_id])
        
    return weights_v


def LR_classifier(example, weights):

    prob = sigmoid(sum(np.mat(np.append(1, example))*weights))
    if prob > 0.5: 
        return 1
    else: 
        return 0
    
    

if __name__ == '__main__':

    # set default
    iter_num = 150
    alpha = 0.01
    method = 'batch'

    # get user input
    opts,args = getopt.getopt(sys.argv[1:], '-m:-a:-i:', ['method=', 'alpha=', 'iternum='])
    for opt_name, opt_value in opts:
        if opt_name in ('-m', '--method'):
            method = str(opt_value)
        if opt_name in ('-a', '--alpha'):
            alpha = float(opt_value)
        if opt_name in ('-i', '--iternum'):
            iter_num = int(opt_value)   


    # read in data
    train_set = pd.read_csv('../Toydata/mnist_27_training.csv', index_col = 0)
    test_set = pd.read_csv('../Toydata/mnist_27_testing.csv', index_col = 0)
    
    train_set = auxfuns.coding_label_bi(train_set, lab_1 = 7)
    test_set = auxfuns.coding_label_bi(test_set, lab_1 = 7)

    
    # gradient ascent
    if method == 'batch':
        weigths = gradDescent(train_set, a = alpha, iter_num = iter_num)
    # stochastic gradient ascent 1
    elif method == 'sgd1':
        weigths = StocGD_1(train_set, a = alpha, iter_num = iter_num)
    # stochastic gradient ascent 2
    else:
        weigths = StocGD_2(train_set, a = alpha, iter_num = iter_num)
       
    # prediction
    pred = []
    for i in range(test_set.shape[0]):
        pred = pred + [LR_classifier(test_set.iloc[i, 1:], weigths)]
    
    # error rate
    er = auxfuns.cat_error_rate(test_set.y, pred)
    print('Error Rate: ' + str(er) + ' (' + method + ')')
    
    if method == 'sgd2':
        print('You could run multiple times and calculate an average.')
