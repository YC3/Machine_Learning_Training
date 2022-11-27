import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys, getopt
from collections import Counter, OrderedDict
sys.path.append('../')
import auxfuns 


'''
Author: YC3

Example:
python3 kNN.py -k 8 -f False

If want to see a scatter plot of training data:
python3 kNN.py -k 8 -f True
''' 


def knn_est_label(x, data, lab, k):

    # calculate distance
    sqDiff = np.power((np.tile(x, (data.shape[0], 1)) - data), 2)
    sumSqDiff = sqDiff.sum(axis = 1)
    sqDiff = sumSqDiff.apply(np.sqrt)

    # sorting
    idx = pd.Index(sqDiff).sort_values(ascending = True, return_indexer = True)[1]

    # top k labels
    labTopK = np.asarray(lab)[idx[0:k]]
    classes = np.unique(labTopK)

    # voting
    return OrderedDict(Counter(labTopK)).popitem(last = False)[0]




if __name__ == '__main__':

    # get user input
    opts,args = getopt.getopt(sys.argv[1:], '-k:-f:', ['k=', 'figure='])
    for opt_name, opt_value in opts:
        if opt_name in ('-k', '--k'):
            k_in = int(opt_value)
        if opt_name in ('-f', '--figure'):
            if_plot = eval(opt_value)

    # read in data
    train_set = pd.read_csv('../Toydata/mnist_27_training.csv', index_col = 0)
    test_set = pd.read_csv('../Toydata/mnist_27_testing.csv', index_col = 0)

    # scale to [0, 1]
    train_set = pd.concat([train_set.y, auxfuns.scaling01(train_set.iloc[:, 1:])], axis = 1)
    test_set = pd.concat([test_set.y, auxfuns.scaling01(test_set.iloc[:, 1:])], axis = 1)

    # plot out
    if if_plot:
        colors = ['blue', 'gold']
        c2 = plt.scatter(train_set.x_1[train_set.y == 2], train_set.x_2[train_set.y == 2], marker = '.', color = colors[0], alpha = 0.5, s = 400)
        c7 = plt.scatter(train_set.x_1[train_set.y == 7], train_set.x_2[train_set.y == 7], marker = '.', color = colors[1], alpha = 0.5, s = 400)
        plt.legend((c2, c7), ('2', '7'), scatterpoints = 1, loc = 'center left', ncol = 1, fontsize = 10, bbox_to_anchor = (1, 0.5))
        plt.show()

    # predict labels in testing set
    pred = np.array([])
    for i in range(test_set.shape[0]):
        pred_lab = knn_est_label(x = test_set.iloc[i, 1:], data = train_set.iloc[:, 1:], lab = train_set.iloc[:, 0], k = k_in)
        pred = np.append(pred, pred_lab)

    # error rate
    er = auxfuns.cat_error_rate(test_set.y, pred)
    print('Error Rate: ' + str(er) + ' (k = ' + str(k_in) + ')')
