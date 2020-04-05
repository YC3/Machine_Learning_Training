import numpy as np
import pandas as pd
import operator
import math
from numpy.random import random 
import matplotlib.pyplot as plt 
import sys, getopt

sys.path.append('../')
import auxfuns 


'''
Author: YC3

Parameters:
-t, --type: 'mean' to predict with average in leaf nodes, or 'reg' to fit a linear regression model
-e, --error: float, min error reduction to continue
-s, --size: int, min number of data points in each leaf node

Example:
python3 CART.py -t reg -e 0.5 -s 10

python3 CART.py -t mean -e 0.5 -s 10

''' 


class Node():
    
    def __init__(self, feature = None, value = None, left = None, right = None):
        
        self.f = feature
        self.v = value
        self.l = left
        self.r = right
       

 
def Split(data, feature, value):
    
    data_l = data.loc[data.iloc[:, feature] <= value, :]
    data_r = data.loc[data.iloc[:, feature] > value, :]
    return data_l, data_r



def leaf_Mean(data):
    
    return np.mean(data.iloc[:, -1])
 


def error_Squared(data):
    
    return np.var(data.iloc[:, -1])*data.shape[0]
   
 

def linear_Reg(data):
    
    n, m = data.shape
    y = np.mat(data.iloc[:, -1]).reshape(n, 1)
    X = np.mat(np.concatenate([np.ones((data.shape[0], 1)), data.iloc[:, 0:(m - 1)]], axis = 1))
    X2 = X.T*X

    if np.linalg.det(X2) == 0:
        raise NameError('Singular. Try a larger size.')
    w = X2.I*(X.T*y)
    return w, X, y
       
  
  
def leaf_Reg(data):
    
    w, X, y = linear_Reg(data)
    return w



def error_Reg(data):
    
    w, X, y = linear_Reg(data)
    return np.sum(np.power(y - X*w, 2))


    
def splitSelect(data, leafFun, errorFun, minErrorReduce, minLeafSize):
    
    n, m = data.shape
    totalError = errorFun(data)
    leafMean = leafFun(data)
        
    # consistency
    if len(set(data.iloc[:, -1])) == 1:
        return None, leafMean
    
    minError = np.inf
    sel_feature = 0
    sel_value = 0
    
    for i in range(m - 1):
        for v in set(data.iloc[:, i]):
            data_l, data_r = Split(data, i, v)
            
            # min leaf node size
            if data_l.shape[0] < minLeafSize or data_r.shape[0] < minLeafSize: 
                continue
                
            totalError2 = errorFun(data_l) + errorFun(data_r)
            if totalError2 < minError:
                minError = totalError2
                sel_feature = i
                sel_value = v
         
    # min error reduction
    if (totalError - minError) < minErrorReduce:
        return None, leafMean
    
    data_l, data_r = Split(data, sel_feature, sel_value)
    
    # min leaf node size
    if data_l.shape[0] < minLeafSize or data_r.shape[0] < minLeafSize: 
        return None, leafMean
            
    return sel_feature, sel_value
        
        

def CreateTree(data, leafFun, errorFun, minErrorReduce, minLeafSize):
    
    feature, value = splitSelect(data, leafFun, errorFun, minErrorReduce, minLeafSize)

    if feature == None: 
        return value
    
    tree = Node()
    tree.f = feature
    tree.v = value
    data_l, data_r = Split(data, feature, value)
    tree.l = CreateTree(data_l, leafFun, errorFun, minErrorReduce, minLeafSize)
    tree.r = CreateTree(data_r, leafFun, errorFun, minErrorReduce, minLeafSize)
    
    return tree
    
    

def CollapseTree(tree):
    if isinstance(tree.l, Node):
        tree.l = CollapseTree(tree.l)
    if isinstance(tree.r, Node): 
        tree.r = CollapseTree(tree.r)

    return (tree.l + tree.r)/2
    


def PruneTree(tree, data, leafFun):

    if data.shape[0] == 0:
        return CollapseTree(tree)
    
    if isinstance(tree.l, Node) or isinstance(tree.r, Node):
        data_l, data_r = Split(data, tree.f, tree.v)
        
    if leafFun == 'mean':
        if isinstance(tree.l, Node): tree.l = PruneTree(tree.l, data_l, 'mean')
        if isinstance(tree.r, Node): tree.r = PruneTree(tree.r, data_r, 'mean')
    if leafFun == 'reg':
        if isinstance(tree.l, Node): tree.l = PruneTree(tree.l, data_l, 'reg')
        if isinstance(tree.r, Node): tree.r = PruneTree(tree.r, data_r, 'reg')
    
    if not isinstance(tree.l, Node) and not isinstance(tree.r, Node): 
        data_l, data_r = Split(data, tree.f, tree.v)
        
        m = data.shape[1]
        
        if leafFun == 'mean':
            Error_merge = np.sum((data.iloc[:, -1] - (tree.l + tree.r)/2)**2)
            Error_sep = np.sum((data_l.iloc[:, -1] - tree.l)**2) + np.sum((data_r.iloc[:, -1] - tree.r)**2)
            
        if leafFun == 'reg':
            y1 = np.mat(data.iloc[:, -1]).T
            X1 = np.mat(np.concatenate([np.ones((data.shape[0], 1)), data.iloc[:, 0:(m - 1)]], axis = 1))
            Error_merge = np.sum(np.power(y1 - X1*(tree.l + tree.r)/2, 2))
            
            y2 = np.mat(data_l.iloc[:, -1]).T
            X2 = np.mat(np.concatenate([np.ones((data_l.shape[0], 1)), data_l.iloc[:, 0:(m - 1)]], axis = 1))
            y3 = np.mat(data_r.iloc[:, -1]).T
            X3 = np.mat(np.concatenate([np.ones((data_r.shape[0], 1)), data_r.iloc[:, 0:(m - 1)]], axis = 1))            
            Error_sep = np.sum(np.power(y2 - X2*tree.l, 2)) + np.sum(np.power(y3 - X3*tree.r, 2))
            
        # merge if can reduce error
        if Error_merge < Error_sep:
            print('pruned 1 branch')
            return (tree.l + tree.r)/2
        # no merge if can't
        else:
            return tree
    else:
        return tree  

    

def CART_predict(leaf, data, leafFun):
    
    if leafFun == 'mean':
        return float(leaf)
    if leafFun == 'reg':
        X = np.mat([1] + list(data))
        return float(X*leaf)
    
    

def retrieveTree(tree, data, leafFun):
    
    if not isinstance(tree, Node): return CART_predict(tree, data, leafFun)
        
    if data.iloc[tree.f] <= tree.v:
        if isinstance(tree.l, Node):
            return retrieveTree(tree.l, data, leafFun)
        else:
            return CART_predict(tree.l, data, leafFun)
    if data.iloc[tree.f] > tree.v:
        if isinstance(tree.r, Node):
            return retrieveTree(tree.r, data, leafFun)
        else:
            return CART_predict(tree.r, data, leafFun)
    
    

    
if __name__ == '__main__':
  
    # get user input
    opts,args = getopt.getopt(sys.argv[1:], '-t:-e:-s:', ['type=', 'error', 'size='])
    for opt_name, opt_value in opts:
        if opt_name in ('-t', '--type'):
            leafType = str(opt_value)
        if opt_name in ('-e', '--error'):
            err = float(opt_value)
        if opt_name in ('-s', '--size'):
            N = int(opt_value)
  

    # read in data
    data = pd.read_csv('../Toydata/temp_carbon.csv', index_col = 0).dropna(axis = 0, how = 'any')

    # split data
    train_set, test_set = auxfuns.trainTest_split(data, p_train = 0.8)

    # take average on leaf nodes
    if leafType == 'mean':
        tree_avg = CreateTree(data = train_set, leafFun = leaf_Mean, errorFun = error_Squared, minErrorReduce = err, minLeafSize = N)
        ptree_avg = PruneTree(tree = tree_avg, data = test_set, leafFun = 'mean')

    # linear regression on leaf nodes
    if leafType == 'reg':
        tree_reg = CreateTree(data = train_set, leafFun = leaf_Reg, errorFun = error_Reg, minErrorReduce = err, minLeafSize = N)
        ptree_reg = PruneTree(tree = tree_reg, data = test_set, leafFun = 'reg')

    # predict
    n = test_set.shape[0]
    y_hat = []

    for i in range(n):
        if leafType == 'mean':
            y_hat = y_hat + [retrieveTree(ptree_avg, test_set.iloc[i, 0:-1], leafType)]
        if leafType == 'reg':
            y_hat = y_hat + [retrieveTree(ptree_reg, test_set.iloc[i, 0:-1], leafType)]
        
    # correlation between y and y_hat
    pcc = np.corrcoef(np.array(y_hat), test_set.iloc[:, -1])[1, 0]

    print('PCC is: ', round(pcc, 3))
