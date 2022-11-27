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
This script implements the 'ID.3 algorithm', which can split datasets with nominal values.
Recursion is used here to build a decision tree.

Parameters:
-t, --tree: bool, print tree object or not
-s, --save: bool, save tree object or not

Example:
python3 DecisionTree_ID3.py -t True -s False

''' 


def entropy(lab):
    
    total = len(lab)
    
    # number of examples in each class
    classes = np.unique(lab)
    labCount = np.array([])
    
    for i in np.arange(len(classes)):
        labCount = np.append(labCount, sum([1 if x == classes[i] else 0 for x in lab]))
        
  # compute entropy
    h = 0.0
    for i in labCount:
        p = i/total
        h += p*abs(math.log(p, 2))
    
    return h


def branch_split(data, f_idx, value):
    
    data_fcol = data.iloc[:, f_idx]
    data_sel = data.loc[data_fcol == value, :]
    
    data_left = data_sel.iloc[:, list(np.arange(f_idx)) + list(np.arange(f_idx + 1, data.shape[1]))]
    
    return data_left


def split_select(data):
    
    f_num = data.shape[1]
  
    h0 = entropy(data.iloc[:,0])    # entropy before splitting
  
    infoGain_max = 0                # starting maximum info gain
    feat_sel = -1                   # selected feature

    # calculate info gain for splitting on every feature
    for f in range(1, f_num - 1):    

        uni_val = set(data.iloc[:, f])
    
        # info gain from every possible split split
        h_split = 0
        for v in uni_val:
            data_sub = branch_split(data, f, v)
            p = len(data_sub)/len(data)
            h_split += p*entropy(data_sub.iloc[:,0])  
        infoGain = h0 - h_split     
  
        # update the maximum info gain
        if infoGain > infoGain_max:
            infoGain_max = infoGain
            feat_sel = f

    return feat_sel
    


def class_vote(data):
    
    lab_v = set(data.iloc[:, 0])
    count_class = {}
    for i in lab_v:
        count_class[i] = sum(data.iloc[:, 0] == i)
    
    max_class = max([(value, key) for key, value in count_class.items()])[1]

    return max_class



def grow_tree(data):
    
    # stop when the items of a class have the same labels
    if len(set(data.iloc[:, 0])) == 1:  
        return data.iloc[0, 0]

    # stop when only 1 feature left
    if data.shape[1] == 2:       
        return class_vote(data)

    # choose best feature to split data
    fet_id = split_select(data)
    if fet_id != -1:    # cannot split: multiple features left but highly correlated

        fet_sel = data.columns[fet_id]
        
        tree = {fet_sel:{}}

        uni_val = set(data.iloc[:, fet_id])

        for v in uni_val:
            data_sub = branch_split(data, fet_id, v)
            tree[fet_sel][v] = grow_tree(data_sub)
    else:
        return class_vote(data)

    return tree


def tree_classifier_id3(tree, example):

    fet = list(tree.keys())[0]
    val = str(example[fet])

    tree1 = tree[fet]

    for i in tree1.keys():
        if val == i:
            if type(tree1[i]).__name__ == 'dict':
                lab = tree_classifier_id3(tree1[i], example)
            else:
                lab = tree1[i]   
    return lab
  

if __name__ == '__main__':
  
    # get user input
    opts,args = getopt.getopt(sys.argv[1:], '-t:-s:', ['tree=', 'save='])
    for opt_name, opt_value in opts:
        if opt_name in ('-t', '--tree'):
            if_print_tree = eval(opt_value)
        if opt_name in ('-s', '--save'):
            if_save_tree = eval(opt_value)
  

    # read in data
    data = pd.read_csv('../Toydata/olive.csv', index_col = 0)
    del(data['area'])
    

    # since ID.3 does not take numeric data, need to categorize numeric features
    for i in data.columns.values[1:]:
        data[i] = auxfuns.feature_cut(data[i], n_class = 4)
    print(data.head())
    

    # get the tree
    tree_o = grow_tree(data)

    if if_print_tree:
        print(tree_o)

    if if_save_tree:
        auxfuns.tree_save(tree_o, 'obj_tree')


    # classify new samples
    pred = np.array([])
    for i in range(data.shape[0]):
        pred = np.append(pred, tree_classifier_id3(tree_o, data.iloc[i, :]))

    print(pred[0:10])
    print(np.array(data.region)[0:10])

    # error rate
    er = round(auxfuns.cat_error_rate(np.array(data.region), np.array(pred)), 5)
    print('Error Rate: ' + str(er))
