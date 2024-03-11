from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pickle as pkl
import os
import h5py
import pandas as pd
import random
import pdb
import math
from random import randint, sample
from sklearn.model_selection import KFold


def load_data(dataset):
    print("Loading lncRNAdisease dataset")

    path_dataset = 'raw_data/' + dataset + '/interMatrix.npy'
    net = np.load(path_dataset)
    path_dataset = 'raw_data/' + dataset + '/disSim.npy'
    v_features = np.load(path_dataset)
    path_dataset = 'raw_data/' + dataset + '/miSim.npy'
    u_features = np.load(path_dataset)

    num_list = [len(u_features)]
    num_list.append(len(v_features))
    temp = np.zeros((net.shape[0], net.shape[1]), int)
    u_features = np.hstack((u_features, net))
    v_features = np.hstack((net.T, v_features))

    a = np.zeros((1, u_features.shape[0] + v_features.shape[0]), int)
    b = np.zeros((1, v_features.shape[0] + u_features.shape[0]), int)
    u_features = np.vstack((a, u_features))
    v_features = np.vstack((b, v_features))

    num_lncRNAs = net.shape[0]
    num_diseases = net.shape[1]

    row, col, _ = sp.find(net)
    perm = random.sample(range(len(row)), len(row))
    row, col = row[perm], col[perm]
    sample_pos = (row, col)
    print("the number of all positive sample:", len(sample_pos[0]))

    print("sampling negative links for train and test")
    sample_neg = ([], [])
    net_flag = np.zeros((net.shape[0], net.shape[1]))
    X = np.ones((num_lncRNAs, num_diseases))
    net_neg = X - net
    row_neg, col_neg, _ = sp.find(net_neg)
    perm_neg = random.sample(range(len(row_neg)), len(row))
    row_neg, col_neg = row_neg[perm_neg], col_neg[perm_neg]
    sample_neg = (row_neg, col_neg)
    sample_neg = list(sample_neg)
    print("the number of all negative sample:", len(sample_neg[0]))

    u_idx = np.hstack([sample_pos[0], sample_neg[0]])
    v_idx = np.hstack([sample_pos[1], sample_neg[1]])
    labels = np.hstack([[1] * len(sample_pos[0]), [0] * len(sample_neg[0])])

    l1 = np.zeros((1, net.shape[1]), int)
    print(l1.shape)
    net = np.vstack([l1, net])
    print("old net:", net.shape)
    l2 = np.zeros((net.shape[0], 1), int)
    net = np.hstack([l2, net])
    print("new net:", net.shape)

    u_idx = u_idx + 1
    v_idx = v_idx + 1
    
    return u_features, v_features, net, labels, u_idx, v_idx, num_list