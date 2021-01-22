import torch
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale


def dim_reduction(INPUT, n=50):
    test_emb = torch.load(INPUT)
    test_emb_p = test_emb['model_state_dict']['place_embedding.weight'][:1310]
    test_emb_cpu = test_emb_p.to(torch.device("cpu")).numpy()

    test_emb_w = test_emb['model_state_dict']['word_embedding.weight'][:1310]
    test_emb_w_cpu = test_emb_w.to(torch.device("cpu")).numpy()
    conc_emb = []
    for i in range(len(test_emb_cpu)):
        new_emb_i = np.concatenate((test_emb_cpu[i], test_emb_w_cpu[i]), axis = None)
        conc_emb.append(new_emb_i)
    conc_emb = np.array(conc_emb)
    conc_emb = np.nan_to_num(conc_emb)
    conc_emb = scale(conc_emb)
    print('Initial: ', conc_emb.shape)

    pca=PCA(n_components = n)
    pca.fit(conc_emb)
    conc_reduction=pca.transform(conc_emb)
    print('After PCA: ', conc_reduction.shape)
    return conc_reduction


def GAE_dim_reduction(INPUT, n=50):
    with open(INPUT, 'rb') as f:
        emb = pickle.load(f)
    conc_emb = emb
    conc_emb = np.nan_to_num(conc_emb)
    conc_emb = scale(conc_emb)
    print('Initial: ', conc_emb.shape)

    pca=PCA(n_components = n)
    pca.fit(conc_emb)
    conc_reduction=pca.transform(conc_emb)
    print('After PCA: ', conc_reduction.shape)
    return conc_reduction
    

def outputEmb(PATH, FIP_PATH, ZIP_PATH):
    with open(PATH, 'rb') as f:
        z_np = pickle.load(f)
    with open(FIP_PATH, 'rb') as f:
        fips_dic = pickle.load(f)
    fips = [int(fip) for fip in fips_dic.keys()]
    test_emb = z_np
    col_name = []
    for i in range(test_emb.shape[1]):
        col_name.append('emb_dimension_' + str(i))

    emb_col = []
    for i in range(test_emb.shape[1]):
        emb_col.append([])
    for i in range(len(col_name)):
        for emb in test_emb:
            emb_col[i].append(emb[i])

    dic = {'fips': fips}
    fips_emb = pd.DataFrame(data = dic)
    for i in range(len(col_name)):
        fips_emb[col_name[i]] = emb_col[i]
    zip_file = pd.read_csv(ZIP_PATH)
    zip_col = zip_file[['fips', 'zipcode']]
    output = pd.merge(fips_emb, zip_col, how='left', on=['fips'])
    return output
