# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

import scanpy as sc
import anndata as ad
# from utils import *
from datetime import datetime
from time import time
import torch.multiprocessing as mp
# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
# from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SEED = 2021

# Control sources of randomness
torch.manual_seed(SEED)
random.seed(SEED)

############################# Test lr on cross_organ dataset ##############################
data = sc.read_h5ad('./Data/human_15organ_subset_normed.h5ad')
methods = np.unique(data.obs['orig.ident'])
index_methods = data.obs['orig.ident']

label = data.obs.celltype
data = data.X

for val_i in range(len(methods)):
	print(methods[val_i])
	train_index = index_methods!=methods[val_i]
	val_index = index_methods==methods[val_i]
	X_train, y_train = data[train_index], label[train_index]
	X_test, y_test = data[val_index], label[val_index]
	cv_results = {}
	for c in [1e-3, 1e-2, 1e-1, 1]:
	    # print("c={}".format(c))
	    lr = LogisticRegression(random_state=0, penalty="l1", C=c, solver="liblinear")
	    res = cross_validate(lr, X_train, y_train, scoring=['accuracy'])
	    cv_results[c] = np.mean(res['test_accuracy'])
	# print(cv_results)
	#choose best c and calc performance on val_dataset
	best_ind = np.argmax(list(cv_results.values()))
	c = list(cv_results.keys())[best_ind]
	# print("best c={}".format(c))
	lr = LogisticRegression(random_state=0, penalty="l1", C=c, solver="liblinear")
	lr.fit(X_train, y_train)
	# print("train set accuracy: " + str(np.around(lr.score(X_train, y_train), 4)))
	print("test set accuracy: " + str(np.around(lr.score(X_test, y_test), 4)))
	val_macro_f1 = f1_score(y_test, lr.predict(X_test), average="macro")
	print("test set macro F1: " + str(np.around(val_macro_f1, 4)))

