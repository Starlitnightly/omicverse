# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:dataloader.py
# @Software:PyCharm
# @Created Time:2024/2/20 11:23 AM
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union
import scanpy as sc
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import scvi
import lmdb
import dgl
import torch
import os.path as osp
import json
import anndata
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse
from tqdm import tqdm
import random
import copy
sys.path.insert(0, "../")
from scgpt import SubsetsBatchSampler
from scLLM_utils.dataset import SeqDataset


def Get_DataLoader(dataset,args,shuffle=False,drop_last=False,num_workers=0,**kwargs):
    if args.task=='Cell_annotation':
        return cell_dataloader(dataset=dataset,args=args,shuffle=shuffle,
                                          drop_last=drop_last,num_workers=num_workers,**kwargs)
    elif args.task=='Integration':
        return cell_dataloader(dataset=dataset,args=args,shuffle=shuffle,
                                          drop_last=drop_last,num_workers=num_workers,**kwargs)
    elif args.task=='Pretraining':
        return Pretraining_dataloader(dataset=dataset,args=args,shuffle=shuffle,
                                          drop_last=drop_last,num_workers=num_workers,**kwargs)
    elif args.task=='GRN_inference':
        return GRN_dataloader(dataset=dataset, args=args, shuffle=shuffle,
                                          drop_last=drop_last,num_workers=num_workers,**kwargs)
    elif args.task=='Peturnbation':
        return Peturnbation_dataloader(dataset=dataset, args=args,shuffle=shuffle,
                                          drop_last=drop_last,num_workers=num_workers, **kwargs)

def cell_dataloader(dataset,args,shuffle,drop_last,num_workers,**kwargs):
    intra_domain_shuffle=kwargs['intra_domain_shuffle']
    if args.per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = dataset["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        dataset=SeqDataset(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                args.batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            pin_memory=True,
        )
        return data_loader
    dataset = SeqDataset(dataset)
    if args.distributed:
        sampler=DistributedSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader

def Pretraining_dataloader(dataset,args,shuffle,drop_last,num_workers,**kwargs):
    if args.distributed:
        sampler=DistributedSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=kwargs['pin_memory'],
        sampler=sampler,
    )
    return data_loader

def GRN_dataloader(dataset,args,shuffle,drop_last,num_workers,**kwargs):
    pass

def Peturnbation_dataloader(dataset,args,shuffle,drop_last,num_workers,**kwargs):
    pass



