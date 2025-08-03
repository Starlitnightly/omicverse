#!/usr/bin/env python3
# coding: utf-8
"""
@file: scbert_dataset.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/16  create file.
"""
from torch.utils.data import Dataset
import random
import torch
from scipy import sparse
import numpy as np
import anndata as ad
import scanpy as sc
from biollm.utils.log_manager import LogManager
from scipy.sparse import issparse
import pickle as pkl
from sklearn.model_selection import StratifiedShuffleSplit


logger = LogManager().logger


class SCDataset(Dataset):
    def __init__(self, data, bin_num, label=None):
        super().__init__()
        self.data = data
        self.label = label
        self.bin_num = bin_num

    def __getitem__(self, index):
        rand_start = index
        # rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0] if issparse(self.data) else self.data[rand_start]
        full_seq[full_seq > (self.bin_num - 2)] = self.bin_num - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        if self.label is not None:
            seq_label = self.label[rand_start]
        else:
            seq_label = None
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


def make_scbert_adata(adata, ref_genes):
    if adata.raw is not None:
        adata.X = adata.raw.X
        logger.info("Use the raw X...")
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    logger.info(adata)
    if np.min(adata.X) != 0:
        return None
    if 'gene_name' in adata.var_keys():
        adata.var_names = adata.var['gene_name'].values
    new_data = np.zeros((adata.X.shape[0], len(ref_genes)))
    useful_gene_index = np.where(adata.var_names.isin(ref_genes))
    useful_gene = adata.var_names[useful_gene_index]
    if len(useful_gene) == 0:
        raise ValueError("No gene names in ref gene, please check the adata.var_names are gene Symbol!")

    logger.info('useful gene index: {}'.format(len(useful_gene)))
    use_index = [ref_genes.index(i) for i in useful_gene]
    if not sparse.issparse(adata.X):
        new_data[:, use_index] = adata.X[:, useful_gene_index[0]]
    else:
        new_data[:, use_index] = adata.X.toarray()[:, useful_gene_index[0]]
    new_data = sparse.csr_matrix(new_data)
    new_adata = ad.AnnData(X=new_data)
    new_adata.var_names = ref_genes
    new_adata.obs = adata.obs
    logger.info('end to make scbert adata for model, start preprocess')
    if adata.X.min() >= 0:
        normalize_total = False
        log1p = False
        if adata.X.max() > 30:
            log1p = True
            if adata.X.max() - np.int32(adata.X.max()) == np.int32(0):
                normalize_total = 1e4
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=normalize_total)
            logger.info('do normalize total...')
        if log1p:
            sc.pp.log1p(adata)
            logger.info('do log1p...')
    else:
        raise Exception('the express matrix have been scale, exit!')
    logger.info('end to preprocess')
    return new_adata


def make_dataset(adata, output_dir, bin_num):
    label_dict, label = np.unique(np.array(adata.obs['celltype']),
                                  return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    # store the label dict and label for prediction
    with open(f'{output_dir}/label_dict', 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open(f'{output_dir}/label', 'wb') as fp:
        pkl.dump(label, fp)

    label = torch.from_numpy(label)
    data = adata.X
    # random split
    train_data, val_data, train_label, val_label = split_data(data, label, 0.8, 42)
    train_dataset = SCDataset(train_data, bin_num, train_label)
    val_dataset = SCDataset(val_data, bin_num, val_label)
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
    # for index_train, index_val in sss.split(data, label):
    #     data_train, label_train = data[index_train], label[index_train]
    #     data_val, label_val = data[index_val], label[index_val]
    #     train_dataset = SCDataset(data_train, bin_num, label_train)
    #     val_dataset = SCDataset(data_val, bin_num, label_val)
    return train_dataset, val_dataset


def split_data(data, label, train_ratio=0.8, random_seed=42):
    """
    将数据划分为训练集和验证集。

    Args:
        data (np.ndarray): 数据数组。
        label (np.ndarray): 标签数组。
        train_ratio (float): 训练集所占比例，默认为 0.8。
        random_seed (int, optional): 随机种子，保证划分可重复。

    Returns:
        tuple: (train_data, val_data, train_label, val_label)
    """
    assert len(data) == len(label), "Data and label must have the same length."

    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)

    # 随机打乱索引
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # 打乱数据和标签
    data = data[indices]
    label = label[indices]

    # 计算训练集大小
    train_size = int(len(data) * train_ratio)

    # 划分训练集和验证集
    train_data = data[:train_size]
    val_data = data[train_size:]
    train_label = label[:train_size]
    val_label = label[train_size:]

    return train_data, val_data, train_label, val_label
