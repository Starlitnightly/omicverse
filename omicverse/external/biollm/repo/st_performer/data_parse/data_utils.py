#!/usr/bin/env python3
# coding: utf-8
"""
@file: data_utils.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/11/21  create file.
"""
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import pandas as pd
import os


def qc(adata):
    adata = exp_to_array(adata)
    adata.obs['total_counts'] = cal_total_counts(adata.X)
    adata.var['total_counts'] = cal_total_cells_counts(adata.X) # 基因在所有细胞中表达的总count数
    adata.obs['n_genes_by_counts'] = cal_n_genes_by_counts(adata.X)
    adata.var['n_cells_by_counts'] = cal_n_cells_by_counts(adata.X)  # 基因在多少个细胞中表达
    return adata


def calculate_sparsity(data):
    total_elements = np.prod(data.shape)
    non_zero_elements = data.nnz
    sparsity = 1 - (non_zero_elements / total_elements)
    sparsity_dict = {"sparcity": sparsity}
    return sparsity_dict


def cal_total_counts(exp_array):
    total_count = np.array(exp_array.sum(1)).reshape(-1)
    return total_count


def cal_n_cells_by_counts(exp_array):
    n_cells = np.count_nonzero(exp_array, axis=0)
    return n_cells


def cal_n_genes_by_counts(exp_array):
    n_genes_by_counts = np.count_nonzero(exp_array, axis=1)
    return n_genes_by_counts


def cal_total_cells_counts(exp_array):
    n_cells_by_counts = np.array(exp_array.sum(0)).reshape(-1)
    return n_cells_by_counts


def filter_cells(adata, min_counts=200, n_genes_by_counts=3):
    index = adata.obs['total_counts'] > min_counts
    if n_genes_by_counts:
        index = index | (adata.obs['n_genes_by_counts'] > n_genes_by_counts)
    adata.X = adata.X[index, :]
    adata.obs = adata.obs[index]
    return adata


def filter_genes(adata, n_cells_by_counts=3):
    index = adata.var['n_cells_by_counts'] > n_cells_by_counts
#     if n_cells_by_counts:
#         index = index | (adata.var['n_cells_by_counts'] > n_cells_by_counts)
    adata.X = adata.X[:, index]
    adata.var = adata.var[index]
    return adata


def normalize(adata):
    if np.min(adata.X) < 0:
        print('scale data')
        return None
    elif np.max(adata.X) < 20:
        print('log1p data')
        return adata
    else:
        if is_raw_count(adata):
            print('raw data')
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
    return adata


def exp_to_array(adata):
    if issparse(adata.X):
        adata.X = adata.X.toarray()
    return adata


def is_raw_count(adata):
    if issparse(adata.X):
        flag = np.max(adata.X[0:10, :].toarray() - np.int32(adata.X[0:10, :].toarray())) == np.int32(0)
    else:
        flag = np.max(adata.X[0:10, :] - np.int32(adata.X[0:10, :])) == np.int32(0)
    return flag


### panglaodb
def cal_data_distribute(path):
    adata = sc.read_h5ad(path)
    adata.uns = calculate_sparsity(adata.X)
    adata = exp_to_array(adata)
    adata = qc(adata)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata.uns['cell_avg_total_counts'] = adata.obs['total_counts'].mean()
    adata.uns['cell_avg_n_genes_by_counts'] = adata.obs['n_genes_by_counts'].mean()
    adata.uns['gene_avg_total_counts'] = adata.var['total_counts'].mean()
    adata.uns['gene_avg_n_cells_by_counts'] = adata.var['n_cells_by_counts'].mean()
    adata.uns['pct_avg_counts_mt'] = adata.obs['pct_counts_mt'].mean()
    adata.uns['total_cells'] = adata.obs.shape[0]
#     sc.pp.filter_cells(adata, min_counts=1000)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.uns['total_cells_pass'] = adata.obs.shape[0]
    return adata


def stat_panglao_data(indir, meta_path, outdir):
    df = pd.read_csv(meta_path, header=None)
    result = []
    for i in os.listdir(indir)[0:3]:
        try:
            human_path = os.path.join(indir, i)
            adata = cal_data_distribute(human_path)
            sra = os.path.basename(human_path).split('.')[0].split('_')[0]
            srs = None if len(os.path.basename(human_path).split('.')[0].split('_')) ==1 else os.path.basename(human_path).split('.')[0].split('_')[-1]
            flag = df[0] == sra
            if srs:
                flag = flag & (df[1] == srs)
            meta = df[flag]
            adata.uns['organ'] = meta[2].values[0]
            adata.uns['platform'] = meta[3].values[0]
            adata.uns['species'] = meta[4].values[0]
            adata.uns['sequencer'] = meta[5].values[0]
            adata.uns['dataset'] = os.path.basename(human_path)
            adata.uns['path'] = os.path.join(outdir, i)
            adata.uns['source_dataset_id'] = sra if not srs else f'{sra}_{srs}'
            adata.uns['source'] = 'panglaodb'
            adata.write(adata.uns['path'])
            result.append(adata.uns)
            print('end: ', i)
        except Exception as e:
            print('error: ', i)
            print(e)
    stat_df = pd.DataFrame(result)
    stat_df.to_csv(os.path.join(outdir, 'dataset_stat.csv'))
    return stat_df


if __name__ == '__main__':
    indir = '/home/share/huada/home/qiuping1/workspace/llm/data/panglao/human/'
    outdir = '/home/share/huada/home/qiuping1/workspace/llm/data/panglao/human_qc'
    meta_path = '/home/share/huada/home/qiuping1/workspace/llm/data/panglao/metadata/metadata.txt'
    stat_panglao_data(indir, meta_path, outdir)
