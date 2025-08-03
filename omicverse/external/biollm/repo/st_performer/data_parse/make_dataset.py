#!/usr/bin/env python3
# coding: utf-8
"""
@file: make_dataset.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/11/20  create file.
"""

from st_performer.tokenizer import GeneTokenizer
from st_performer.data_parse.data_utils import *
from utils import data2lmdb
import scanpy as sc
import json
from collections import OrderedDict


class MakeDataset:
    def __init__(self, h5ad_file, gene_vocab, lmdb_path, cell_dict, is_st=False, organ=None):
        self.h5ad_file = h5ad_file
        self.tokenizer = GeneTokenizer(gene_vocab)
        self.organ = organ
        self.is_st = is_st
        self.lmdb_path = lmdb_path
        self.cell_dict = cell_dict


    def parse_express(self, do_qc=False):
        adata = sc.read_h5ad(self.h5ad_file)
        if self.organ:
            index = adata.obs['root_organ'] == self.organ
            adata = adata[index, :]
        if is_raw_count(adata):
            adata = qc(adata)
        if do_qc:
            adata = filter_cells(adata)
            n_cells_by_counts = int(adata.shape[0] * 0.05)
            adata = filter_genes(adata, min_cells=3, n_cells_by_counts=n_cells_by_counts)
        adata = normalize(adata)
        sc.pp.highly_variable_genes(adata)
        return adata

    def make_finetune_dataset(self, lmdb_num, celltype_count):
        adata = sc.read_h5ad(self.h5ad_file)
        adata = adata[~adata.obs['ontology_name'].isna(), :]
        print('before filter: ', adata.X.shape)
        # sc.pp.filter_cells(adata, min_genes=200)
        # sc.pp.filter_genes(adata, min_cells=3)
        # adata = normalize(adata)
        # index = ~(adata.obs['ontology_name'] == 'notAvailable')
        # adata = adata[index, :]
        # print('after filter: ', adata.X.shape)
        ref_genes = [i for i in self.tokenizer.vocab]
        adata = self.merge_ref_genes(adata, ref_genes)
        _, _, sample_num = data2lmdb(adata, self.lmdb_path, lmdb_num, finetune=True, cell_label=self.cell_dict, celltype_count=celltype_count)
        print(self.cell_dict)
        print(celltype_count)
        return sample_num

    def merge_ref_genes(self, adata, ref_genes):
        adata.var_names = adata.var['Symbol'].map(lambda x: x.lower())
        index = np.isin(adata.var_names, ref_genes)
        adata = adata[:, index]
        print('ref genes: ', len(ref_genes), 'adata shape: ', adata.shape)
        adata.var['gene_id'] = adata.var_names.map(self.tokenizer.vocab)
        return adata


if __name__ == '__main__':
    import os
    from collections import defaultdict

    make_finetune = True
    if make_finetune:
        h5ad_dir = '/home/share/huada/home/qiuping1/workspace/llm/data/finetune/mouse_brain/sc'
        gene_vocab = './test/gene_vocab_inner.txt'
        lmdb = '/home/share/huada/home/qiuping1/workspace/llm/data/finetune/mouse_brain/sc/finetune.db'
        output = '/home/share/huada/home/qiuping1/workspace/llm/data/finetune/mouse_brain/sc/cell_dict.json'
        cell_dict = OrderedDict()
        celltype_count = defaultdict(int)
        sample_num = 0
        for f in os.listdir(h5ad_dir)[-2:]:
            f_path = os.path.join(h5ad_dir, f)
            if f.endswith('.h5ad'):
                print(f)
                make_data_obj = MakeDataset(f_path, gene_vocab, lmdb, cell_dict, False, 'brain')
                sample_num = make_data_obj.make_finetune_dataset(sample_num, celltype_count)
                print(sample_num)
        with open(output, 'w') as w:
            json.dump(cell_dict, w)
