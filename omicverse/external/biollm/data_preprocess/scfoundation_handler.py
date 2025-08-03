#!/usr/bin/env python3
# coding: utf-8
"""
@author: Qianqian Chen  chenqianqian@genomics.cn
@last modified by: Qianqian Chen
@file: geneformer_handler.py
@time: 2025/4/8 16:20
"""
from .data_handler import DataHandler
from ..dataset.sc_dataset import ScfoundationDataset
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from ..repo.scfoundation.get_embedding import main_gene_selection
import torch


class ScfoundationHandler(DataHandler):
    def __init__(self, vocab_path):
        """
        Initializes the ScfoundationHandler with the given H5AD file and vocabulary.

        Args:
            adata (str): the AnnData obj.
            vocab_path (str): Path to the vocabulary file.
        """
        super().__init__(vocab_path)


    def load_data(self, adata=None, data_path=None, max_none_zore=None):
        """
        Loads gene expression data and applies sparse selection based on non-zero thresholding.

        Args:
            adata (AnnData, optional): Annotated data object containing gene expression data.
            max_none_zore (int, optional): Threshold for non-zero values in gene expression data.

        Returns:
            pd.DataFrame: Processed gene expression data.
        """
        if adata is None:
            adata = sc.read_h5ad(data_path)
        print(adata)
        idx = adata.obs_names.tolist()
        col = adata.var_names.tolist()
        if issparse(adata.X):
            gexpr_feature = adata.X.toarray()
        else:
            gexpr_feature = np.array(adata.X)
        if max_none_zore:
            none_zero = gexpr_feature > 0
            none_zero_num = none_zero.sum(1)
            index = np.argwhere(none_zero_num > max_none_zore).reshape(-1)
            for i in index:
                none_zero_index = np.argwhere(none_zero[i]).reshape(-1)
                np.random.shuffle(none_zero_index)
                mask_num = none_zero_num[i] - max_none_zore
                mask_index = none_zero_index[0: mask_num]
                gexpr_feature[i][mask_index] = 0
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
        self.logger.info('covert gene feature into 19264')
        gene_list = list(self.vocab.get_stoi().keys())
        gexpr_feature = gexpr_feature.loc[:, gexpr_feature.columns.isin(gene_list)]
        gexpr_feature, to_fill_columns, var = main_gene_selection(gexpr_feature, gene_list)
        assert gexpr_feature.shape[1] == 19264
        return gexpr_feature

    def make_encoder_input(self, gexpr_feature, tgthighres):
        """
        Constructs encoder input features by combining gene expression data and target resolution.

        Args:
            gexpr_feature (pd.DataFrame): Gene expression feature data.
            tgthighres (str): Target resolution information.

        Returns:
            np.ndarray: Prepared input features for the encoder.
        """
        x = gexpr_feature.values
        totalcount = x.sum(axis=1).reshape(-1, 1)
        if tgthighres[0] == 'f':
            pretrain_gene_x = np.concatenate([x, np.log10(totalcount * float(tgthighres[1:])), np.log10(totalcount)], axis=1)
        elif tgthighres[0] == 'a':
            pretrain_gene_x = np.concatenate([x, np.log10(totalcount + float(tgthighres[1:])), np.log10(totalcount)], axis=1)
        elif tgthighres[0] == 't':
            pretrain_gene_x = np.concatenate([x, np.full_like(totalcount, np.float32(tgthighres[1:])), np.log10(totalcount)], axis=1)
        else:
            raise ValueError('tgthighres must be start with f, a or t')

        return pretrain_gene_x

    def prepare_data(self, adata, label_dict, label_key=None, max_none_zore=None):

        array_train = self.load_data(adata, max_none_zore=max_none_zore)
        if label_key:
            label_train = [label_dict.get(key, len(label_dict)) for key in adata.obs[label_key]]
            dataset_train = {"x": array_train, "targets": label_train}
        else:
            dataset_train = {"x": array_train}
        return dataset_train

    def make_dataset(self, adata, label_dict, finetune=False, label_key=None, for_train=False):
        if finetune and for_train:
            dataset = self.prepare_data(adata, label_dict, label_key)
            dataset = ScfoundationDataset(torch.tensor(np.array(dataset["x"])),
                                          torch.tensor(dataset["targets"]).long())
        else:
            dataset = self.prepare_data(adata, label_dict)
            dataset = ScfoundationDataset(torch.tensor(np.array(dataset["x"])))
        return dataset