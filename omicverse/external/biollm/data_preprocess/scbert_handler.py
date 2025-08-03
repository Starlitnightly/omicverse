#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: scbert_handler.py
@time: 2025/3/25 15:10
"""
from biollm.data_preprocess.data_handler import DataHandler
from biollm.dataset.sc_dataset import ScbertDataset
import numpy as np
from scipy import sparse
import anndata as ad


class ScbertHandler(DataHandler):
    """
    A data handler for processing single-cell transcriptomics data for use in ScBERT.

    This class is designed to handle data preprocessing and merging tasks for single-cell RNA-seq datasets,
    making it compatible with the ScBERT model. It includes methods to merge AnnData objects, process the
    gene expressions, and create datasets for model training.

    Args:
        h5ad_path (str): Path to the H5AD file containing the single-cell data.
        vocab_path (str): Path to the vocabulary file used for the model.

    Attributes:
        ref_genes (list): A list of reference gene symbols corresponding to the vocabulary.
        logger (logging.Logger): A logger instance to log progress and information.

    Methods:
        merge_adata(adata):
            Merges a given AnnData object with the reference genes to match the gene expression matrix.

        make_dataset(adata, bin_num, obs_key=None):
            Creates a ScbertDataset from the AnnData object, with optional label assignment.
    """

    def __init__(self, vocab_path):
        """
        Initializes the ScbertHandler with the given H5AD file and vocabulary.

        Args:
            adata (str):the AnnData obj.
            vocab_path (str): Path to the vocabulary file.
        """
        super().__init__(vocab_path)
        self.ref_genes = [self.id2gene[i] for i in range(len(self.id2gene))]

    def merge_adata(self, adata):
        """
        Merges a given AnnData object with reference genes to match the gene expression matrix.

        This method ensures that the input AnnData object contains gene names that match the reference genes.
        It constructs a new AnnData object with gene expression data aligned to the reference genes.

        Args:
            adata (AnnData): The input AnnData object containing gene expression data.

        Returns:
            AnnData: A new AnnData object with gene expression data aligned to the reference genes.

        Raises:
            ValueError: If no matching gene names are found between `adata.var_names` and reference genes.
        """
        new_data = np.zeros((adata.X.shape[0], len(self.ref_genes)))
        useful_gene_index = np.where(adata.var_names.isin(self.ref_genes))
        useful_gene = adata.var_names[useful_gene_index]
        if len(useful_gene) == 0:
            raise ValueError("No gene names in ref gene, please check that adata.var_names contain valid gene symbols!")

        self.logger.info('useful gene index: {}'.format(len(useful_gene)))
        use_index = [self.ref_genes.index(i) for i in useful_gene]
        if not sparse.issparse(adata.X):
            new_data[:, use_index] = adata.X[:, useful_gene_index[0]]
        else:
            new_data[:, use_index] = adata.X.toarray()[:, useful_gene_index[0]]
        new_data = sparse.csr_matrix(new_data)
        new_adata = ad.AnnData(X=new_data)
        new_adata.var_names = self.ref_genes
        new_adata.obs = adata.obs
        self.logger.info('end to make scbert adata for model.')
        return new_adata

    def make_dataset(self, adata, bin_num, obs_id_key=None):
        """
        Creates a ScbertDataset from the given AnnData object.

        This method processes the gene expression data and optionally assigns labels based on the provided
        observation key. It returns a dataset compatible with the ScBERT model.

        Args:
            adata (AnnData): The AnnData object containing gene expression data.
            bin_num (int): The number of bins for binning the expression data.
            obs_id_key (str, optional): The key in `adata.obs` containing the labels. Defaults to None.

        Returns:
            ScbertDataset: A dataset object that is ready for model training or evaluation.
        """
        data = adata.X
        label = adata.obs[obs_id_key] if obs_id_key else None
        dataset = ScbertDataset(data, bin_num, label)
        return dataset

    def preprocess(self, adata, var_key, obs_key, obs_id_output, n_hvg, normalize_total=1e4):
        adata = self.check_adata(adata, var_key, obs_key, obs_id_output)
        adata = self.filter_genes(adata)
        adata = self.merge_adata(adata)
        adata = self.normalize_data(adata, n_hvg, normalize_total)
        return adata
