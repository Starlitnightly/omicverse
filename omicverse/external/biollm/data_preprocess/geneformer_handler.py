#!/usr/bin/env python3
# coding: utf-8
"""
@author: Qianqian Chen  chenqianqian@genomics.cn
@last modified by: Qianqian Chen
@file: geneformer_handler.py
@time: 2025/4/8 11:12
"""
from biollm.data_preprocess.data_handler import DataHandler
from biollm.dataset.sc_dataset import GeneformerDataset
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
import numpy as np
from biollm.repo.geneformer.tokenizer import TranscriptomeTokenizer

class GeneformerHandler(DataHandler):
    def __init__(self, vocab_path, gene_median_file):
        """
        Initializes the GeneformerHandler with the given H5AD file and vocabulary.

        Args:
            adata (str): the AnnData obj.
            vocab_path (str): Path to the vocabulary file.
        """
        super().__init__(vocab_path)
        self.vocab_file = vocab_path
        self.gene_median_file = gene_median_file

    def load_data(self, adata=None, data_path=None, cell_type_key=None, nproc=16, add_length=True):
        """
        Loads and tokenizes single-cell data, preparing it for embedding extraction.

        Args:
            adata (AnnData, optional): Annotated data object for single-cell data.
            data_path (str, optional): Path to data file if adata is not provided.
            cell_type_key (str, optional): Key for cell type annotation.
            nproc (int, optional): Number of processes for tokenization. Default is 16.
            add_length (bool, optional): Whether to add sequence length information. Default is True.

        Returns:
            Dataset: Tokenized dataset for model input.
        """
        if data_path is not None and adata is None:
            adata = sc.read_h5ad(data_path)
        # if adata.raw is not None:
        #     adata.X = adata.raw.X
        # if adata.X.max() - np.int32(adata.X.max()) != 0:
        #     raise ValueError('Anndata.X must be raw count!')
        if 'n_counts' not in adata.obs.columns:
            if not issparse(adata.X):
                express_x = csr_matrix(adata.X)
            else:
                express_x = adata.X
            adata.obs["n_counts"] = np.ravel(express_x.sum(axis=1))
        if cell_type_key is not None:
            attr_dict = {cell_type_key: "cell_type"}
        else:
            attr_dict = None

        tk = TranscriptomeTokenizer(custom_attr_name_dict=attr_dict,
                                    gene_median_file=self.gene_median_file,
                                    token_dictionary_file=self.vocab_file,
                                    nproc=nproc)

        tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)

        tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata, add_length=add_length)

        return tokenized_dataset


    def make_dataset(self, adata, data_path, cell_type_key, nproc, add_length):
        tokenized_dataset = self.load_data(adata, data_path, cell_type_key, nproc, add_length)
        # dataset = GeneformerDataset(tokenized_dataset)
        return tokenized_dataset