#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: data_handler.py
@time: 2025/3/25 14:37
"""
import scanpy as sc
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import numpy as np
from abc import ABC, abstractmethod
from ..utils.log_manager import LogManager
import json
from ..repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from sklearn.model_selection import train_test_split


class DataHandler(ABC):

    def __init__(self, vocab_path):
        self.vocab_file = vocab_path
        self.logger = LogManager().logger
        self.vocab = self.load_vocab()
        self.gene2idx = self.get_gene2idx()
        self.id2gene = self.get_idx2gene()

    def load_vocab(self):
        vocab = GeneVocab.from_file(self.vocab_file)
        return vocab

    def get_gene2idx(self):
        """
        Retrieves gene-to-index mapping from vocabulary.

        Returns:
            dict: Dictionary mapping gene names to indices.
        """
        return self.vocab.get_stoi()

    def get_idx2gene(self):
        """
        Retrieves a mapping from index to gene IDs.

        This method returns a list of gene IDs based on the vocabulary.

        Returns:
            List[str]: A list of gene IDs corresponding to the model's vocabulary.
        """
        return self.vocab.get_itos()

    def check_adata(self, adata, var_key=None, obs_key=None, obs_id_output=None):
        """
        Reads an H5AD file and processes the AnnData object.

        If the AnnData object contains a raw matrix, it replaces `adata.X` with `adata.raw.X`.
        Additionally, it maps categorical labels in `obs_key` to numerical IDs if specified.

        Args:
            adata: AnnData obj
            obs_key (str, optional): The key in `adata.obs` containing categorical labels to be mapped to IDs.
            obs_id_output (str, optional): The output filename to save the mapping of labels to IDs.

        Returns:
            AnnData: Processed AnnData object with raw matrix set (if available) and mapped label IDs (if `obs_key` is provided).

        """
        if var_key:
            adata.var_names = adata.var[var_key].values
        if adata.raw is not None:
            adata.X = adata.raw.X
            self.logger.info("Use the raw X...")
        if obs_key:
            label2id = self.obs_label2id(adata, obs_key, obs_id_output)
            adata.obs[f'{obs_key}_id'] = adata.obs[obs_key].map(label2id)
        return adata

    def obs_label2id(self, adata, obs_key, output_path=None):
        """
        Maps categorical labels in `adata.obs[obs_key]` to numerical IDs and optionally saves the mapping.

        If the `obs_key` is provided, it maps the unique labels in the corresponding column of `adata.obs`
        to numerical IDs starting from 0. The mapping is then returned as a dictionary. If an `output_path`
        is specified, the mapping is also saved to a JSON file.

        Args:
            adata (AnnData): The AnnData object containing the observation metadata (`adata.obs`).
            obs_key (str): The key in `adata.obs` containing categorical labels to be mapped to IDs.
            output_path (str, optional): The output filename to save the mapping of labels to IDs. If not provided, the mapping is not saved.

        Returns:
            dict: A dictionary mapping each unique label in `adata.obs[obs_key]` to a numerical ID.
        """
        labels = np.unique(adata.obs[obs_key])
        label2id = dict(zip(labels, range(len(labels))))
        if output_path:
            with open(output_path, 'w') as fd:
                json.dump(label2id, fd)
        return label2id

    def filter_genes(self, adata):
        """
        Filters genes in the AnnData object based on the vocabulary attribute.

        Args:
            adata (AnnData): Annotated single-cell data matrix.

        Returns:
            AnnData: Filtered AnnData object with genes matching the vocabulary.

        Raises:
            Exception: If vocabulary is not set.
        """
        if self.vocab is None:
            raise Exception("No vocabulary, please set vocabulary first")
        if not adata.var.index.isin(list(self.gene2idx.keys())).any():
            print('Automatically converting gene symbols to ensembl ids...')
            adata.var.index = self.symbol_to_ensembl(adata.var.index.tolist())
            if (adata.var.index == '0').all():
                raise ValueError(
                    'None of AnnData.var.index found in pre-trained gene set.')
            adata.var_names_make_unique()
        adata.var['is_in_vocab'] = [1 if gene in self.vocab else 0 for gene in adata.var_names]
        self.logger.info(f'match {np.sum(adata.var["is_in_vocab"])}/{adata.var.shape[0]} genes in vocab of size {len(self.vocab)}')
        adata = adata[:, adata.var["is_in_vocab"] > 0].copy()
        return adata

    @staticmethod
    def symbol_to_ensembl(gene_list):
        import mygene
        mg = mygene.MyGeneInfo()
        return mg.querymany(gene_list, scopes='symbol', fields='ensembl.gene', as_dataframe=True,
                            species='human').reset_index().drop_duplicates(subset='query')['ensembl.gene'].fillna(
            '0').tolist()

    def normalize_data(self, adata, n_hvg=0, batch_key=None, normalize_total=1e4):
        """
        Normalizes and processes the data in an AnnData object by performing total normalization, log transformation,
        and selection of highly variable genes (HVGs).

        The method first checks whether a log transformation (`log1p`) is necessary, depending on the maximum value
        in the dataset. If the maximum value is less than 20, it disables the log transformation. It also checks if
        the total normalization should be performed based on the characteristics of the data matrix.

        If `normalize_total` is enabled, the total count for each cell is scaled to the specified target sum.
        If `log1p` is enabled, a log-transformation is applied to the data. Lastly, if `n_hvg` is specified, the top
        `n_hvg` highly variable genes are selected and retained.

        Args:
            adata (AnnData): The AnnData object containing the expression data to be normalized.
            n_hvg (int, optional): The number of highly variable genes to retain. If set to 0, no genes are selected.
            normalize_total (float, optional): The target sum for total normalization. Default is 1e4. If total normalization
                is not performed, it is ignored.

        Returns:
            AnnData: The normalized and processed AnnData object with log-transformed data (if applicable),
            total normalization (if applicable), and the selection of highly variable genes (if `n_hvg` is provided).
        """
        log1p = True
        if adata.X.max() < 20:
            log1p = False
            normalize_total = False
        elif adata.X.max() - np.int32(adata.X.max()) != np.int32(0):
            normalize_total = False
        else:
            pass
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=normalize_total)
        if log1p:
            sc.pp.log1p(adata)
        if n_hvg:
            if batch_key is None:
                self.logger.warning("No batch_key is provided, will use all cells for HVG selection.")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True, batch_key=batch_key)
        return adata

    def preprocess(self, adata, var_key, obs_key, obs_id_output, n_hvg, normalize_total=1e4):
        """
        Reads, filters, and normalizes a single-cell transcriptomic dataset, returning the processed AnnData object.

        This method first reads an H5AD file containing the single-cell data using `read_h5ad`, followed by filtering
        genes using `filter_genes`, and then normalizing the data using `normalize_data`. The processed AnnData object
        is returned after these steps.

        Args:
            var_key (str): The key in the gene metadata to be used for further processing.
            obs_key (str): The key in the celltype metadata to be used for further processing.
            obs_id_output (str): The output filename to save the mapping of labels to IDs.
            n_hvg (int): The number of highly variable genes to retain during normalization.
            normalize_total (float, optional): The target sum for total normalization. Default is 1e4.

        Returns:
            AnnData: The processed AnnData object, which has been filtered and normalized, ready for downstream analysis.
        """
        adata = self.check_adata(adata, var_key, obs_key, obs_id_output)
        adata = self.filter_genes(adata)
        adata = self.normalize_data(adata, n_hvg, normalize_total)
        return adata

    @abstractmethod
    def make_dataset(self, *args, **kwargs):
        """
        Converts an AnnData object into a PyTorch Dataset.

        This method should be implemented by subclasses to transform the input
        AnnData object (`adata`) into a PyTorch Dataset, which can then be used
        for training or evaluation in PyTorch-based workflows.

        Args:
            adata (AnnData): The AnnData object containing single-cell data to be transformed into a dataset.
            **kwargs: Additional arguments that may be passed to specific dataset implementations.

        Returns:
            torch.utils.data.Dataset: A PyTorch Dataset containing the processed data from `adata`.
        """
        pass

    @staticmethod
    def make_dataloader(dataset, batch_size, ddp_train, shuffle, drop_last, num_workers=4):
        """
        Creates a PyTorch DataLoader for training or evaluation.

        This method constructs a DataLoader for the given dataset, with support for distributed data parallel (DDP) training.
        If DDP training is enabled (`ddp_train=True`), the sampler is set to `DistributedSampler` for distributed training.
        Otherwise, a `SequentialSampler` or `RandomSampler` is used based on whether shuffling is enabled. The DataLoader
        is configured with the specified batch size, shuffle option, number of workers, and other parameters.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to be loaded by the DataLoader.
            batch_size (int): The number of samples per batch.
            ddp_train (bool): A flag indicating whether to use DDP training. If `True`, `DistributedSampler` is used.
            shuffle (bool): Whether to shuffle the dataset. If `True`, data is shuffled before each epoch.
            drop_last (bool): Whether to drop the last batch if it is smaller than the specified batch size.
            num_workers (int, optional): The number of subprocesses to use for data loading. Default is 4.

        Returns:
            torch.utils.data.DataLoader: The configured DataLoader for the given dataset.
        """
        if ddp_train:
            sampler = DistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset) if not shuffle else RandomSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
        )
        return data_loader

    @staticmethod
    def split_adata(adata, train_ratio=0.8, random_state=42):
        """
        Randomly splits an AnnData object into training and testing sets.

        This method splits the given AnnData object into two subsets: a training set and a testing set.
        The proportion of the data allocated to the training set is controlled by `train_ratio`, and the splitting
        process can be controlled with a random seed for reproducibility.

        Args:
            adata (AnnData): The AnnData object to be split into training and testing sets.
            train_ratio (float, optional): The proportion of the data to be used for the training set. Default is 0.8.
            random_state (int, optional): The random seed for reproducibility of the split. Default is 42.

        Returns:
            tuple: A tuple containing two AnnData objects:
                - `adata_train` (AnnData): The training set.
                - `adata_test` (AnnData): The testing set.
        """
        indices = np.arange(adata.n_obs)
        train_idx, test_idx = train_test_split(indices, train_size=train_ratio, random_state=random_state)
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()
        return adata_train, adata_test
