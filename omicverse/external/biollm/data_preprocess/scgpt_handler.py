#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: scgpt_handler.py
@time: 2024/3/3 15:02
"""
from .data_handler import DataHandler
from ..dataset.sc_dataset import ScgptDataset
import numpy as np
from scipy import sparse
from ..repo.scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch, random_mask_value
import torch


class ScgptHandler(DataHandler):
    def __init__(self, vocab_path):
        """
        Initializes the ScgptHandler with the given H5AD file and vocabulary.

        Args:
            adata (str): the AnnData obj.
            vocab_path (str): Path to the vocabulary file.
        """
        super().__init__(vocab_path)
    
    def binning(self, adata, bin_num):
        self.logger.info("Binning data ...")
        if not isinstance(bin_num, int):
            raise ValueError(
                "Binning arg must be an integer, but got {}.".format(bin_num)
            )
        n_bins = bin_num  # NOTE: the first bin is always a spectial for zero
        binned_rows = []
        bin_edges = []
        layer_data = adata.X
        layer_data = layer_data.A if sparse.issparse(layer_data) else layer_data
        if layer_data.min() < 0:
            raise ValueError(
                f"Assuming non-negative data, but got min value {layer_data.min()}."
            )
        for row in layer_data:
            if row.max() == 0:
                self.logger.warning(
                    "The input data contains all zero rows. Please make sure "
                    "this is expected. You can use the `filter_cell_by_counts` "
                    "arg to filter out all zero rows."
                )
                binned_rows.append(np.zeros_like(row, dtype=np.int64))
                bin_edges.append(np.array([0] * n_bins))
                continue
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            non_zero_digits = self._digitize(non_zero_row, bins)
            assert non_zero_digits.min() >= 1
            assert non_zero_digits.max() <= n_bins - 1
            binned_row = np.zeros_like(row, dtype=np.int64)
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
            bin_edges.append(np.concatenate([[0], bins]))
        adata.layers['X_binned'] = np.stack(binned_rows)
        adata.obsm["bin_edges"] = np.stack(bin_edges)
        return adata

    def preprocess(self, adata, var_key, obs_key, obs_id_output, n_hvg, bin_num, normalize_total=1e4):
        adata = super().preprocess(adata, var_key, obs_key, obs_id_output, n_hvg, normalize_total)
        adata = self.binning(adata, bin_num)
        return adata

    @staticmethod
    def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.
        side (:class:`str`, optional):
            The side to use for digitization. If "one", the left side is used. If
            "both", the left and right side are used. Default to "one".

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        if side == "one":
            return left_digits

        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    @staticmethod
    def tokenize_and_pad(adata, vocab, args, sort_seq_batch=False, obs_id_key=None, batch_id_key=None):
        all_counts = adata.layers['X_binned'].A if sparse.issparse(adata.layers['X_binned']) \
            else adata.layers['X_binned']
        genes = list(adata.var_names)
        if not batch_id_key:
            adata.obs["batch_id"] = 0
        else:
            adata.obs["batch_id"] = adata.obs[batch_id_key]
        if not obs_id_key:
            adata.obs["celltype_id"] = 0
        else:
            adata.obs["celltype_id"] = adata.obs[obs_id_key]
        train_celltype_labels = np.array(adata.obs["celltype_id"].tolist())  # make sure count from 0
        train_batch_labels = np.array(adata.obs["batch_id"].tolist())
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)
        tokenized_train = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
        )
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=args.mask_ratio,
            mask_value=args.mask_value,
            pad_value=args.pad_value,
        )
        input_gene_ids_train = tokenized_train["genes"]
        input_values_train = masked_values_train
        target_values_train = tokenized_train["values"]
        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
            train_sort_ids = np.argsort(train_batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
            tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]
        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
            "celltype_labels": tensor_celltype_labels_train,
        }
        return train_data_pt

    def make_dataset(self, adata, vocab, args, sort_seq_batch, obs_id_key, batch_id_key):
        train_data_pt = self.tokenize_and_pad(adata, vocab, args, sort_seq_batch, obs_id_key, batch_id_key)
        dataset = ScgptDataset(train_data_pt)
        return dataset
