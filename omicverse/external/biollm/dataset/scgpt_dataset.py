#!/usr/bin/env python3
# coding: utf-8
"""
@file: scgpt_dataset.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/05/10  create file.
"""
from scipy.sparse import issparse
import numpy as np
from sklearn.model_selection import train_test_split
from biollm.repo.scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch, random_mask_value
from typing import Dict,Tuple
import os
import torch
from torch.utils.data import DataLoader, Dataset
from biollm.repo.scgpt import SubsetsBatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def make_dataset(adata, vocab, args, do_split=True):
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
    if 'batch_id' not in adata.obs_keys():
        adata.obs["batch_id"] = 0
    if 'celltype_id' not in adata.obs_keys():
        adata.obs["celltype_id"] = 0
    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)
    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    if do_split:
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )
        tokenized_train = tokenize_and_pad_batch(
            train_data,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
        )
        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            append_cls=True,
            include_zero_gene=args.include_zero_gene,
        )
        return tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels
    else:
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
        return tokenized_train, celltypes_labels, batch_ids


def make_train_data(adata, vocab, args, sort_seq_batch=False):
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
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


def prepare_data(tokenized_train, train_celltype_labels, train_batch_labels,  args, sort_seq_batch=False):
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


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 2,
    per_seq_batch_sample: bool = False,
    ddp_train: bool = False
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
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

