#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: cellplm_handler.py
@time: 2025/4/16 9:55
"""
from biollm.data_preprocess.data_handler import DataHandler
from biollm.repo.CellPLM.utils.data import TranscriptomicDataset
import scanpy as sc
import warnings


class CellplmHandler(DataHandler):
    def __init__(self, vocab_path):
        """
        Initializes the ScbertHandler with the given H5AD file and vocabulary.

        Args:
            adata (str): the AnnData obj.
            vocab_path (str): Path to the vocabulary file.
        """
        super().__init__(vocab_path)

    def make_dataset(self, adata, obs_key, order_required):
        dataset = TranscriptomicDataset(adata, label_fields=obs_key, order_required=order_required)
        return dataset
