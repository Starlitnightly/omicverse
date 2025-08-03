#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: bio_task.py
@time: 2024/3/12 11:33
"""
from ..utils.log_manager import LogManager
import torch
from ..loader.scgpt import Scgpt
from ..loader.mamba import Scmamba
from ..loader.scbert import Scbert
from ..loader.scfoundation import Scfoundation
from ..loader.geneformer import Geneformer
from ..loader.cellplm import CellPLM
import scanpy as sc
from ..utils.utils import load_config
import numpy as np
import wandb
import os
import anndata as ad
from ..utils.preprocess import preprocess_adata
from ..repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
import sys
import importlib


class BioTask(object):
    """
    The BioTask class provides a standardized framework for executing analysis tasks on single-cell data.
    It handles model loading, data processing, and device configuration, enabling seamless integration
    of different pre-trained loader for various analytical tasks.

    Attributes:
        cfs_file (str): Path to the configuration file specifying task parameters and model choices.
        args (Namespace): Parsed arguments from the configuration file.
        device (torch.device): Device configuration, set based on args.
        gene2ids (dict): Mapping of genes to identifiers, initialized as None.
        load_obj (object): Model loader object, initialized based on model choice in args.
        model (torch.nn.Module): Loaded model based on the model type in args.
        vocab (dict): Vocabulary for gene identifiers, loaded from model loader if available.
        is_master (bool): Flag to check if the process is the main process for distributed training.
        wandb (wandb.Run or None): Weights & Biases tracking object, initialized if tracking is enabled.

    Methods:
        __init__(self, cfs_file, data_path=None, load_model=True):
            Initializes BioTask, loads configuration, device, and optionally the model.

        load_model(self):
            Loads and returns the pre-trained model based on the specified model type in args.

        read_h5ad(self, h5ad_file=None, preprocess=True, filter_gene=False):
            Reads and preprocesses single-cell data from an h5ad file, with optional gene filtering.

        filter_genes(self, adata):
            Filters genes in the AnnData object based on the vocabulary, logging the match rate.

        run(self):
            Placeholder for the main task execution method, to be implemented in subclasses.
    """
    def __init__(self, cfs_file, data_path=None, load_model=True, labels_num=0):
        """
        Initializes the BioTask instance with configuration, device settings, and model loading.

        Args:
            cfs_file (str): Path to the configuration file.
            data_path (str, optional): Path to the input data file, overrides default if provided.
            load_model (bool): Flag to indicate whether the model should be loaded on initialization.

        Raises:
            Exception: If configuration is missing required attributes.
        """
        self.cfs_file = cfs_file
        self.args = load_config(cfs_file)
        self.labels_num = labels_num

        self.logger = LogManager().logger
        if self.args.device == 'cpu' or self.args.device.startswith('cuda'):
            self.device = torch.device(self.args.device)
        else:
            self.device = torch.device('cuda:' + self.args.device)
        self.gene2ids = None
        self.load_obj = None
        self.data_handler = None
        if data_path is not None:
            self.args.input_file = data_path
        if load_model:
            self.model = self.load_model()
            self.data_handler = self.load_obj.data_handler
        self.vocab = self.load_vocab()
        self.is_master = int(os.environ['RANK']) == 0 if 'RANK' in os.environ else True
        if 'weight_bias_track' in self.args and self.args.weight_bias_track and self.is_master:

            wandb.init(project=self.args.project_name, name=self.args.exp_name, config=self.args)
            self.wandb = wandb
        else:
            self.wandb = None
        if 'output_dir' in self.args:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir, exist_ok=True)


        # if self.model is not None:
        #     self.model = self.model.to(self.device)

    def load_vocab(self):
        """
        Loads the vocabulary used for gene tokenization.

        Returns:
            GeneVocab: Vocabulary object with gene-to-index mappings.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        return vocab

    def get_gene2idx(self):
        return self.vocab.get_stoi()

    def get_id2gene(self):
        return self.vocab.get_itos()

    def load_model(self):
        """
        Loads the specified foundational model based on configuration.

        Returns:
            torch.nn.Module: The loaded model instance, or None if model type is unsupported.

        Raises:
            ValueError: If model type in configuration is unsupported.
        """
        if self.args.model_used == 'scgpt':
            self.load_obj = Scgpt(self.args)
            return self.load_obj.model
        elif self.args.model_used == 'scmamba':
            self.load_obj = Scmamba(self.args)
            return self.load_obj.model
        elif self.args.model_used == 'scbert':
            self.load_obj = Scbert(self.args)
            return self.load_obj.model
        elif self.args.model_used == 'scfoundation':
            self.load_obj = Scfoundation(args=self.args)
            return self.load_obj.model
        elif self.args.model_used == 'geneformer':
            self.load_obj = Geneformer(self.args, self.labels_num)
            return self.load_obj.model
        elif self.args.model_used == 'cellplm':
            self.load_obj = CellPLM(args=self.args)
            return self.load_obj.model
        else:
            raise ValueError(f'{self.args.model_uses} is out of range!')

    def run(self):
        """
        Placeholder method to execute the specific analysis task. Should be implemented by subclasses.

        Raises:
            NotImplementedError: Indicates that this method should be overridden by subclasses.
        """
        raise NotImplementedError("Not implemented")

    def llm_embedding(self, emb_type, adata=None, gene_ids=None):
        return self.load_obj.get_embedding(emb_type, adata=adata, gene_ids=gene_ids)

