#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: loader_base.py
@time: 2024/3/3 15:02
"""
from ..utils.log_manager import LogManager
from ..utils.utils import load_config
import torch
import wandb
import os
import re
from ..repo.scgpt.tokenizer.gene_tokenizer import GeneVocab


class LoaderBase(object):
    """
    The LoadLlm class provides the foundational structure for loading, initializing, and managing
    large language loader (LLMs) within the BioLLM framework. It supports model configuration loading,
    parameter initialization from pre-trained weights, and enables flexible integration with tracking
    platforms like Weights & Biases for logging and experiment management.

    Attributes:
        logger (Logger): Logger instance for logging information during model operations.
        args (Namespace): Parsed arguments loaded from configuration file or provided as input.
        model (torch.nn.Module or None): Model instance, initialized after loading pre-trained weights.
        vocab (dict or None): Vocabulary mapping for model inputs, set by derived classes if applicable.
        is_master (bool): Flag indicating if the process is the master process in distributed settings.
        wandb (wandb.Run or None): Weights & Biases tracking instance, initialized if tracking is enabled.
    """
    def __init__(self, args=None, cfs_file=None):
        """
        Initializes the LoadLlm class, setting up logging, configuration loading, and tracking if specified.

        Args:
            args (Namespace, optional): Model and task configuration parameters. If None, loads from file.
            cfs_file (str, optional): Path to the configuration file. Required if args is None.

        Raises:
            ValueError: If both args and cfs_file are None.
        """
        self.logger = LogManager().logger
        self.args = args if args is not None else load_config(cfs_file)
        self.model = None
        self.vocab = None
        self.is_master = int(os.environ['RANK']) == 0 if 'RANK' in os.environ else True
        if 'weight_bias_track' in self.args and self.args.weight_bias_track and self.is_master:
            wandb.init(project=self.args.project_name, name=self.args.exp_name, config=self.args)
            self.wandb = wandb
        else:
            self.wandb = None

    def load_pretrain_model(self, model_file, model, load_param_prefixs=None):
        """
        Loads pre-trained weights into the model, selectively loading parameters that match
        the current model's architecture.

        Args:
            model_file (str): Path to the file containing pre-trained model weights.
            model (torch.nn.Module, optional): Model instance to load weights into.
                If None, uses the class's model attribute.
            load_param_prefixs (list of str, optional): List of parameter prefixes to selectively load.

        Returns:
            torch.nn.Module: The model with updated weights from the pre-trained file.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        # only load params that are in the model and match the size
        model = model if model is not None else self.model
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location='cpu')
        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']
        pretrained_dict = {re.sub(r'module.', '', k): v for k, v in pretrained_dict.items()}
        if load_param_prefixs is not None:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in load_param_prefixs])
            }
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            self.logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def init_model(self, model=None):
        """
        Initializes the model with pre-trained weights if specified in the configuration.

        Args:
            model (torch.nn.Module, optional): The model instance to initialize. Defaults to the class's model.

        Returns:
            torch.nn.Module: The initialized model, with weights loaded if specified in the configuration.
        """
        load_param_prefixs = self.args.load_param_prefixs if 'load_param_prefixs' in self.args else None
        if 'model_file' in self.args:
            model = self.load_pretrain_model(self.args['model_file'], model, load_param_prefixs)
        return model

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Abstract method for retrieving model embeddings. Must be implemented in subclasses.

        Args:
            emb_type (str): Specifies the type of embedding to retrieve (e.g., cell or gene embedding).

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError('must implement get_embedding!')

    def freezon_model(self, keep_layers=[-2]):
        pass

    def get_idx2gene(self):
        """
        Retrieves a mapping from index to gene IDs.

        This method returns a list of gene IDs based on the vocabulary.

        Returns:
            List[str]: A list of gene IDs corresponding to the model's vocabulary.
        """
        return self.vocab.get_itos()

    def get_gene2idx(self):
        """
        Retrieves the gene-to-index mapping from the vocabulary.

        Returns:
            dict: Mapping of gene names to indices.
        """
        return self.vocab.get_stoi()

    def load_vocab(self):
        """
        Loads the vocabulary used for gene tokenization.

        Returns:
            GeneVocab: Vocabulary object with gene-to-index mappings.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        return vocab

    def load_model(self):
        raise NotImplementedError('must implement lod_model!')

    def get_gene_embedding(self, gene_ids):
        raise NotImplementedError('must implement get_gene_embedding!')

    def get_gene_expression_embedding(self, adata):
        raise NotImplementedError('must implement get_gene_expression_embedding!')

    def get_cell_embedding(self, adata, pool_type):
        raise NotImplementedError('must implement get_cell_embedding!')
