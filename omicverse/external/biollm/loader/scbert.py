#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: scbert.py
@time: 2024/3/11 15:02
"""
from biollm.loader.loader_base import LoaderBase
from biollm.repo.scbert.performer_pytorch.performer_pytorch import PerformerLM
from biollm.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from biollm.dataset.scbert_dataset import SCDataset, make_scbert_adata
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
from scipy.sparse import issparse
import pickle as pkl
import torch
from tqdm import tqdm
from biollm.data_preprocess.scbert_handler import ScbertHandler

class Scbert(LoaderBase):
    """
    LoadScbert is a class for loading and managing the SCBERT model.
    This class inherits from LoadLlm and provides functionalities such as loading pretrained loader,
    generating gene and cell embeddings, and preparing data for training.

    Attributes:
        num_tokens (int): Number of gene express bins.
        max_seq_len (int): Maximum sequence length for input data.
        use_g2v (bool): Indicator to use gene-to-vector (g2v) embeddings.
        g2v_file (str): Path to the file containing g2v embeddings.
        vocab (GeneVocab): Vocabulary object for gene-to-index mapping.
        model (PerformerLM): Loaded SCBERT model.
        gene2idx (dict): Mapping from gene names to indices.
    """
    def __init__(self, args):
        """
        Initializes LoadScbert with specific model settings and loads necessary components.

        Args:
            args (Namespace): Arguments object with settings for model, device, and file paths.
        """
        super(Scbert, self).__init__(args)
        self.num_tokens = args.n_bins
        self.max_seq_len = args.max_seq_len
        self.use_g2v = args.use_g2v
        self.g2v_file = args.g2v_file
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.gene2idx = self.get_gene2idx()
        self.init_model()
        self.model = self.model.to(self.args.device)
        self.data_handler = ScbertHandler(self.args.vocab_file)

    def load_model(self):
        """
        Loads the SCBERT model architecture with Performer-based attention.

        Returns:
            PerformerLM: The initialized SCBERT model.
        """
        model = PerformerLM(
            num_tokens=self.num_tokens,
            dim=200,
            depth=6,
            max_seq_len=self.max_seq_len,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=self.use_g2v,
            g2v_file=self.g2v_file
        )
        return model

    def get_gene_embedding(self, gene_ids):
        """
        Generates embeddings for a set of gene IDs.

        Args:
            gene_ids (torch.Tensor): Tensor of gene IDs to embed.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        emb = self.model.pos_emb.emb
        self.logger.info('start to get gene embedding!')
        gene_emb = emb(gene_ids)
        print('gpu used: ', torch.cuda.memory_allocated())
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        idx2genes = self.get_idx2gene()
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2genes[i] for i in gene_ids]
        return {'gene_emb': gene_emb.detach().cpu().numpy(), 'gene_names': gene_names}

    def freezon_model(self, keep_layers=[-2]):
        """
        Freezes the model layers except for specific normalization and selected layers,
        which can be fine-tuned.

        Args:
            keep_layers (list): List of layers to keep unfrozen for fine-tuning.
        """
        model_param_count = 0
        ft_param_count = 0
        for param in self.model.parameters():
            model_param_count += param.numel()
            param.requires_grad = False
        for param in self.model.norm.parameters():
            param.requires_grad = True
            ft_param_count += param.numel()
        for i in keep_layers:
            for name, param in self.model.performer.net.layers[i].named_parameters():
                param.requires_grad = True
                ft_param_count += param.numel()
        self.logger.info(f"Total pretrain-model Encoder Params {model_param_count}")
        self.logger.info(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")

    def get_dataloader(self,
                       adata,
                       var_key,
                       obs_key,
                       n_hvg,
                       bin_num,
                       batch_size,
                       ddp_train,
                       shuffle,
                       drop_last,
                       num_workers=1,
                       obs_id_output=None,
                       ):
        adata = self.data_handler.preprocess(adata, var_key, obs_key, obs_id_output, n_hvg)
        if obs_key:
            dataset = self.data_handler.make_dataset(adata, bin_num=bin_num, obs_id_key=f'{obs_key}_id')
        else:
            dataset = self.data_handler.make_dataset(adata, bin_num=bin_num, obs_id_key=None)
        dataloader = self.data_handler.make_dataloader(dataset, batch_size, ddp_train, shuffle, drop_last, num_workers)
        return dataloader

    def get_gene_expression_embedding(self, adata, pool='mean'):
        """
        Obtains gene expression embeddings by pooling the model’s cell-level embeddings.

        Args:
            adata (AnnData): Single-cell data.
            pool (str): Pooling method, either 'mean' or 'sum'.

        Returns:
            np.ndarray: Gene expression embeddings as a NumPy array.
        """
        self.logger.info("start to get gene expression!")
        data_loader = self.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=None, n_hvg=0,
                                          bin_num=self.args.n_bins, batch_size=self.args.batch_size, ddp_train=False,
                                          shuffle=False, drop_last=False)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            pool_emb = torch.zeros((len(self.gene2idx), self.args.embsize)).to(self.args.device)
            for index, data in enumerate(data_loader):
                data = data.to(self.args.device)
                cell_encode_x = self.model(data, return_encodings=True)  # [batch size, max_seq_len, dim]
                cell_encode_x = cell_encode_x[:, :-1, :]
                pool_emb += cell_encode_x.sum(dim=0)
                if pool == 'mean':
                    pool_emb = pool_emb / adata.shape[0]
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info("get gene expression Done!")
        id2gene = self.get_idx2gene()
        gene_names = [id2gene[i] for i in range(len(id2gene))]
        return {'gene_names': gene_names, 'gene_emb': pool_emb.detach().cpu().numpy()}

    def get_cell_embedding(self, adata, cell_emb_type='cls'):
        """
        Extracts cell embeddings using specified aggregation methods.

        Args:
            adata (AnnData): Single-cell data.
            cell_emb_type (str): Embedding aggregation type ('cls', 'sum', or 'mean').

        Returns:
            np.ndarray: Cell embeddings as a NumPy array.
        """
        self.logger.info("start to get cell embedding!")
        data_loader = self.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=None, n_hvg=0,
                                          bin_num=self.args.n_bins, batch_size=self.args.batch_size, ddp_train=False,
                                          shuffle=False, drop_last=False)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            cell_embeddings = []
            for index, data in tqdm(enumerate(data_loader), desc='get scbert cell embedding: '):
                data = data.to(self.args.device)
                cell_encode_x = self.model(data, return_encodings=True)  # [batch size, max_seq_len, dim]
                if cell_emb_type == 'cls':
                    cell_emb = cell_encode_x[:, -1, :]
                elif cell_emb_type == 'sum':
                    cell_emb = cell_encode_x[:, 0:-1, :].sum(axis=1)
                else:
                    cell_emb = cell_encode_x[:, 0:-1, :].mean(axis=1)
                cell_embeddings.append(cell_emb.detach().cpu().numpy())
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        self.logger.info("end to get cell embedding!")
        return cell_embeddings

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Obtains embeddings for either genes, cells, or gene-expression profiles.

        Args:
            emb_type (str): Type of embedding ('gene', 'cell', or 'gene-expression').
            adata (AnnData, optional): Single-cell data for cell or gene-expression embeddings.
            gene_ids (torch.Tensor, optional): Gene IDs for gene embeddings.

        Returns:
            np.ndarray: Embedding data based on the requested type.
        """
        self.model = self.model.eval()
        assert emb_type in ['gene', 'cell',
                            'gene-expression'], 'Invalid embedding type, must be gene, cell or gene-expression'
        if emb_type == 'gene' and gene_ids is None:
            raise ValueError('gene_ids must not be None if emb_type is gene!')
        if emb_type != 'gene' and adata is None:
            raise ValueError('adata must not be None if emb_type is cell or gene-expression!')
        if emb_type == 'gene':
            return self.get_gene_embedding(gene_ids)
        elif emb_type == 'cell':
            return self.get_cell_embedding(adata, cell_emb_type=self.args.cell_emb_type)
        else:
            return self.get_gene_expression_embedding(adata)

    def encoder(self, batch_data):
        """
        Encodes a batch of data using the SCBERT model.

        Args:
            batch_data (torch.Tensor): Batch of input data to encode.

        Returns:
            torch.Tensor: Encoded representations of the input data.
        """
        cell_encode_x = self.model(batch_data)
        return cell_encode_x
