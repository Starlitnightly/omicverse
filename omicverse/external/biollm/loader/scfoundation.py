#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :scfoundation.py
# @Time      :2024/4/15 17:41
# @Author    :Qianqian Chen

import torch
import pandas as pd
from .loader_base import LoaderBase
from scipy.sparse import issparse
from ..repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from ..repo.scfoundation.load import load_model_frommmf
from ..repo.scfoundation.get_embedding import main_gene_selection
import numpy as np
from ..repo.scfoundation.load import gatherData, getEncoerDecoderData
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from ..utils.preprocess import preprocess_adata
from ..data_preprocess.scfoundation_handler import ScfoundationHandler


class Scfoundation(LoaderBase):
    """
    LoadScfoundation class is a specialized loader for scFoundation model in the BioLLM framework.
    It initializes and manages a single-cell model, processes gene expression data, and generates gene and cell embeddings.

    Attributes:
    vocab : GeneVocab
        Gene vocabulary loaded from a specified file.
    model : torch.nn.Module
        The pretrained foundational model loaded from the specified file.
    config : dict
        Configuration dictionary for the model, including settings for the encoder and decoder.
    device : torch.device
        The computational device (CPU or GPU) where the model runs.
    gene2idx : dict
        Dictionary mapping gene symbols to indices.
    """
    def __init__(self, args=None, cfs_file=None):
        super(Scfoundation, self).__init__(args, cfs_file)
        self.vocab = self.load_vocab()
        self.model, self.config = self.load_model()
        self.device = torch.device(self.args.device)
        self.model.to(self.device)
        self.gene2idx = self.get_gene2idx()
        self.data_handler = ScfoundationHandler(self.args.vocab_file)

    def load_model(self):
        """
        Loads the foundational model and configuration from a specified file.

        Returns:
            tuple: A tuple containing the pretrained foundational model (torch.nn.Module)
        """
        model, config = load_model_frommmf(self.args.model_file, key=self.args.key)
        return model, config

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Retrieves the embedding for genes, cells, or gene expressions, depending on the specified type.

        Args:
            emb_type (str): Type of embedding to generate ('gene', 'cell', 'gene-expression').
            adata (AnnData, optional): Annotated data object required for 'cell' and 'gene-expression' embeddings.
            gene_ids (list of int, optional): Gene IDs required for 'gene' embedding.

        Returns:
            np.ndarray: The computed embeddings as a NumPy array.
        """

        if adata is not None:
            adata = preprocess_adata(adata, self.args.n_hvg if 'n_hvg' in self.args else False)
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
            return self.get_cell_embedding(adata)
        else:
            return self.get_gene_expression_embedding(adata)
        return emb

    def get_gene_embedding(self, gene_ids):
        """
        Computes embeddings for specified genes using the model's positional embedding layer.

        Args:
            gene_ids (list of int): IDs of genes to generate embeddings for.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        emb = self.model.pos_emb(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info('start to get gene embedding!')
        idx2gene = self.get_idx2gene()
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2gene[i] for i in gene_ids]
        return {'gene_emb': emb.detach().cpu().numpy(), 'gene_names': gene_names}

    def get_dataloader(self,
                       adata,
                       label_dict,
                       finetune,
                       label_key,
                       for_train,
                       batch_size,
                       ddp_train,
                       shuffle,
                       drop_last):
        dataset = self.data_handler.make_dataset(adata,label_dict, finetune=finetune, label_key=label_key, for_train=for_train)
        dataloader = self.data_handler.make_dataloader(dataset, batch_size, ddp_train, shuffle, drop_last)
        return dataloader


    def get_gene_expression_embedding(self, adata, pool='mean'):
        """
        Obtains gene expression embeddings by pooling the model’s cell-level embeddings.

        Args:
            adata (AnnData): Single-cell data.
            pool (str): Pooling method, either 'mean' or 'max'.

        Returns:
            np.ndarray: Gene expression embeddings as a NumPy array.
        """
        df = self.data_handler.load_data(adata, data_path=self.args.input_file, max_none_zore=self.args.max_none_zero_num)
        pretrain_gene_x = self.data_handler.make_encoder_input(df, self.args.tgthighres)
        self.model.to_final = None
        with torch.no_grad(), torch.cuda.amp.autocast():
            pool_emb = torch.zeros((len(self.gene2idx), 512)).to(self.args.device)
            for i in tqdm(range(0, pretrain_gene_x.shape[0], self.args.batch_size)):
                x = pretrain_gene_x[i: i+self.args.batch_size, :]
                x = torch.from_numpy(x).to(self.args.device)
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(
                x.float(), x.float(), self.config)
                out = self.model.forward(x=encoder_data, padding_label=encoder_data_padding,
                                    encoder_position_gene_ids=encoder_position_gene_ids,
                                    encoder_labels=encoder_labels,
                                    decoder_data=decoder_data,
                                    mask_gene_name=False,
                                    mask_labels=None,
                                    decoder_position_gene_ids=decoder_position_gene_ids,
                                    decoder_data_padding_labels=decoder_data_padding,
                                    )
                out = out[:, :19264, :].contiguous()
                pool_emb += out.sum(dim=0)
                if pool == 'mean':
                    pool_emb = pool_emb / adata.shape[0]
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info("get gene expression Done!")
        idx2gene = self.get_idx2gene()
        gene_names = [idx2gene[i] for i in range(19264)]
        return {'gene_names': gene_names, 'gene_emb': pool_emb.detach().cpu().numpy()}

    def get_cell_embedding(self, adata, pool='max'):
        """
        Obtains cell embeddings by processing the gene expression data through the model.

        Args:
            adata (AnnData): Single-cell data.
            pool (str): Pooling method, either 'all' or 'max'.

        Returns:
            np.ndarray: Cell embeddings as a NumPy array.
        """
        df = self.data_handler.load_data(adata, data_path=self.args.input_file, max_none_zore=self.args.max_none_zero_num)
        pretrain_gene_x = self.data_handler.make_encoder_input(df, self.args.tgthighres)
        cell_embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in tqdm(range(0, pretrain_gene_x.shape[0], self.args.batch_size)):
                x = pretrain_gene_x[i: i+self.args.batch_size, :]
                x = torch.from_numpy(x).to(self.args.device)
                value_labels = x > 0
                data_gene_ids = torch.arange(19266, device=x.device).repeat(x.shape[0], 1)
                x, x_padding = gatherData(x, value_labels, self.config['pad_token_id'])
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.config['pad_token_id'])
                x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                position_emb = self.model.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = self.model.encoder(x, x_padding)
                geneemb1 = geneemb[:, -1, :]
                geneemb2 = geneemb[:, -2, :]
                geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
                if pool == 'all':
                    geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
                elif pool == 'max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                cell_embeddings.append(geneembmerge.detach().cpu().numpy())
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        self.logger.info("end to get cell embedding!")
        return cell_embeddings


if __name__ == '__main__':
    from biollm.utils.utils import load_config
    import pickle as pkl
    import os
    import scanpy as sc

    config_file = '../../tutorials_bak/zero-shot/configs/scfoundation_cell_emb.toml'
    configs = load_config(config_file)

    obj = Scfoundation(configs)
    print(obj.args)

    adata = sc.read_h5ad(configs.input_file)
    adata = adata[:1000, :]
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    print('embedding shape:', emb.shape)
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir, exist_ok=True)
    with open(obj.args.output_dir + f'/scfoundation_{obj.args.emb_type}_emb.pk', 'wb') as w:
        res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
        pkl.dump(emb, w)
