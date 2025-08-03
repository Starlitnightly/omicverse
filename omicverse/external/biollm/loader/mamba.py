#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: mamba.py
@time: 2024/3/3 11:13
"""
import pickle

import numpy as np
from typing import List
from biollm.loader.loader_base import LoaderBase
from biollm.repo.scmamba.scmamba.mambaLM import MambaModel
from biollm.repo.scmamba.scLLM_utils.dataset import prepare_cell_data, filter_gene, SeqDataset
from biollm.repo.scmamba.scgpt.preprocess import Preprocessor
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from biollm.repo.scmamba.scgpt.preprocess import binning
import dgl
import os
from biollm.repo.scmamba.scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch


class Scmamba(LoaderBase):
    def __init__(self, args):
        super(Scmamba, self).__init__(args)
        self.do_pert = self.args.do_pert if 'do_pert' in self.args else False
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.init_model()
        self.model = self.model.to(self.args.device)

    def load_model(self):
        ntokens = len(self.vocab)

        model = MambaModel(
            ntoken=ntokens,
            d_model=self.args.embsize,
            nlayers=self.args.nlayers,
            vocab=self.vocab,
            n_input_bins=self.args.n_bins,
            pad_value=self.args.pad_value,
            do_pert=self.do_pert
        )
        return model

    def freezon_model(self, keep_layers=[-2]):
        model_param_count = 0
        ft_param_count = 0
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                if param.requires_grad:
                     model_param_count += param.numel()
                param.requires_grad = False
        for i in keep_layers:
            for name, param in self.model.mamba_encoder.layers[i].named_parameters():
                param.requires_grad = True
                ft_param_count += param.numel()
        self.logger.info(f"Total pretrain-model Encoder Params {model_param_count}")
        self.logger.info(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")

    def load_vocab(self):
        vocab = GeneVocab.from_file(self.args.vocab_file)
        special_tokens = ['<pad>', '<cls>', '<eoc>']
        for token in special_tokens:
            if token not in vocab:
                vocab.append_token(token)
        return vocab

    def get_gene2idx(self):
        return self.vocab.get_stoi()

    def get_idx2gene(self):
        return self.vocab.get_itos()

    def get_gene_embedding(self, gene_ids):
        self.logger.info('start to get gene embedding!')
        gene_embeddings = self.model.encoder(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info('finished get gene embedding!')
        return gene_embeddings.detach().cpu().numpy()

    def load_dataset(self, adata, do_preprocess=True, split_data=False):
        if do_preprocess:
            adata = self.preprocess_adata(adata)
        train_dataset, val_dataset, _, _ = prepare_cell_data(adata=adata, adata_test=None, args=self.args,
                                                             vocab=self.vocab, split_data=split_data)
        train_dataset = SeqDataset(train_dataset)
        if val_dataset is not None:
            val_dataset = SeqDataset(val_dataset)
        return train_dataset, val_dataset

    def preprocess_adata(self, adata):
        if 'do_preprocess' in adata.uns:
            self.logger.info('the adata was already preprocessed, pass the step!')
            return adata
        adata, _ = filter_gene(vocab=self.vocab, adata=adata, is_master=False, logger=None)
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=True,  # step 1
            filter_cell_by_counts=True,  # step 2
            normalize_total=self.args.data_is_raw,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=self.args.data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=self.args.hvg_number,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if self.args.data_is_raw else "cell_ranger",
            binning=self.args.n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata)
        adata.uns['do_preprocess'] = True
        return adata

    def get_dataloader(self, dataset, drop_last=False, num_workers=0):
        if self.args.distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
        )
        return data_loader

    def encoder(self, batch_data):
        input_gene_ids = batch_data["gene_ids"].to(self.args.device)
        input_values = batch_data["values"].to(self.args.device)
        if self.args.graph_sort and self.args.layer_emb:
            sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.args.device)
        else:
            sorted_layer_idx = None
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
        embeddings = self.model._encode(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            sorted_layer_idx=sorted_layer_idx
        )  # batch_size * max_seq_len * dim
        return embeddings

    def get_cell_embedding(self, adata, do_preprocess=False):
        self.logger.info('start to get cell embedding!')
        dataset, _ = self.load_dataset(adata, do_preprocess=do_preprocess, split_data=False)
        self.logger.info('load dataset Done!')
        data_loader = self.get_dataloader(dataset)
        self.logger.info('get dataloader Done!')
        cell_embeddings = np.zeros((adata.shape[0], self.args.embsize), dtype=np.float32)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for batch_data in tqdm(data_loader, desc='Cell embedding'):
                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                if self.args.graph_sort and self.args.layer_emb:
                    sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.args.device)
                else:
                    sorted_layer_idx = None
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
                output = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    sorted_layer_idx=sorted_layer_idx
                )
                embeddings = output['cell_emb']  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count: count + len(embeddings)] = embeddings
                count += len(embeddings)
            # cell_embeddings = cell_embeddings / np.linalg.norm(
            #     cell_embeddings, axis=1, keepdims=True
            # )
            if self.wandb:
                total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
                self.wandb.log({'GPU': total_gpu})
            self.logger.info("get cell embedding Done!")
            return cell_embeddings

    def get_gene_expression_embedding(self, adata, pool='mean', do_preprocess=False):
        self.logger.info("start to get gene expression!")
        dataset, _ = self.load_dataset(adata, do_preprocess=do_preprocess, split_data=False)
        data_loader = self.get_dataloader(dataset)
        gene_ids = None
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            pool_emb = torch.zeros((adata.shape[1], self.args.embsize)).to(self.args.device)
            for batch_data in tqdm(data_loader, desc='Gene expression embedding'):
                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                if gene_ids is None:
                    gene_ids = input_gene_ids[0, :].detach().cpu().numpy()
                input_values = batch_data["values"].to(self.args.device)
                if self.args.graph_sort and self.args.layer_emb:
                    sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.args.device)
                else:
                    sorted_layer_idx = None
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
                embeddings = self.model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    sorted_layer_idx=sorted_layer_idx
                )  # batch_size * max_seq_len * dim
                pool_emb += embeddings.sum(dim=0)
            if pool == 'mean':
                pool_emb = pool_emb / adata.shape[0]
        self.logger.info("get gene expression Done!")
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        idx2gene = self.get_idx2gene()
        gene_names = [idx2gene[i] for i in gene_ids]
        return pool_emb.detach().cpu().numpy(), gene_names

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
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

    def make_pertdata(self, express_x: np.ndarray, gene_list: List,  return_pt: bool = True):
        self.logger.info("making perturbation data for mamba model.")
        if self.args.input_style == 'binned':
            express_x = binning(express_x, self.args.n_bins)
        gene2idx = self.get_gene2idx()
        gene_ids = np.array([gene2idx[gene] for gene in gene_list if gene in gene2idx])
        if self.args.graph_sort:
            graph = dgl.load_graphs(os.path.join(self.args.graph_path, 'kb_acyclic_reg_cxg.dgl'))[0][0]
        else:
            graph = None
        self.logger.info("start to tokenize data for mamba model")
        tokenized_train = tokenize_and_pad_batch(
            express_x,
            gene_ids,
            max_len=self.args.max_seq_len,
            vocab=self.vocab,
            pad_token=self.args.pad_token,
            pad_value=self.args.pad_value,
            append_cls=self.args.append_cls,  # append <cls> token at the beginning
            include_zero_gene=self.args.include_zero_gene,
            graph=graph,
            return_pt = return_pt
        )
        train_data_dict = {
            'gene_ids': tokenized_train['genes'],
            'values': tokenized_train['values'],
            'sorted_layer_idx': tokenized_train['sorted_layer_idx']
        }
        if self.args.graph_sort:
            train_data_dict['sorted_index'] = tokenized_train['sorted_index']
        self.logger.info("making perturbation data, Done.")
        return train_data_dict


if __name__ == '__main__':
    from biollm.utils.utils import load_config
    import scanpy as sc

    config_file = '../config/embeddings/scmamba_emb.toml'
    configs = load_config(config_file)
    adata = sc.read_h5ad(configs.input_file)
    obj = Scmamba(configs)
    adata = obj.preprocess_adata(adata)
    obj.args['max_seq_len'] = adata.shape[1]
    print(obj.args)
    if 'gene_name' not in adata.var:
        adata.var['gene_name'] = adata.var.index.values
    gene_ids = list(obj.get_gene2idx().values())
    gene_ids = np.array(gene_ids)
    gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(configs.device)
    emb, genenames = obj.get_embedding(obj.args.emb_type, adata, gene_ids)
    print('embedding shape:', emb.shape)
    print(len(genenames))
    with open(obj.args.output_dir + f'/scmamba_{obj.args.emb_type}_emb.pk', 'wb') as w:
        pickle.dump(emb, w)
