#!/usr/bin/env python3
# coding: utf-8
"""
@file: embedding.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/12  create file.
"""
from torch import nn
from typing import Optional
import numpy as np
import torch


class GeneEmbedding(nn.Module):
    def __init__(self, gene_num: int, embedding_dim: int, pad_index: Optional[int] = None):
        super().__init__()
        self.emb = nn.Embedding(gene_num, embedding_dim, pad_index)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm(x)
        return x


class Gene2VecEmbedding(nn.Module):
    def __init__(self, gene2vec_file):
        super().__init__()
        gene2vec_weight = np.load(gene2vec_file)
        # gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        # t = torch.arange(x.shape[1], device=x.device)
        return self.emb(x)


class OrganEmbedding(nn.Module):
    def __init__(self, organ_num: int, embedding_dim: int, pad_index: Optional[int] = None):
        super().__init__()
        self.emb = nn.Embedding(organ_num, embedding_dim, pad_index)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm(x)
        return x


class ExpressBinEmbedding(nn.Module):
    def __init__(self, exp_bin_num: int, embedding_dim: int, pad_index: Optional[int] = None):
        super().__init__()
        self.emb = nn.Embedding(exp_bin_num, embedding_dim, pad_index)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm(x)
        return x


class SequenceEmbedding(nn.Module):
    def __init__(self, sequence_num: int, embedding_dim: int, pad_index: Optional[int] = None):
        super().__init__()
        self.emb = nn.Embedding(sequence_num, embedding_dim, pad_index)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.norm(x)
        return x


class ExpressValueEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_value: int = 30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, embedding_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.max_value = max_value

    def forward(self, x):

        # x: [batch_size, seq_len], expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)
