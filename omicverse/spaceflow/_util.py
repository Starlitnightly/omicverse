# -*- coding: utf-8 -*-
import torch
import numpy as np

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index