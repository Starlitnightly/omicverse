import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.gcn import sparse_or_dense_dropout
from ..utils import to_sparse_tensor

__all__ = [
    'ImprovedGCN',
    'ImpGraphConvolution',
]


class ImpGraphConvolution(nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_own = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_nbr = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_own, gain=2.0)
        nn.init.xavier_uniform_(self.weight_nbr, gain=2.0)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight_nbr) + x @ self.weight_own + self.bias


class ImprovedGCN(nn.Module):
    """An improved GCN architecture.

    This version uses two weight matrices for self-propagation and aggregation,
    doesn't use batchnorm, and uses Tanh instead of ReLU nonlinearities.
    Has more stable training / faster convergence than standard GCN for overlapping
    community detection.

    This improved architecture was inspired by https://arxiv.org/abs/1906.12192
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, layer_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([ImpGraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(ImpGraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if layer_norm:
            self.layer_norm = [
                nn.LayerNorm([dim], elementwise_affine=False) for dim in hidden_dims
            ]
        else:
            self.layer_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(0)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = torch.tanh(x)
                if self.layer_norm is not None:
                    x = self.layer_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
