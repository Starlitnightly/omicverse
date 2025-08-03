import torch.nn as nn
import torch
import numpy as np
import random
import os
import logging

def set_seed(rndseed, cuda: bool = True, extreme_mode: bool = False):
    os.environ["PYTHONHASHSEED"] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # dgl.seed(rndseed)
    # dgl.random.seed(rndseed)
    logging.info(f"Setting global random seed to {rndseed}")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class DSBNNorm(nn.Module):
    def __init__(self, dim: int, domain_num: int, domain_label: str = 'dataset', eps: float = 1e-6, flip_rate=0.3):
        super().__init__()
        self.eps = eps
        self.domain_label = domain_label
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(domain_num+1)])
        self.flip_rate = flip_rate

    def forward(self, xdict):
        h = xdict['h']
        if self.training and random.random()<self.flip_rate:
            for i in xdict[self.domain_label].unique():
                h[xdict[self.domain_label]==i] = self.bns[i.item()+1](h[xdict[self.domain_label]==i])
        else:
            h = self.bns[0](h)
        return h

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name, n, h=4):
    if name == "layernorm":
        return nn.LayerNorm(n)
    elif name == "batchnorm":
        return nn.BatchNorm1d(n)
    elif name == "groupnorm":
        return nn.GroupNorm(h, n)
    elif name == 'rmsnorm':
        return RMSNorm(n)
    else:
        return nn.Identity()