import torch
import torch.nn as nn
from ..utils import create_norm

class ResMLPEncoder(nn.Module):
    def __init__(self, num_hidden, num_layers, dropout, norm, covariates_dim=0):
        super().__init__()
        self.layers = nn.ModuleList()
        assert num_layers > 1, 'At least two layer for MLPs.'
        for i in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.PReLU(),
                nn.Dropout(dropout),
                create_norm(norm, num_hidden)
            ))
        self.out_layer = nn.Sequential(
            nn.Linear(num_hidden * (num_layers - 1), num_hidden),
            nn.PReLU(),
            nn.Dropout(dropout),
            create_norm(norm, num_hidden)
        )

    def forward(self, x_dict):
        hist = []
        x = x_dict['h']
        for layer in self.layers:
            x = layer(x)
            hist.append(x)
        return self.out_layer(torch.cat(hist, 1))

class MLPEncoder(nn.Module):
    def __init__(self, num_hidden, num_layers, dropout, norm, covariates_dim=0):
        super().__init__()
        self.layers = nn.ModuleList()
        # assert num_layers > 1, 'At least two layer for MLPs.'
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.PReLU(),
                nn.Dropout(dropout),
                # nn.BatchNorm1d(num_hidden),
                create_norm(norm, num_hidden)
            ))

    def forward(self, x_dict):
        x = x_dict['h']
        for layer in self.layers:
            x = x + layer(x)
        return {'hidden': x}