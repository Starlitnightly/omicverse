import torch
import torch.nn as nn
from ..utils import create_norm
import torch.nn.functional as F

class ResMLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num):
        super().__init__()
        self.layers = nn.ModuleList()
        assert num_layers > 1, 'At least two layer for MLPs.'
        for i in range(num_layers - 1):
            dim = hidden_dim if i>0 else in_dim
            self.layers.append(nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                create_norm(norm, hidden_dim)
            ))
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers - 1), out_dim),
            nn.PReLU(),
        )
        self.batch_emb = nn.Embedding(batch_num, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_dict):
        hist = []
        batch_labels = x_dict['batch']
        x = x_dict['h']
        for layer in self.layers:
            x = layer(x)
            x = x + self.layer_norm(self.batch_emb(batch_labels))
            hist.append(x)
        return {'recon': self.out_layer(torch.cat(hist, 1)), 'latent': x_dict['h']}

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num=0, dataset_num=0, platform_num=0, out_act=nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        # assert num_layers > 1, 'At least two layer for MLPs.'
        covariate_num = batch_num + dataset_num + platform_num
        for i in range(num_layers - 1):
            dim = hidden_dim if i > 0 else in_dim
            self.layers.append(nn.Sequential(
                nn.Linear(dim + covariate_num, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                create_norm(norm, hidden_dim),
            ))

        self.out_layer = [nn.Linear(hidden_dim, out_dim)]
        if out_act is not None:
            self.out_layer.append(out_act)
        self.out_layer = nn.Sequential(*self.out_layer)
        # self.batch_emb = nn.Embedding(covariate_num, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.batch_num = batch_num
        self.dataset_num = dataset_num
        self.platform_num = platform_num


    def reset_batch_emb(self):
        self.batch_emb.reset_parameters()
    
    def forward(self, x_dict):
        covariates = []
        if self.batch_num > 0:
            covariates.append(F.one_hot(x_dict['batch'], num_classes=self.batch_num))
        if self.dataset_num > 0:
            covariates.append(F.one_hot(x_dict['dataset'], num_classes=self.dataset_num))
        if self.platform_num > 0:
            covariates.append(F.one_hot(x_dict['platform'], num_classes=self.platform_num))
        x = x_dict['h']
        # x = x + self.batch_emb(batch_labels)#self.layer_norm(self.batch_emb(batch_labels))
        for i, layer in enumerate(self.layers):
            x = torch.cat([x]+covariates, 1)
            x = layer(x)
            # if i == 0:
            #     x += self.lib_emb(x_dict['lib_size'])
        return {'recon': self.out_layer(x), 'latent': x_dict['h']}