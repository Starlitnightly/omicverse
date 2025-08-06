from torch import nn
from .autoencoders import VAELatentLayer, SplitLatentLayer, MergeLatentLayer, GMVAELatentLayer
from .adversarial import AdversarialLatentLayer
from .contrastive import ECSLatentLayer
import torch
import logging
from abc import ABC
from abc import abstractmethod
from ..utils import DSBNNorm

# class AbstractLatentLayer(ABC):s
#     def __init__(self):
#         self.is_adverserial = True
#
#     @abstractmethod
#     def forward(self, h, g):
#         pass

def create_latent_layer(**config) -> nn.Module:
    if config['type'] == 'adversarial':
        return AdversarialLatentLayer(**config)
    elif config['type'] == 'vae':
        return VAELatentLayer(**config)
    elif config['type'] == 'split':
        return SplitLatentLayer(**config)
    elif config['type'] == 'merge':
        return MergeLatentLayer(**config)
    elif config['type'] == 'gmvae':
        return GMVAELatentLayer(**config)
    elif config['type'] == 'vqvae':
        return VQVAELatentLayer(**config)
    elif config['type'] == 'ecs':
        return ECSLatentLayer(**config)
    else:
        raise ValueError(f"Unrecognized latent model name: {config['type']}")

class PlaceholderLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.is_adversarial = False

    def forward(self, x_dict):
        return x_dict['h'], torch.tensor(0.).to(x_dict['h'].device)

class LatentModel(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.layers = nn.ModuleList([PlaceholderLayer()])
        self.alias_dict = {}
        if configs is not None:
            for c in configs:
                self.layers.append(create_latent_layer(**c))

    def forward(self, x_dict):
        total_loss = 0
        for layer in self.layers:
            x_dict['h'], loss = layer(x_dict)
            total_loss += loss
        return x_dict['h'], total_loss

    def add_layer(self, **config):
        if 'alias' in config:
            self.alias_dict[config['alias']] = len(self.layers)
        else:
            self.alias_dict[config['type']] = len(self.layers)
        self.layers.append(create_latent_layer(**config))

    def get_layer(self, alias):
        return self.layers[self.alias_dict[alias]]

    def d_train(self, x_dict):
        loss = 0
        for layer in self.layers:
            if layer.is_adversarial:
                loss += layer.d_iter(x_dict)
        return loss

class PreLatentNorm(nn.Module):
    def __init__(self, type='none', enc_hid=None, dataset_num=None):
        super().__init__()
        self.type = type
        if type not in ['none', 'dsbn', 'ln']:
            raise NotImplementedError(f'"{type}" type of pre latent norm is not implemented.')
        if type == 'dsbn':
            self.norm = DSBNNorm(enc_hid, dataset_num)
        elif type == 'ln':
            self.norm = nn.LayerNorm(enc_hid)

    def forward(self, xdict):
        if self.type == 'dsbn':
            return self.norm(xdict)
        elif self.type == 'ln':
            return self.norm(xdict['h'])
        else:
            return xdict['h']
