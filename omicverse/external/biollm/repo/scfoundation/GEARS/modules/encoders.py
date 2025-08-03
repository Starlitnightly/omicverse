#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: encoders.py
@time: 2024/3/3 15:02
"""
import importlib
import logging
import os
import pickle
import re
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any
from .performer_module import PerformerModule
from .transformer import pytorchTransformerModule
from .mae_autobin import MaeAutobin

import torch

#from omegaconf import OmegaConf
from torch import nn, Tensor
import math
import sys 
from ...scFoundation.model.load import *


def next_16x(x):
    return int(math.ceil(x / 16) * 16)

def gatherData(data, labels, pad_token_id):
    """
    """
    value_nums = labels.sum(1)
    max_num = next_16x(max(value_nums))

    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def getEncoerDecoderData(data, data_raw, config):
    """
    """
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_data_labels,
                                                    config['pad_token_id'])
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels,
                                                config['pad_token_id'])
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids

def select_module(config, sub_config, module_name):
    if module_name == 'performer':
        return PerformerModule(
            max_seq_len=config['seq_len'],
            dim=sub_config['hidden_dim'],
            depth=sub_config['depth'],
            heads=sub_config['heads'],
            dim_head=sub_config['dim_head'],
            ff_dropout=sub_config.get('ff_dropout',0.0),
            attn_dropout=sub_config.get('attn_dropout',0.0)
        )
    elif module_name == 'transformer':
        return pytorchTransformerModule(
            max_seq_len=config['seq_len'],
            dim=sub_config['hidden_dim'],
            depth=sub_config['depth'],
            heads=sub_config['heads']
        )

    else:
        print('module type error')
        exit(0)


def choose_binset(full_seq, bin_set, bin_class,count_num=5,highres=0):
    if bin_set == 'bin_2':
        full_seq = full_seq[:,:-1]
        full_seq[full_seq > (bin_class - 2)] = bin_class - 2
        full_seq = full_seq.long()
    elif bin_set == 'bin_3':
        full_seq = full_seq[:,:-1]
        full_seq[full_seq == 0] = -1
        full_seq[full_seq > (bin_class - 3)] = bin_class - 3
        full_seq = (full_seq.long() + 1).long()
    elif bin_set == 'no_bin':
        pass

    elif bin_set == 'bin_3_resolution_append':
        inputsum = full_seq[:,-1].unsqueeze(1)
        full_seq = full_seq[:,:-1]
        interval = 2/(count_num-1)
        sappend = bin_class-count_num+torch.clamp(torch.log10(inputsum),min=2,max=4)//interval - 2//interval
        tappend = bin_class-count_num+torch.clamp(torch.log10(inputsum)+highres,min=2,max=4)//interval - 2//interval

        full_seq[full_seq==0]=-1  
        full_seq[full_seq > (bin_class - 3 - count_num)] = bin_class - 3 - count_num  
        full_seq = (full_seq + 1).long()  
        full_seq = torch.cat((full_seq, tappend,sappend),dim=1).long() 

    elif bin_set == 'autobin':
        full_seq = full_seq[:,:-1]

    elif bin_set == 'autobin_resolution_append':
        inputsum = full_seq[:,-1].unsqueeze(1)
        full_seq = full_seq[:,:-1]
        inputindex = torch.log10(inputsum)
        full_seq = torch.cat((full_seq,inputindex+highres,inputindex),dim=1) 
    else:
        print('{} is wrong!'.format(bin_set))
        raise NotImplementedError
    return full_seq

class PerformerLikeEncoder(nn.Module):
    def __init__(self, config, hidden_size=None):
        super().__init__()
        self.config = config
        ckp = torch.load(self.config['load_path'])
        model_config = ckp["configs"]
        self.model_config = model_config

        model_type = None
        if "model_type" in self.model_config:
            model_type = self.model_config["model_type"]
        elif self.config.get("model_type", None) is not None:
            model_type = self.config["model_type"]
        assert model_type is not None, "model_type should be provided in either model_config or configs"
        self.model_type = model_type
        kwargs = self.init_module(ckp)
        self.num_tokens = model_config["n_class"]

        assert hidden_size is None or hidden_size == model_config["hidden_dim"], f"gears.hidden_size ({hidden_size}) should be equal to dim ({kwargs['dim']})"

    def init_module(self, ckp):
        model_type = self.model_type
        model_config = self.model_config

        '''
        from minsheng
        '''
        for cfg in ['dim_head','emb_dropout','ff_dropout','attn_dropout']:
            if cfg not in model_config:
                if cfg == 'dim_head':
                    model_config[cfg]=64
                else:
                    model_config[cfg]=0

        kwargs = dict(
            num_tokens=model_config["n_class"],
            max_seq_len=model_config["seq_len"],
            dim=model_config["hidden_dim"],
            depth=model_config["depth"],
            heads=model_config["heads"],
            dim_head=model_config["dim_head"],
            g2v_position_emb=False,
            emb_dropout=model_config['emb_dropout'],
            ff_dropout=model_config['ff_dropout'],
            attn_dropout=model_config['attn_dropout']
        )

        from ..modules.performergau import PerformerGAU_mse
        self.m = PerformerGAU_mse(**kwargs)

        m_state_dict = ckp["model_state_dict"]
        self.m.load_state_dict(m_state_dict)
        del self.m.to_out
        del self.m.to_final

        return kwargs

    @classmethod
    def from_params(cls, **kwargs):
        from omegaconf import OmegaConf
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(config)

    def forward(self, x):
        x = x.clone()
        x = choose_binset(x, self.config['bin_set'], self.num_tokens)

        return self.m.forward(x, return_encodings=True)

class MAEAutobinencoder(nn.Module):
    def __init__(self, config, hidden_size=None):
        super().__init__()
        self.config = config
        ckp = torch.load(self.config['load_path'])
        ckp = ckp['gene']
        ckp = convertconfig(ckp)
        model_config = ckp["configs"]
        self.model_config = model_config

        model_type = None
        if "model_type" in self.model_config:
            model_type = self.model_config["model_type"]
        elif self.config.get("model_type", None) is not None:
            model_type = self.config["model_type"]
        assert model_type is not None, "model_type should be provided in either model_config or configs"
        self.model_type = model_type
        kwargs = self.init_module(ckp)
        self.num_tokens = model_config["n_class"]

        assert hidden_size is None or hidden_size == model_config['decoder']['hidden_dim'], f"gears.hidden_size ({hidden_size}) should be equal to dim ({kwargs['decoder_embed_dim']})"

    def init_module(self, ckp):
        model_type = self.model_type
        model_config = self.model_config

        encoder_config =model_config['encoder']
        decoder_config = model_config['decoder']
        encoder = select_module(model_config, encoder_config, model_config['encoder']['module_type'])
        decoder = select_module(model_config, decoder_config, model_config['decoder']['module_type'])

        kwargs = dict(
            num_tokens=model_config['n_class'],
            max_seq_len=model_config['seq_len'],
            embed_dim=model_config['encoder']['hidden_dim'],
            decoder_embed_dim=model_config['decoder']['hidden_dim'],
            bin_num=model_config['bin_num'],
            pad_token_id=model_config['pad_token_id'],
            mask_token_id=model_config['mask_token_id']
        )

        model = MaeAutobin(**kwargs)
        model.encoder = encoder
        model.decoder = decoder
        self.m = model
        m_state_dict = ckp["model_state_dict"]
        self.m.load_state_dict(m_state_dict)
        self.m.to_final = None
        return kwargs

    @classmethod
    def from_params(cls, **kwargs):
        from omegaconf import OmegaConf
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(config)

    def forward(self, x):
        x = x.clone()
        x = choose_binset(x, self.config['bin_set'], self.num_tokens,highres=self.config['highres'])
        encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(x,x,self.model_config)
        out = self.m.forward(x=encoder_data, padding_label=encoder_data_padding,
                    encoder_position_gene_ids=encoder_position_gene_ids,
                    encoder_labels=encoder_labels,
                    decoder_data=decoder_data,
                    mask_gene_name=False,
                    mask_labels=None,
                    decoder_position_gene_ids=decoder_position_gene_ids,
                    decoder_data_padding_labels=decoder_data_padding,
                    )
        out = out[:,:19264,:].contiguous()
        return out