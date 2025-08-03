# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import torch
import sys 
import os
import numpy as np
import random
from .pretrainmodels import select_model
import math
import pandas as pd

def next_16x(x):
    return int(math.ceil(x / 16) * 16)

def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic: # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def gatherDatanopad(data, labels, pad_token_id):
    max_num = labels.sum(1)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data


    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels

def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums)


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


def convertconfig(ckpt):
    newconfig = {}
    newconfig['configs']={}
    
    # Handle both 'config' and 'configs' keys
    config_key = 'configs' if 'configs' in ckpt else 'config'
    state_dict_key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
    
    if config_key not in ckpt:
        raise KeyError(f"Neither 'configs' nor 'config' found in checkpoint. Available keys: {list(ckpt.keys())}")
    
    if state_dict_key not in ckpt:
        raise KeyError(f"Neither 'state_dict' nor 'model_state_dict' found in checkpoint. Available keys: {list(ckpt.keys())}")
    
    config_data = ckpt[config_key]
    
    # Extract model type - handle different config structures
    if isinstance(config_data, dict) and 'model' in config_data:
        model_type = config_data['model']
        
        # Extract model config
        if 'model_config' in config_data and model_type in config_data['model_config']:
            for key, val in config_data['model_config'][model_type].items():
                newconfig['configs'][key] = val
        
        # Extract dataset config
        if 'dataset_config' in config_data and 'rnaseq' in config_data['dataset_config']:
            for key, val in config_data['dataset_config']['rnaseq'].items():
                newconfig['configs'][key] = val
                
    else:
        # If config_data is already flattened, use it directly
        if isinstance(config_data, dict):
            for key, val in config_data.items():
                newconfig['configs'][key] = val
            # Try to extract model_type
            model_type = config_data.get('model_type', config_data.get('model', 'flash_all'))
        else:
            # Fallback to default
            model_type = 'flash_all'
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    state_dict_data = ckpt[state_dict_key]
    
    # Handle different state dict structures
    for key, val in state_dict_data.items():
        if 'model.' in key:
            # Remove 'model.' prefix if present
            new_key = str(key).split('model.')[1]
        else:
            new_key = key
        d[new_key] = val
        
    newconfig['configs']['model_type'] = model_type
    newconfig['model_state_dict'] = d
    newconfig['configs']['pos_embed'] = False
    newconfig['configs']['device'] = 'cuda'
    return newconfig

def load_model(best_ckpt_path, device):
    model_data = torch.load(best_ckpt_path,map_location=device)
    if not model_data.__contains__('configs'):
        print('***** No configs *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['configs']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict)
    return model.cuda(),config

def load_model_frommmf(best_ckpt_path, key='gene'):
    checkpoint = torch.load(best_ckpt_path,map_location='cpu')
    
    # Check if the requested key exists
    if key not in checkpoint:
        available_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
        mmf_keys = [k for k in available_keys if k in ['gene', 'cell', 'rde']]
        
        if mmf_keys:
            raise KeyError(f"Key '{key}' not found in checkpoint. Available MMF keys: {mmf_keys}. "
                          f"Try using one of these keys instead.")
        else:
            raise KeyError(f"Key '{key}' not found in checkpoint. Available keys: {available_keys}. "
                          f"This may not be a valid MMF format checkpoint.")
    
    model_data = checkpoint[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__('configs'):
        print('***** No configs *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['configs']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict)
    return model.cuda(),config

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns

def get_genename():
    gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    return list(gene_list_df['gene_name'])

def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')

def getEncoerDecoderData(data, data_raw, config):
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
