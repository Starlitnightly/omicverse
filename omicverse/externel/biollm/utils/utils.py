#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: utils.py.py
@time: 2024/3/3 11:26
"""
import munch
import toml
import json
import numpy as np
import torch
from functools import wraps
#import pynvml
from sklearn.model_selection import train_test_split


def load_config(config_file):
    args = munch.munchify(toml.load(config_file))
    if args.model_used in ('scgpt', 'scmamba'):
        with open(args.model_param_file, 'r') as fd:
            params = json.load(fd)
        for p in params:
            if p not in args:
                args[p] = params[p]
    return args


def gene2vec_embedding(g2v_file, g2v_genes):
    gene2vec_weight = np.load(g2v_file)
    gene_emb_dict = {}
    with open(g2v_genes, 'r') as fd:
        gene_list = [line.strip('\n') for line in fd]
    for i in range(len(gene_list)):
        gene_emb_dict[gene_list[i]] = gene2vec_weight[i]
    return gene_emb_dict


def cal_model_params(model: torch.nn.Module) -> int:
    """
    calculate model parameters
    """
    model_param_count = 0
    for param in model.parameters():
        model_param_count += param.numel()
    return model_param_count


def cal_gpu_memory(gpu_index: int):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return memory_info.used / (1024**3)


def gpu_memory(gpu_index: int):
    def gpu_resource(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            memory = cal_gpu_memory(gpu_index)
            print('gpu: ', memory, 'G')
        return wrapper
    return gpu_resource


def get_reduced(tensor, device, dest_device):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / torch.distributed.get_world_size()
    return tensor_mean


def distributed_concat(tensor, num_total_examples, world_size):
    """
    合并不同进程的inference结果
    """
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def split_data(x: np.ndarray, y: np.ndarray, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split data into training, validation, and test sets.

    Parameters:
    x (np.ndarray): Feature matrix.
    y (np.ndarray): Labels.
    train_ratio (float): Proportion of data for training.
    val_ratio (float): Proportion of data for validation.
    test_ratio (float): Proportion of data for testing.
    random_state (int): Random seed for reproducibility.

    Returns:
    Tuple: (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(1 - train_ratio), random_state=random_state)
    val_size = val_ratio / (val_ratio + test_ratio)  # Normalize validation size in remaining data
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(1 - val_size), random_state=random_state)

    return x_train, y_train, x_val, y_val, x_test, y_test
