# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:mamba_CA_new.py
# @Software:PyCharm
# @Created Time:2024/3/6 4:55 PM
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import dgl
import time
from typing import Tuple, Dict
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import AdversarialDiscriminator,TransformerModel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
)
from mambaLM import MambaModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scLLM_utils.dataset import Load_Data,SeqDataset
from scLLM_utils.utils import seed_all
from scLLM_utils.dataloader import Get_DataLoader
import argparse

import torch.distributed as dist
from scLLM_utils.utils import get_reduced
import datetime
from pprint import pprint
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Cell_annotation', help='Name of task')#
parser.add_argument("--data_name", type=str, default='ms', choices=['ms','mye','pancreas','zheng68k'],help='Name of dataset')#
parser.add_argument("--model_name", type=str, default='gpt',choices=['gpt','mamba'], help='Finetuned model name.')#
parser.add_argument("--run_name", type=str, default='debug', help='name of experiment.')#
parser.add_argument("--distributed", type=bool, default=False, help='debug mode, single gpu device')#
parser.add_argument("--single_gpu", type=bool, default=False, help='single gpu device, but not debug mode')#
parser.add_argument("--do_train", type=bool, default=True, help='Train or inference')#
parser.add_argument("--MVC", type=bool, default=False, help='Masked value prediction for cell embedding')#
parser.add_argument("--MLM", type=bool, default=False, help='whether to use masked language modeling, currently it is always on')#
parser.add_argument("--ADV", type=bool, default=False, help='Adversarial training for batch correction')#
parser.add_argument("--CCE", type=bool, default=False, help='contrastive cell embedding objective')#
parser.add_argument("--DAB", type=bool, default=False, help='domain adaptation by reverse backpropagation. set to 2 for separate optimizer')#
parser.add_argument("--CLS", type=bool, default=True, help='celltype classification objective')#
# general hyperparameters
parser.add_argument("--epochs", type=int, default=1, help='Number of epochs.')#
parser.add_argument("--num_workers", type=int, default=64, help='Number of workers.')#
parser.add_argument("--seed", type=int, default=1927, help='Random seed.')#
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate.')
parser.add_argument("--batch_size", type=int, default=64, help='Number of batch size.')
parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--lr_ADV", type=float, default=1e-3, help='learning rate for discriminator, used when ADV is True.')
parser.add_argument("--log_interval",type=int, default=100,help='interval of log.')
# parser.add_argument("--forcing_wandb", type=bool, default=False, help='whether open wandb by force')#
parser.add_argument("--save_eval_interval",type=int, default=5, help='interval of evaluation')#
parser.add_argument("--schedule_ratio",type=float, default=0.9, help='ratio of epochs for learning rate schedule')#
parser.add_argument("--amp",type=bool, default=True,help='Automatic Mixed Precision')#
# path-related
parser.add_argument("--data_path", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/data', help='Path of data for finetune.')#
parser.add_argument("--load_model", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human', help='Path of pretrained model.')
parser.add_argument("--save_dir", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves', help='Directory of checkpoint and result to save.')
parser.add_argument("--vocab_file", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human/vocab.json', help='Path of vocab, available if load_model is None')
parser.add_argument("--graph_path", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/graph', help='Path of graph')#
# if load model, batch_size, layer_size, nlayers, nhead will be ignored
parser.add_argument("--layer_size", type=int, default=128, help='Size of embedding.')#
parser.add_argument("--nhead", type=int, default=4, help='number of attention head')#
parser.add_argument("--nlayers", type=int, default=4, help='number of transformer layers')#
parser.add_argument("--mask_ratio", type=float, default=0.0, help='ratio of masked token.')#
parser.add_argument("--fast_transformer", type=bool, default=True, help='Using fast-attn or not')#
parser.add_argument("--fast_transformer_backend", type=str, default="flash",choices=["flash","linear"], help='architecture style of the decoder')#

parser.add_argument("--append_cls", type=bool, default=False, help='append <cls> token as first token')#
parser.add_argument("--pre_norm", type=bool, default=False, help='normalize previously')#
parser.add_argument("--freeze", type=bool, default=False, help='freeze')#
parser.add_argument("--DSBN", type=bool, default=False, help='Domain-spec batchnorm')#
parser.add_argument("--ecs_thres", type=float, default=0.0, help='Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable.')#
parser.add_argument("--dab_weight", type=float, default=0.0, help='weight of dab.')#
parser.add_argument("--input_emb_style", type=str, default='continuous',choices=['continuous','category','scaling'], help='the style of input emb')#
parser.add_argument("--cell_emb_style", type=str, default="avg-pool",choices=['final','cls','avg-pool','w-pol'], help='method for generating cell emb')#
parser.add_argument("--mvc_decoder_style", type=str, default='inner product',choices=['inner product','concat query','sum query'], help='architecture style of the decoder')#
parser.add_argument("--graph_sort", type=bool, default=True, help='using graph sorting')#
# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=False, help='whether sort the adata by batch_id')#
parser.add_argument("--include_zero_gene", type=bool, default=False, help='whether include gene with zero expression value')#
parser.add_argument("--input_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='input representation')#
parser.add_argument("--output_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='output representation')#
parser.add_argument("--max_seq_len", type=int, default=3000, help='max length of gene sequence')

# init
args = parser.parse_args()
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"

# choose the mode running this script: 【single-gpu, distributed or debug modes】
assert not (args.single_gpu and args.distributed)
if args.distributed:# multi GPU mode
    rank = int(os.environ.get('LOCAL_RANK',0))
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    is_master = local_rank == 0
    dist.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(args.seed + torch.distributed.get_rank())
else:
    if args.single_gpu:# single GPU mode
        local_rank=int(os.environ['LOCAL_RANK'])
        rank = int(os.environ["RANK"])
    else: #debug mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["CUDA_VISIBLE_DEVICES"]='3'
        local_rank = 0
        args.log_interval = 10
        args.model_name='gpt'
        args.batch_size=16
    is_master=True
    world_size=1
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)
print(f'current devices: {device}')
if is_master:
    model_ckpt_name = "ckpt-"+args.load_model.split('/')[-1] if args.load_model is not None else 'from_scratch'
    ## wandb setting
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name=f'{args.data_name}_{args.model_name}_({model_ckpt_name})_{args.run_name}' \
               f'{"_CCE" if args.CCE else ""}_{args.cell_emb_style}_{"wZero_" if args.include_zero_gene else ""}{now}'
    wandb_tags=[args.data_name,args.model_name,args.cell_emb_style,"graph_sort" if args.graph_sort else "w/o graph_sort",
                "w Zero" if args.include_zero_gene else "w/o Zero","CCE" if args.CCE else "w/o CCE","ECS" if args.ecs_thres > 0 else "No ECS"]
    if args.sweep_count==0:
        run = wandb.init(
            config=args.__dict__,
            job_type=args.task,
            project="scLLM-CA",
            name=wandb_name,
            tags=wandb_tags,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        pprint(args.__dict__)
    else:
        with open('../configs/sweep_CA.cfg','r') as fswp:
            sweep_config=json.load(fswp)
            fswp.close()
        for k,v in args.__dict__.items():
            if k not in sweep_config["parameters"].keys():
                sweep_config["parameters"].update({k:{'value':v}})
        sweep_name=f"{args.data_name}_{args.model_name}(ckpt-{model_ckpt_name}_{args.run_name})"
        sweep_config.update({'name':sweep_name})
        sweep_id = wandb.sweep(sweep=sweep_config, project="CA_swp")
        run = wandb.init()
        del args
        args=wandb.config

###################################################################################3

if is_master:
    if args.sweep_count>0:
        wandb.agent(sweep_id, main, count=args.sweep_count)
    else:
        main(args=args)
        # %%
        artifact = wandb.Artifact(f"best_model", type="model")
        glob_str = os.path.join(save_dir, "best_model.pt")
        artifact.add_file(glob_str)
        run.log_artifact(artifact)
        run.finish()
        wandb.finish()
        gc.collect()