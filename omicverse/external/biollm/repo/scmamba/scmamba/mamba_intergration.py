# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:mamba_intergration.py
# @Software:PyCharm
# @Created Time:2023/12/20 10:26 AM
# %%
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
# from umap import UMAP
from typing import List, Tuple, Dict, Optional
import warnings
import torch
import scanpy as sc
from anndata import AnnData
from scanpy import read_h5ad
import datetime
from sklearn.metrics import silhouette_score
import shutil

import scvi
import numpy as np
import wandb
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import argparse
sys.path.insert(0, "../")
from scgpt.model import TransformerModel
from scgpt.utils import eval_scib_metrics
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import scgpt as scg
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from mambaLM import MambaModel
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import eval_scib_metrics
from scLLM_utils.dataset import Load_Data,SeqDataset
from scLLM_utils.utils import seed_all
from scLLM_utils.dataloader import Get_DataLoader
from scLLM_utils.utils import get_reduced
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Integration', help='Name of task')#
parser.add_argument("--data_name", type=str, default="pancreas", choices=['PBMC10K',"pancreas","perirhinal","covid"],help='Name of dataset')#
parser.add_argument("--model_name", type=str, default='mamba', choices=['mamba',"gpt"],help='Finetuned model name.')#
parser.add_argument("--distributed", type=bool, default=False, help='debug mode, single gpu device')#
parser.add_argument("--single_gpu", type=bool, default=False, help='single gpu device, but not debug mode')#
parser.add_argument("--do_train", type=bool, default=True, help='Train or inference')#
parser.add_argument("--GEPC", type=bool, default=False, help='Masked value prediction for cell embedding.')#
parser.add_argument("--run_name", type=str, default='debug', help='name of experiment.')#
parser.add_argument("--append_cls", type=bool, default=False, help='append <cls> token as first token')#
parser.add_argument("--freeze", type=bool, default=False, help='freeze')#
parser.add_argument("--umap", type=bool, default=False, help='output umap or not')#
parser.add_argument("--CLS", type=bool, default=False, help='celltype classification objective')#
parser.add_argument("--CCE", type=bool, default=False, help='contrastive cell embedding objective')#
parser.add_argument("--MLM", type=bool, default=False, help='whether to use masked language modeling, currently it is always on')#
# general hyperparameters
parser.add_argument("--epochs", type=int, default=1, help='Number of epochs.')#
parser.add_argument("--seed", type=int, default=1927, help='Random seed.')#
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate.')#
parser.add_argument("--batch_size", type=int, default=64, help='Number of batch size.')#
parser.add_argument("--lr", type=float, default=1e-6, help='Learning rate.')#
parser.add_argument("--log_interval",type=int, default=100,help='interval of log.')#
# parser.add_argument("--forcing_wandb", type=bool, default=False, help='whether open wandb by force')#
parser.add_argument("--save_eval_interval",type=int, default=5,help='interval of evaluation')#
parser.add_argument("--schedule_ratio",type=float, default=0.9,help='ratio of epochs for learning rate schedule')#
parser.add_argument("--amp",type=bool, default=True,help='Automatic Mixed Precision')#
# path-related
parser.add_argument("--data_path", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/data', help='Path of data for finetune.')#
parser.add_argument("--load_model", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/mamba/gst_ori_initemb', help='Path of pretrained model.')
parser.add_argument("--save_dir", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves', help='Directory of checkpoint and result to save.')
parser.add_argument("--vocab_file", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human/vocab.json', help='Path of vocab, available if load_model is None')
parser.add_argument("--graph_path", type=str, default='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/graph', help='Path of graph')#
# if load model, batch_size, layer_size, nlayers, nhead will be ignored
parser.add_argument("--fast_transformer", type=bool, default=True, help='Using fast-attn or not')#
parser.add_argument("--layer_size", type=int, default=128, help='Size of embedding.')#
parser.add_argument("--nhead", type=int, default=4, help='number of attention head')#
parser.add_argument("--nlayers", type=int, default=12, help='number of transformer layers')#
parser.add_argument("--mask_ratio", type=float, default=0.4, help='ratio of masked token.')#
parser.add_argument("--pre_norm", type=bool, default=False, help='normalize previously')#
parser.add_argument("--input_emb_style", type=str, default='continuous',choices=['continuous','category','scaling'], help='the style of input emb')#
parser.add_argument("--cell_emb_style", type=str, default="avg-pool",choices=['final','cls','avg-pool','w-pol','attn'], help='method for generating cell emb')#
parser.add_argument("--graph_sort", type=bool, default=True, help='using graph sorting')#
# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--n_hvg", type=int,default=-1, help='whether to subset the raw data to highly variable genes. -1: turn off hvg, positive: number of hvg')#
parser.add_argument("--ecs_thres", type=float, default=0.8, help='Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable.')#
parser.add_argument("--dab_weight", type=float, default=1.0, help='weight of dab.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=True, help='whether sort the adata by batch_id')#
parser.add_argument("--include_zero_gene", type=bool, default=False, help='whether include gene with zero expression value')#
parser.add_argument("--input_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='input representation')#
parser.add_argument("--output_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='output representation')#
parser.add_argument("--DSBN", type=bool, default=True, help='Domain-spec batchnorm')#
parser.add_argument("--explicit_zero_prob", type=bool, default=True, help='whether explicit bernoulli for zeros')#
parser.add_argument("--max_seq_len", type=int, default=3000, help='max length of gene sequence')
parser.add_argument("--bimamba_type", type=str, default='none',choices=['v1','v2','none'], help='tpye of bimamba')#

args = parser.parse_args()
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"

# choose the mode running this script: 【single-gpu, distributed or debug modes】
assert not (args.single_gpu and args.distributed)
debug=False
if args.distributed:# multi GPU mode
    rank = int(os.environ.get('RANK',0))
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
        os.environ["CUDA_VISIBLE_DEVICES"]='2'
        local_rank = 0
        args.log_interval = 10
        args.cell_emb_style='attn'
        args.model_name='mamba'
        args.epochs=1
        debug = True
        args.MLM=False
        # args.umap = True
        # args.GEPC = True
        # args.CCE=True
        args.CLS=True
        args.data_name = "perirhinal"
        # "perirhinal","covid"
        # args.load_model='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/saves/Pretraining/cellxgene/Mamba/gst_emb_attnMVC_mr3'
        args.dab_weight=1
        args.n_hvg = 1200
    is_master=True
    world_size=1
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)
print(f'current devices: {device}')
if args.model_name=='gpt':
    args.append_cls=True
    args.cell_emb_style='cls'
    args.load_model = '/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human'

if is_master:
    model_ckpt_name = args.load_model.split('/')[-1] if args.load_model is not None else 'from_scratch'
    ## wandb setting
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name = f'{args.task}_{args.data_name}_{args.model_name}_(ckpt-{model_ckpt_name})_{args.run_name}' \
                 f'_{args.cell_emb_style}_hvg{args.n_hvg}_{now}'
    wandb_tags = ['Finetune', args.task, args.data_name,args.model_name,'gtp_sort' if args.graph_sort else '',\
                  args.data_name,args.cell_emb_style]
    run = wandb.init(
        config=args.__dict__,
        job_type=args.task,
        project="scLLM-ITG",
        name=wandb_name,
        tags=wandb_tags,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    print(args.__dict__)

# set_seed(args.seed)
# %%
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = args.mask_ratio
n_input_bins = args.n_bins
mask_value = -1
pad_value = -2

if args.n_hvg!=-1:
    n_hvg = args.n_hvg
    args.max_seq_len = n_hvg + 1
else:
    n_hvg =False # number of highly variable genes
per_seq_batch_sample = args.per_seq_batch_sample
DSBN = args.DSBN  # Domain-spec batchnorm
explicit_zero_prob = args.explicit_zero_prob  # whether explicit bernoulli for zeros

# %% validate settings
assert args.input_style in ["normed_raw", "log1p", "binned"]
assert args.output_style in ["normed_raw", "log1p", "binned"]
assert args.input_emb_style in ["category", "continuous", "scaling"]
if args.input_style == "binned":
    if args.input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif args.input_style == "log1p" or args.input_style == "normed_raw":
    if args.input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )
if args.input_emb_style == "category":
    mask_value = args.n_bins + 1
    pad_value = args.n_bins  # for padding gene expr values
    n_input_bins = args.n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = args.n_bins
# %%
if is_master:
    save_dir=os.path.join(args.save_dir,args.task,args.data_name,args.model_name,args.run_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
else:
    logger=None

############Step 2: Load the pre-trained scMamba model params and disc.###################
if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if is_master:
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = args.layer_size  # embedding dimension
    d_hid = args.layer_size  # dimension of the feedforward network in TransformerEncoder
    nlayers = args.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
    nhead = args.nhead  # number of heads in nn.MultiheadAttention
    dropout = args.dropout  # dropout probability
    n_layers_cls =3
    vocab_file=args.vocab_file
vocab = GeneVocab.from_file(vocab_file)
pad_token = "<pad>"
mask_token='<mask>' if '<mask>' in vocab.vocab.itos_ else '<eoc>'
unk_token='<pad>' if args.append_cls else '<cls>'
special_tokens = [pad_token, mask_token, unk_token]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
vocab.set_default_index(vocab["<pad>"])
if is_master:
    shutil.copy(vocab_file, save_dir / "vocab.json")

###########################Step 3: Load and pre-process data for the cell-type annotation task##########################
data_path=os.path.join(args.data_path,args.task,args.data_name)
train_data_pt,valid_data_pt,test_data_pt,num_batch_types,adata_test,num_cell_types,id2type=Load_Data(data_path=data_path,args=args,logger=logger,
                                                                 vocab=vocab,is_master=is_master,mask_value=mask_value,
                                                                 pad_value=pad_value,pad_token=pad_token)

###########################Step 4: Model instantiation and params loading##########################
ntokens = len(vocab)  # size of vocabulary
if args.model_name=='mamba':
    model =MambaModel(
        ntoken=ntokens,
        d_model=embsize,
        nlayers=nlayers,
        nlayers_cls=3,
        n_cls=num_cell_types if args.CLS else 1,
        device=device,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=args.GEPC,
        do_dab=True,
        do_cce=args.CCE,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        cell_emb_style=args.cell_emb_style,
        ecs_threshold=args.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        pre_norm=args.pre_norm,
        do_pretrain=False,
        if_bimamba=args.bimamba_type!="none",
        bimamba_type=args.bimamba_type,
        if_devide_out=False,
        init_layer_scale=None)
elif args.model_name=='gpt':
    model = TransformerModel(
        ntoken=ntokens,
        d_model=embsize,
        nhead=nhead,
        d_hid=embsize,
        nlayers=nlayers,
        nlayers_cls=3,
        n_cls=num_cell_types if args.CLS else 1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=args.GEPC,
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        ecs_threshold=args.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=args.fast_transformer,
        pre_norm=args.pre_norm,
    )
if args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        if is_master:
            logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        ckpt_emb_shape=pretrained_dict['encoder.embedding.weight'].size()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if not 'encoder.embedding.weight' in pretrained_dict:
            logger.warning(f'{"!"*30}Embeddings Unavailable{"!"*30}\n'
                           f'Expected shape: {model_dict["encoder.embedding.weight"].size()}\n'
                           f'But got shape: {ckpt_emb_shape} from ckpt {model_file}')
        if is_master:
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    if args.freeze and "encoder" in name and "transformer_encoder" not in name:
    # if args.freeze and "encoder" in name:
        if is_master:
            print(f"freezing weights for: {name}")
        para.requires_grad = False
if is_master:
    post_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    wandb.log(
            {
                "info/pre_freeze_param_count": pre_freeze_param_count,
                "info/post_freeze_param_count": post_freeze_param_count,
            },
    )
model.to(device)
if args.distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_cls,total_cce=0.0,0.0
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                CLS=args.CLS,
                CCE=args.CCE,
                MVC=args.GEPC,
                ECS=args.ecs_thres > 0,
            )
            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss=0.0
            metrics_to_log=dict()
            if args.CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if args.CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            if args.MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log.update({"train/mse": loss_mse.item()})
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})

            if args.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if args.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_gepc_zero_log_prob.item()})
            if args.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + args.dab_weight* loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                if is_master:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
        scaler.step(optimizer)
        scaler.update()
        if is_master:
            wandb.log(metrics_to_log)
        if args.MLM:
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

        total_loss += loss.item()
        total_mse += loss_mse.item() if args.MLM else 0.0
        total_gepc += loss_gepc.item() if args.GEPC else 0.0
        total_cls+= loss_cls.item() if args.CLS else 0.0
        total_cce += loss_cce.item() if args.CCE else 0.0
        total_error += mre.item() if args.MLM else 0.0
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval
            cur_cls=total_cls/log_interval
            cur_cce=total_cce/log_interval

            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            if args.distributed:
                cur_loss=get_reduced(cur_loss, local_rank, 0, world_size)
                cur_mse = get_reduced(cur_mse, local_rank, 0, world_size)
                cur_gepc = get_reduced(cur_gepc, local_rank, 0, world_size)
                cur_error = get_reduced(cur_error, local_rank, 0, world_size)
                cur_cls = get_reduced(cur_cls, local_rank, 0, world_size)
                cur_cce = get_reduced(cur_cce, local_rank, 0, world_size)


            if is_master:
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if args.GEPC else "")
                    + (f"cls {cur_cls:5.2f} |" if args.CLS else "")
                    + (f"cce {cur_cce:5.2f} |" if args.CCE else "")
                )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_cls=0
            total_cce=0
            total_error = 0
            start_time = time.time()
        if debug:
            break
            #pass


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    if args.distributed:
        dist.barrier()
    total_mlm = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    total_mvc=0.0
    total_cce,total_cls=0.0,0.0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    MVC=args.GEPC,
                    CLS=args.CLS,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                )
                loss=0.0
                if args.MLM:
                    mlm_output = output_dict["mlm_output"]

                    masked_positions = input_values.eq(mask_value)
                    loss_mlm = criterion(mlm_output, target_values, masked_positions)
                    loss+=loss_mlm
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                if args.GEPC:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                if args.CLS:
                    cls_out = output_dict["cls_output"]
                    loss_cls = criterion_cls(cls_out, celltype_labels)
            total_mlm += loss_mlm.item() * len(input_gene_ids) if args.MLM else 0
            total_error += masked_relative_error(
                mlm_output, target_values, masked_positions
            ).item() * len(input_gene_ids) if args.MLM else 0
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_mvc+=loss_gepc.item() * len(input_gene_ids) if args.GEPC else 0
            total_cls+=loss_cls.item()* len(input_gene_ids) if args.CLS else 0
            total_num += len(input_gene_ids)
        if args.distributed:
            mlm = get_reduced(total_mlm / total_num, local_rank, 0, world_size)
            mre = get_reduced(total_error / total_num, local_rank, 0, world_size)
            dab = get_reduced(total_dab / total_num, local_rank, 0, world_size)
            mvc=get_reduced(total_mvc / total_num, local_rank, 0, world_size)
            cls=get_reduced(total_cls / total_num, local_rank, 0, world_size)
            sum_mse_dab = mlm + args.dab_weight * dab
            sum_mse_mvc= mlm + mvc
        else:
            mlm = total_mlm / total_num
            mre = total_error / total_num
            dab = total_dab / total_num
            mvc=total_mvc / total_num
            cls=total_cls / total_num
            sum_mse_dab = (total_mlm + args.dab_weight*total_dab) / total_num
            sum_mse_mvc = (total_mlm + total_mvc) / total_num
    if is_master:
        wandb.log(
            {
                "valid/mlm": mlm,
                "valid/mre": mre,
                "valid/dab": dab,
                "valid/mvc": mvc,
                "valid/cls": cls,
                "valid/sum_mse_dab": sum_mse_dab,
                "valid/sum_mse_mvc": sum_mse_mvc,
                "epoch": epoch,
            },
        )

    return mlm,mre,sum_mse_mvc,cls


def eval_testdata(
    model: nn.Module,
    adata_t,
    test_data_pt,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()
    batch_ids = test_data_pt["batch_labels"]
    celltypes= test_data_pt["celltype_labels"]
    # Evaluate cls cell embeddings
    if "cls" in include_types:
        if is_master:
            logger.info("Evaluating cls cell embeddings")
        all_gene_ids, all_values = test_data_pt["gene_ids"], test_data_pt["values"]

        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            if args.distributed:
                model=model.module
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=args.batch_size,
                batch_labels=batch_ids.long() if DSBN else None,
                time_step=0,
                return_np=True,

            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_emb"] = cell_embeddings
        # dep_res=eval_scib_metrics(adata_t,embed_key="X_pca")
        results = eval_scib_metrics(adata_t,embed_key="X_emb")

        # Calculate silhouette score for cell types
        # celltype_silhouette_score = silhouette_score(cell_embeddings, celltypes)#adata_t.obs['celltype']
        #
        # # Calculate silhouette score for batches
        # batch_silhouette_score = silhouette_score(cell_embeddings, batch_ids)#adata_t.obs['batch_ids']
        # results = {'ASW/celltype':celltype_silhouette_score,'ASW/batch':batch_silhouette_score}
        if args.umap:
            # adata_t.obsm["X_test"] = np.random.randn(adata_t.n_obs, 200)
            # sc.pp.neighbors(adata_t, use_rep="X_test")
            sc.pp.neighbors(adata_t, use_rep="X_emb")
            sc.tl.umap(adata_t, min_dist=0.3)
            # umap_obj=UMAP(n_neighbors=10,n_components=2,min_dist=0.3)
            # adata_t.obsm['umap']=umap_obj.fit_transform(adata_t.obsm['X_emb'])
            fig = sc.pl.umap(
                adata_t,
                color=["str_batch"],
                title=[f"batch, ASW/batch = {results.get('ASW_label/batch', 0.0):.4f}"],
                frameon=False,
                return_fig=True,
                show=False,
            )
            results["batch_umap"] = fig

            # sc.pp.neighbors(adata_t, use_rep="X_emb")
            # sc.tl.umap(adata_t, min_dist=0.3)
            fig = sc.pl.umap(
                adata_t,
                color=["celltype"],
                title=[
                    f"celltype, ASW/celltype = {results.get('ASW_label', 0.0):.4f}",
                ],
                frameon=False,
                return_fig=True,
                show=False,
            )

            results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results


# %%
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
if is_master:
    define_wandb_metrcis()
    wandb.watch(model)

for epoch in range(1, args.epochs + 1):
    if args.distributed:
        dist.barrier()
    epoch_start_time = time.time()
    train_loader = Get_DataLoader(train_data_pt,args=args,shuffle=False,
                                intra_domain_shuffle=True,drop_last=False)
    valid_loader = Get_DataLoader(valid_data_pt,args=args,shuffle=False,
                                intra_domain_shuffle=False,drop_last=False)
    if args.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_mre,sum_loss,cls_loss = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    if is_master:
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)
    if args.CLS:
        sum_loss=cls_loss
    if sum_loss < best_val_loss:
        best_val_loss = sum_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        if is_master:
            logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % args.save_eval_interval == 0 or epoch == args.epochs:
        if is_master:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            # eval on testdata
            results = eval_testdata(
                best_model,
                adata_t=adata_test,
                test_data_pt=test_data_pt,
                include_types=["cls"],
            )
            if args.umap:
                results["batch_umap"].savefig(
                    save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
                )

                results["celltype_umap"].savefig(
                    save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
                )
            metrics_to_log = {"test/" + k: v for k, v in results.items()}
            if args.umap:
                metrics_to_log["test/batch_umap"] = wandb.Image(
                    str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )

                metrics_to_log["test/celltype_umap"] = wandb.Image(
                    str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )
            metrics_to_log["test/best_model_epoch"] = best_model_epoch
            wandb.log(metrics_to_log)
            wandb.log({"avg_bio": results.get("avg_bio", 0.0)})
    if args.distributed:
        dist.barrier()
    scheduler.step()


# %%
# save the best model
if is_master:
    if args.distributed:
        torch.save(best_model.module.state_dict(), save_dir / "best_model.pt")
    else:
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    # %% [markdown]
    # ## Gene embeddings

    # %%
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)

    run.finish()
    wandb.finish()
    gc.collect()

