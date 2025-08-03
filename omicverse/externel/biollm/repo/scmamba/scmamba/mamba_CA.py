# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:mamba_CA.py
# @Software:PyCharm
# @Created Time:2023/12/25 1:15 PM
# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:cell_annotation.py
# @Software:PyCharm
# @Created Time:2023/8/3 9:43 AM
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
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Cell_annotation', help='Name of task')#
parser.add_argument("--data_name", type=str, default='ms', choices=['ms','mye','pancreas','zheng68k'],help='Name of dataset')#
parser.add_argument("--model_name", type=str, default='mamba',choices=['gpt','mamba','bimamba'], help='Finetuned model name.')#
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
parser.add_argument("--load_model", type=str, default="none", help='Path of pretrained model.')
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
parser.add_argument("--cell_emb_style", type=str, default='attn',choices=['final','cls','avg-pool','w-pol','attn'], help='method for generating cell emb')#
parser.add_argument("--mvc_decoder_style", type=str, default='inner product',choices=['inner product','concat query','sum query'], help='architecture style of the decoder')#
parser.add_argument("--graph_sort", type=bool, default=True, help='using graph sorting')#
parser.add_argument("--layer_emb", type=bool, default=False, help='using layer emb or not when using graph sort')#

# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=False, help='whether sort the adata by batch_id')#
parser.add_argument("--include_zero_gene", type=bool, default=False, help='whether include gene with zero expression value')#
parser.add_argument("--input_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='input representation')#
parser.add_argument("--output_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='output representation')#
parser.add_argument("--max_seq_len", type=int, default=3000, help='max length of gene sequence')
parser.add_argument("--bimamba_type", type=str, default='none',choices=['v1','v2','none'], help='tpye of bimamba')#

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
        args.load_model='/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/ckpts/whole_human'
        args.model_name='mamba'
        args.batch_size=16
    is_master=True
    world_size=1
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)
print(f'current devices: {device}')


#################Step1: Specify hyper-parameter setup for cell-type annotation task###################
if is_master:
    model_ckpt_name = args.load_model.split('/')[-1] if args.load_model!="none" else 'from_scratch'
    ## wandb setting
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name=f'{args.task}_{args.data_name}_{args.model_name}_(ckpt-{model_ckpt_name})_{args.run_name}' \
               f'{"_MVC" if args.MVC else ""}{"_ADV" if args.ADV else ""}{"_CCE" if args.CCE else ""}{"_MLM" if args.MLM else ""}{"_CLS" if args.CLS else ""}' \
               f'_{args.cell_emb_style}_{"wZero_" if args.include_zero_gene else ""}{now}'
    wandb_tags=['Finetune',args.task,'SingleNode' if world_size<=4 else 'MultiNode',
                args.data_name,
                "MVC" if args.MVC else "w/o MVC","ADV" if args.ADV else "w/o ADV","w Zero" if args.include_zero_gene else "w/o Zero",
                "CLS" if args.CLS else "w/o CLS","CCE" if args.CCE else "w/o CCE","MLM" if args.MLM else "w/o MLM",args.cell_emb_style]
    run = wandb.init(
        config=args.__dict__,
        job_type=args.task,
        project="scLLM-CA",
        name=wandb_name,
        tags=wandb_tags,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    print(args.__dict__)
# settings for input and preprocessing
mask_value = "auto"  # for masked values, now it should always be auto
# settings for training
MLM = args.MLM  # whether to use masked language modeling, currently it is always on.
CLS = args.CLS  # celltype classification objective
ADV = args.ADV  # Adversarial training for batch correction
CCE = args.CCE  # Contrastive cell embedding objective
MVC = args.MVC  # Masked value prediction for cell embedding
ECS = args.ecs_thres > 0  # Elastic cell similarity objective
DAB = args.DAB  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
explicit_zero_prob = MLM and args.include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training
schedule_interval = max(1,int(args.epochs*0.1))

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
if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False
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
if args.load_model!="none":
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
unk_token='<unk>' if args.append_cls else '<cls>'
special_tokens = [pad_token, mask_token, unk_token]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
vocab.set_default_index(vocab["<pad>"])
if is_master:
    shutil.copy(vocab_file, save_dir / "vocab.json")

###########################Step 3: Load and pre-process data for the cell-type annotation task##########################
data_path=os.path.join(args.data_path,args.task,args.data_name)
train_data_pt,valid_data_pt,test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_test_raw=\
    Load_Data(data_path=data_path,args=args,logger=logger,vocab=vocab,is_master=is_master,
              mask_value=mask_value,pad_value=pad_value,pad_token=pad_token)

###########################Step 4: Model instantiation and params loading##########################
ntokens = len(vocab)  # size of vocabulary
if 'mamba' in args.model_name:
    model =MambaModel(
        ntoken=ntokens,
        d_model=embsize,
        nlayers=nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        device=device,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=args.DSBN,
        n_input_bins=n_input_bins,
        input_emb_style=args.input_emb_style,
        cell_emb_style=args.cell_emb_style,
        mvc_decoder_style=args.mvc_decoder_style,
        ecs_threshold=args.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        pre_norm=args.pre_norm,
        do_pretrain=False,
        if_bimamba=args.bimamba_type!="none",
        bimamba_type=args.bimamba_type,
        if_devide_out=False,
        init_layer_scale=None,
    )
elif args.model_name=='gpt':
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=args.DSBN,
        input_emb_style=args.input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=args.cell_emb_style,
        mvc_decoder_style=args.mvc_decoder_style,
        ecs_threshold=args.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=args.fast_transformer,
        fast_transformer_backend=args.fast_transformer_backend,
        pre_norm=args.pre_norm,
    )

if args.load_model!="none":
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

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=args.schedule_ratio
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=args.schedule_ratio
    )
if ADV:
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    optimizer_E = torch.optim.Adam(model.parameters(), lr=args.lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=args.schedule_ratio
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=args.schedule_ratio
    )

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
if not args.distributed and is_master:
    wandb.watch(model)

###########################Step 5: Train, Valid and Test procedure definition##########################
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        if args.graph_sort and args.layer_emb:
            sorted_layer_idx = batch_data['sorted_layer_idx'].to(device)
        else:
            sorted_layer_idx = None

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or args.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                # generative_training=False
                sorted_layer_idx=sorted_layer_idx
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            if MVC and explicit_zero_prob:
                loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if DAB:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + args.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})
            metrics_to_log.update({"train/loss": loss.item()})


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
            if is_master:
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
        scaler.step(optimizer)
        scaler.update()
        if is_master:
            wandb.log(metrics_to_log)
        if ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or args.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                # generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()


        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_adv_E += loss_adv_E.item() if ADV else 0.0
        total_adv_D += loss_adv_D.item() if ADV else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        if batch % args.log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            cur_mse = total_mse / args.log_interval
            cur_cls = total_cls / args.log_interval if CLS else 0.0
            cur_cce = total_cce / args.log_interval if CCE else 0.0
            cur_mvc = total_mvc / args.log_interval if MVC else 0.0
            cur_ecs = total_ecs / args.log_interval if ECS else 0.0
            cur_dab = total_dab / args.log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / args.log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / args.log_interval if ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / args.log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / args.log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / args.log_interval
            # ppl = math.exp(cur_loss)
            if args.distributed:
                cur_loss=get_reduced(cur_loss, local_rank, 0, world_size)
                cur_mse = get_reduced(cur_mse, local_rank, 0, world_size)
                cur_cls = get_reduced(cur_cls, local_rank, 0, world_size)
                cur_cce= get_reduced(cur_cce, local_rank, 0, world_size)
                cur_mvc = get_reduced(cur_mvc, local_rank, 0, world_size)
                cur_cce = get_reduced(cur_cce, local_rank, 0, world_size)
                cur_ecs = get_reduced(cur_ecs, local_rank, 0, world_size)
                cur_dab = get_reduced(cur_dab, local_rank, 0, world_size)
                cur_adv_E = get_reduced(cur_adv_E, local_rank, 0, world_size)
                cur_adv_D = get_reduced(cur_adv_D, local_rank, 0, world_size)
                cur_zero_log_prob = get_reduced(cur_zero_log_prob, local_rank, 0, world_size)
                cur_mvc_zero_log_prob = get_reduced(cur_mvc_zero_log_prob, local_rank, 0, world_size)

            if is_master:
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                    + (f"cls {cur_cls:5.2f} | " if CLS else "")
                    + (f"err {cur_error:5.2f} | " if CLS else "")
                    + (f"cce {cur_cce:5.2f} |" if CCE else "")
                    + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                    + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                    + (f"dab {cur_dab:5.2f} |" if DAB else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                    + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                    + (
                        f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                        if MVC and explicit_zero_prob
                        else ""
                    )
                )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")

def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    if args.distributed:
        dist.barrier()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            if args.graph_sort and args.layer_emb:
                sorted_layer_idx = batch_data['sorted_layer_idx'].to(device)
            else:
                sorted_layer_idx=None

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or args.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                    sorted_layer_idx=sorted_layer_idx
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
    if args.distributed:
        mse=get_reduced(total_loss / total_num,local_rank, 0, world_size)
        mre = get_reduced(total_error / total_num, local_rank, 0, world_size)
        dab= get_reduced(total_dab / total_num, local_rank, 0, world_size)
        sum_mse_dab=mse+args.dab_weight * dab
    else:
        mse=total_loss / total_num
        mre=total_error / total_num
        dab=total_dab / total_num
        sum_mse_dab =  (total_loss + args.dab_weight * total_dab) / total_num
    if is_master:
        wandb.log(
            {
                "valid/mse": mse,
                "valid/err": mre,
                "valid/dab": dab,
                "valid/sum_mse_dab": sum_mse_dab,
                "epoch": epoch,
            },
        )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return mse,mre
def test(model: nn.Module, test_data_pt: torch.Tensor,celltypes_labels) -> float:
    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=args.batch_size*4,
        shuffle=False,
        drop_last=False,
        #num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2) if args.model_name!='debug' else 0,
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )
    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
    if is_master:
        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}"
        )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results


##################Step 6: Finetune scGPT with task-specific objectives###################
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
if is_master:
    define_wandb_metrcis()
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
    val_loss, val_err = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    if is_master:
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
        )
        logger.info("-" * 89)
    if val_loss < best_val_loss :
        best_val_loss  = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        if is_master:
            logger.info(f"Best model with score {best_val_loss:5.4f}")
    if args.distributed:
        dist.barrier()
    scheduler.step()
    if DAB_separate_optim:
    	scheduler_dab.step()
    if ADV:
        scheduler_D.step()
        scheduler_E.step()


###################Step 7: Inference with fine-tuned scGPT model#############
#In the cell-type annotation task, the fine-tuned scGPT predicts cell-type labels for query set as inference.
# The model performance is evaluated on standard classificaton metrics.
# Here we visualize the predicted labels over the scGPT cell embeddings, and present the confusion matrix for detailed classification performance on the cell-group level.
predictions, labels, results = test(best_model, test_data_pt, test_data_pt['celltype_labels'])
adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
# plot
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"] + \
           plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}
if is_master:# only master required testing procedure
    with plt.rc_context({"figure.figsize": (8, 4), "figure.dpi": (300)}):
        sc.pl.umap(
            adata_test_raw,
            color=["celltype", "predictions"],
            palette=palette_,
            show=False,
        )
        plt.savefig(save_dir / "results.png", dpi=300)

    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": id2type
    }
    with open(save_dir / "results.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    results["test/cell_umap"] = wandb.Image(
        str(save_dir / "results.png"),
        caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
    )
    from sklearn.metrics import confusion_matrix
    celltypes = list(celltypes)
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

    results["test/confusion_matrix"] = wandb.Image(
        str(save_dir / "confusion_matrix.png"),
        caption=f"confusion matrix",
    )
    wandb.log(results)
    # save the model into the save_dir
    if args.distributed:
        torch.save(best_model.module.state_dict(), save_dir / "best_model.pt")
    else:
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    # %%
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)

    run.finish()
    wandb.finish()
    gc.collect()