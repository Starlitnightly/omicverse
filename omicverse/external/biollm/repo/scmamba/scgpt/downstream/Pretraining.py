# %%
import copy
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from typing import List, Dict, Optional
import warnings
import torch
import scanpy as sc
from anndata import AnnData

import datetime

import numpy as np
import wandb

from torch import nn
from torch.utils.data import DataLoader,random_split
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from finetune_utils import seed_all
import torch.distributed as dist
import argparse


sys.path.append("../../")
from scLLM_utils.pretrain_testdataset import prepare_test
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scLLM_utils.dataset import Load_Data,prepare_dataloader
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)

from scgpt.utils import eval_scib_metrics
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Pretraining',choices=['Cell_annotation','Integration','Pretraining'], help='Name of task')#
parser.add_argument("--data_name", type=str, default='panglao',choices=['panglao','cellxgene'], help='Name of dataset')#
parser.add_argument("--data_is_raw", type=bool, default=True, help='whether the data is raw')#
parser.add_argument("--model_name", type=str, default='scgpt', help='name of model type.')#
parser.add_argument("--run_name", type=str, default='debug', help='name of experiment.')#
parser.add_argument("--distributed", type=bool, default=False, help='debug mode, single gpu device')#
parser.add_argument("--single_gpu", type=bool, default=False, help='single gpu device, but not debug mode')#
parser.add_argument("--do_train", type=bool, default=True, help='Train or inference')#
parser.add_argument("--GEPC", type=bool, default=True, help='Masked value prediction for cell embedding.')#
# general hyperparameters
parser.add_argument("--epochs", type=int, default=30, help='Number of epochs.')#
parser.add_argument("--seed", type=int, default=1927, help='Random seed.')#
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate.')#
parser.add_argument("--batch_size", type=int, default=8, help='Number of batch size.')#
parser.add_argument("--lr", type=float, default=1e-6, help='Learning rate.')#
parser.add_argument("--num_workers", type=int, default=0, help='number of workers when processing.')#
parser.add_argument("--log_interval",type=int, default=1000,help='interval of log.')#
# parser.add_argument("--forcing_wandb", type=bool, default=False, help='whether open wandb by force')#
parser.add_argument("--save_eval_interval",type=int, default=1,help='interval of evaluation')#
parser.add_argument("--schedule_ratio",type=float, default=0.9,help='ratio of epochs for learning rate schedule')#
parser.add_argument("--amp",type=bool, default=True,help='Automatic Mixed Precision')#
# path-related
parser.add_argument("--data_path", type=str, default='/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/vocab/kidney', help='Path of preprocessed data')#
parser.add_argument("--source_path", type=str, default='/home/share/huada/home/qiuping1/workspace/llm/data/panglao/human_process', help='Path of source lmdb&h5ad data')#
parser.add_argument("--lmdb",type=bool, default=True,help='use lmdb dataset or not')#
parser.add_argument("--load_model", type=str, default=None, help='Path of pretrained model.')
parser.add_argument("--save_dir", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/saves', help='Directory of checkpoint and result to save.')
parser.add_argument("--vocab_file", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/ckpts/whole_human/vocab.json', help='Path of vocab, available if load_model is None')
parser.add_argument("--gene_array_file", type=str, default='/home/share/huada/home/qiuping1/workspace/llm/data/train_data/vocab/tmp/panglao_gene_ids.pk', help='Path of vocab, available if load_model is None')
parser.add_argument("--graph_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/graph', help='Path of graph')#


# if load model, batch_size, layer_size, nlayers, nhead will be ignored
parser.add_argument("--fast_transformer", type=bool, default=True, help='Using fast-attn or not')#
parser.add_argument("--embsize", type=int, default=512, help='Size of embedding.')#
parser.add_argument("--d_hid", type=int, default=512, help='Size of hidden state.')#
parser.add_argument("--nheads", type=int, default=8, help='number of attention head')#
parser.add_argument("--nlayers", type=int, default=12, help='number of transformer layers')#
parser.add_argument("--mask_ratio", type=float, default=0.25, help='ratio of masked token.')#
parser.add_argument("--pre_norm", type=bool, default=False, help='normalize previously')#
parser.add_argument("--n_layers_cls", type=int, default=3, help='number of transformer layers')#
parser.add_argument("--graph_sort", type=bool, default=True, help='using graph sorting')#
parser.add_argument("--sampling_etype", type=str, choices=['share_pathway_with','interact_with','co_expression','ori'],default='co_expression', help='choice of edge type when sampling')
parser.add_argument("--layer_mask", type=bool, default=False, help='using layer mask or not when using graph sort')#
parser.add_argument("--layer_emb", type=bool, default=True, help='using layer emb or not when using graph sort')#

# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--n_hvg", type=int,default=-1, help='whether to subset the raw data to highly variable genes. -1: turn off hvg, positive: number of hvg')#
parser.add_argument("--ecs_thres", type=float, default=0, help='Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable.')#
parser.add_argument("--dab_weight", type=float, default=1.0, help='weight of dab.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=False, help='whether sort the adata by batch_id')#
parser.add_argument("--DSBN", type=bool, default=False, help='Domain-spec batchnorm')#
parser.add_argument("--explicit_zero_prob", type=bool, default=False, help='whether explicit bernoulli for zeros')#
parser.add_argument("--include_zero_gene", type=bool, default=False, help='whether include gene with zero expression value')
parser.add_argument("--use_batch_labels", type=bool, default=False, help='use batch emb or not, turn it off when pretraining')#
parser.add_argument("--max_seq_len", type=int, default=3000, help='max length of gene sequence')



args = parser.parse_args()
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
assert not (args.single_gpu and args.distributed)
debug=False
if args.distributed:# multi GPU mode
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    is_master = rank == 0
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
        os.environ["CUDA_VISIBLE_DEVICES"]='0'
        local_rank = 0
        rank=0
        debug=True
        from config.debug_mode import config_correction
        args=config_correction(args,version='HPC_old')
    device = torch.device("cuda", local_rank)
    is_master=True
    world_size=1
    seed_all(args.seed)
if is_master:
    ## wandb setting
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name=f'{args.task}_{args.data_name}_{args.model_name}{"_topo" if args.graph_sort else ""}{"_hvg"+str(args.n_hvg) if args.n_hvg >0 else ""}_{"fast_" if args.fast_transformer else ""}G{world_size}_{now}'
    wandb_tags=[args.task,f'G{world_size}',"topological_sorting" if args.graph_sort else "",
                args.data_name,"fast_attn" if args.fast_transformer else 'nomal_attn',]
    run = wandb.init(
        config=args.__dict__,
        job_type=args.task,
        project="scLLM-Pretrain",
        name=wandb_name,
        tags=wandb_tags,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    print(args)
print(f'current devices: {device}')

# set_seed(args.seed)
# %%
# settings for input and preprocessing
pad_token = "<pad>"
mask_ratio = args.mask_ratio
n_input_bins = args.n_bins
mask_value = -1
pad_value = -2
valid_ratio=0.1
 # number of highly variable genes

per_seq_batch_sample = args.per_seq_batch_sample
DSBN = args.DSBN  # Domain-spec batchnorm
explicit_zero_prob = args.explicit_zero_prob  # whether explicit bernoulli for zeros


# %%
if is_master:
    save_dir=os.path.join(args.save_dir,args.task,args.data_name,args.model_name,args.run_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")
    os.system(f"cp {args.vocab_file} {os.path.join(save_dir, 'vocab.json')}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    with open(os.path.join(save_dir,'args.json'),'w') as f:
        json.dump(args.__dict__,f)
        f.close()


else:
    logger=None

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
            f"Resume model from {model_file}, the model args will be overriden by the "
            f"config {model_config_file}."
        )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    if is_master:
        logger.info(f"Train model from scratch.")
    embsize = args.embsize
    nhead = args.nheads
    nlayers = args.nlayers
    d_hid = args.d_hid
    vocab_file=args.vocab_file
    # %%
vocab = GeneVocab.from_file(vocab_file)
special_tokens = ["<pad>", "<cls>", "<eoc>","<mask>","<unk>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# %% [markdown]
# ## Loading and preparing data
if not args.lmdb:
    if is_master:print('Load h5ad dataset......')
    data_path=os.path.join(args.data_path,args.task,args.data_name)
    total_data=Load_Data(data_path=data_path,args=args,
                         vocab=vocab,mask_ratio=args.mask_ratio,append_cls=True,
                         include_zero_gene=False,need_length=True,max_seq_len=args.max_seq_len)
    if debug:
        print(total_data[100000])
    valid_size, train_size = int(len(total_data) * valid_ratio), len(total_data) - int(len(total_data) * valid_ratio)
    train_dataset, valid_dataset = random_split(total_data, [train_size, valid_size])
    train_loader = prepare_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    valid_loader = prepare_dataloader(
        valid_dataset,
        batch_size=args.batch_size,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    if is_master:
        logger.info(
            f"Total set number of cells: {valid_size+train_size}, from {total_data.n_files} h5ad files."
            f"\n\t Train: {train_size}, Valid: {valid_size}")
        if not args.include_zero_gene:
            logger.info(f"\n\t Only using non-zero genes:"
                        f"\n\tThe max length of non-zero gene: {total_data.max_non_zero_count}"
                        f"\n\tThe min length of non-zero gene: {total_data.min_non_zero_count}"
                        f"\n\tUniform the length into max_seq_len: {args.max_seq_len}")
        else:
            logger.info(f"\n\t Using all the genes, the length of whole gene: {total_data.max_non_zero_count}")
else:
    if is_master: print('Load lmdb dataset......')
    data_path = os.path.join(args.data_path, args.task, args.data_name)
    train_dataset,valid_dataset = Load_Data(data_path=data_path, args=args,
                           vocab=vocab, mask_ratio=args.mask_ratio, append_cls=True,
                           include_zero_gene=args.include_zero_gene,max_seq_len=args.max_seq_len)
    if debug:
        print(train_dataset[1927])
    train_loader = prepare_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    valid_loader = prepare_dataloader(
        valid_dataset,
        batch_size=args.batch_size,
        args=args,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    if is_master:
        logger.info(
            f"Total set number of cells: {len(train_dataset)+len(valid_dataset)}."
            f"\n\t Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        if not args.include_zero_gene:
            logger.info(f"\n\t Only using non-zero genes:"
                        f"\n\tUniform the length into max_seq_len: {args.max_seq_len}")
        else:
            logger.info(f"\n\t Using all the genes, the length of whole gene, uniform the length into max_seq_len: {args.max_seq_len}")

test_path=os.path.join('/home/share/huada/home/jiangwenjian/proj/scGPT/data/','Pretraining',"test")
adata,gene_ids,gene_ids_in_vocab=prepare_test(test_path,vocab,is_master,args,logger)





ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=args.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=args.GEPC,
    do_dab=False,
    use_batch_labels=args.use_batch_labels,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    ecs_threshold=args.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=args.fast_transformer,
    pre_norm=args.pre_norm,
    topo_graph=args.graph_sort,
    layer_emb=args.layer_emb,
)
if is_master:
    # params={}
    # for name, param in model.named_parameters():
    #     params.update({name:param.numel()})
    total_params = sum(p.numel() for p in model.parameters())
    params={'total_params':total_params}
    print(params)
    wandb.log(params)
if args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        if is_master:
            logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if is_master:
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)
if args.distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

criterion = masked_mse_loss
if args.graph_sort:
    lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, eps=1e-4 if args.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

if is_master:
    wandb.watch(model)

def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch_idx,batch_data in enumerate(loader):

        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["masked_values"].to(device)# masked-> -1
        target_values = batch_data["target_values"].to(device)
        # batch_labels = batch_data["batch_labels"].to(device)
        if args.graph_sort:
            target_sorted_gene = batch_data['sorted_gene_ids'].to(device)
            input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(device)
            sorted_layer_idx = batch_data['sorted_layer_idx'].to(device)
            topo_padding_mask = input_sorted_gene.eq(vocab['<pad>']).to(device)
            topo_needed_to_pred = input_sorted_gene.eq(vocab['<mask>']).to(device)
            masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred)
            target_sorted_gene[masked_pos] = -100
        else:
            input_sorted_gene=None
            topo_padding_mask=None
            sorted_layer_idx = None

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                src=input_gene_ids,
                values=input_values,
                batch_labels=None,
                MVC=args.GEPC,
                src_key_padding_mask=src_key_padding_mask,
                input_sorted_gene=input_sorted_gene,
                topo_padding_mask=topo_padding_mask,
                sorted_layer_idx=sorted_layer_idx

            )

            #masked_positions = input_values.eq(mask_value)  # the postions to predict
            # padded_positions=input_values.eq(pad_value)
            masked_positions = input_values.eq(mask_value)
            #mask=torch.logical_or(padded_positions , masked_positions)
            # loss = loss_mse = criterion(
            #     output_dict["causal_output"][:,:-1], target_values[:,1:], mask[:,:-1]
            # )
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mlm": loss_mse.item()}
            if args.graph_sort:
                logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                label = target_sorted_gene.view(-1).long()
                topo_sorting_loss=lm_criterion(logit,label)

                weight = loss_mse.item()/topo_sorting_loss.item()
                loss = loss + weight*topo_sorting_loss
                metrics_to_log.update({"train/topo_loss": topo_sorting_loss.item()})

            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if args.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values,masked_positions
                )
                weight = loss_mse.item()/loss_gepc.item()
                loss = loss + weight*loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if args.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values,masked_positions
                )

                loss = loss + loss_gepc_zero_log_prob

                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if args.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
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

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if args.GEPC else 0.0
        total_error += mre.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if args.GEPC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            if args.distributed:
                cur_loss=get_reduced(cur_loss, local_rank, 0, world_size)
                cur_mse = get_reduced(cur_mse, local_rank, 0, world_size)
                cur_gepc = get_reduced(cur_gepc, local_rank, 0, world_size)
                cur_error= get_reduced(cur_error, local_rank, 0, world_size)
            if is_master:
                logger.info(
                    f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if args.GEPC else "")
                )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()
            if debug:
                break


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
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["masked_values"].to(device)
            target_values = batch_data["target_values"].to(device)
            if args.graph_sort:
                target_sorted_gene = batch_data['sorted_gene_ids'].to(device)
                input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(device)
                sorted_layer_idx = batch_data['sorted_layer_idx'].to(device)
                topo_padding_mask = input_sorted_gene.eq(vocab['<pad>']).to(device)
                topo_needed_to_pred = input_sorted_gene.eq(vocab['<mask>']).to(device)
                masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred)
                target_sorted_gene[masked_pos] = -100
            else:
                input_sorted_gene = None
                topo_padding_mask = None
                sorted_layer_idx = None
            # batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict=model(
                    src=input_gene_ids,
                    values=input_values,
                    MVC=args.GEPC,
                    batch_labels=None,
                    src_key_padding_mask=src_key_padding_mask,
                    input_sorted_gene=input_sorted_gene,
                    topo_padding_mask=topo_padding_mask,
                    sorted_layer_idx=sorted_layer_idx
                )
                # causal loss
                output_values = output_dict["mlm_output"]
                # masked_positions = input_values.eq(mask_value)
                # padded_positions = input_values.eq(pad_value)
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                # mask = torch.logical_or(padded_positions, masked_positions)
                # loss = criterion(output_values[:,:-1], target_values[:,1:], mask[:,:-1])
                if args.graph_sort:
                    logit=output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                    label=target_sorted_gene.view(-1).long()
                    topo_sorting_loss = lm_criterion(logit, label)
                    loss = loss + topo_sorting_loss
                if args.GEPC:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                # loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            # total_error += masked_relative_error(
            #     output_values[:,:-1], target_values[:,1:], mask[:,:-1]
            # ).item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            #total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
            if debug:
                break
    if args.distributed:
        mse=get_reduced(total_loss / total_num,local_rank, 0, world_size)
        mre = get_reduced(total_error / total_num / total_num, local_rank, 0, world_size)
    else:
        mse=total_loss / total_num
        mre=total_error / total_num
    if is_master:
        wandb.log(
            {
                "valid/mse": mse,
                "valid/mre": mre,
                #"valid/dab": total_dab / total_num,
                #"valid/sum_mse_dab": (total_loss + args.dab_weight * total_dab)
                #/ total_num,
                "epoch": epoch,
            })

    return mse, mre
def get_reduced(tensor, current_device, dest_device, world_size):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(current_device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / world_size
    return tensor_mean

def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        if is_master:
            logger.info("Evaluating cls cell embeddings")
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            if args.distributed:
                available_gene_emb=model.cpu().module.encoder.embedding(torch.tensor(gene_ids)).numpy()#[gene_num,emb]
            else:
                available_gene_emb = model.cpu().encoder.embedding(
                    torch.tensor(gene_ids)).numpy()  # [gene_num,emb]
            #print(f'adata_t: {adata_t.shape}, gene: {available_gene_emb.shape}')
            cell_embeddings=adata_t.X@available_gene_emb[gene_ids_in_vocab>=0,:]
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            if is_master:
                logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results
def callback(save_dir,adata,best_model,best_model_epoch):
    # eval on testdata
    with torch.no_grad():
        results = eval_testdata(
            best_model,
            adata_t=adata,
            include_types=["cls"],
        )
    results["batch_umap"].savefig(
        save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
    )
    results["celltype_umap"].savefig(
        save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
    )
    metrics_to_log = {"test/" + k: v for k, v in results.items()}
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
    return

# %%
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model_epoch=0
best_model = model
if is_master:
    define_wandb_metrcis()
    # callback(save_dir, adata, best_model, best_model_epoch)
    # model.to(device)
for epoch in range(1, args.epochs + 1):
    if args.distributed:
        dist.barrier()
    epoch_start_time = time.time()
    if args.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_mre = evaluate(
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        if is_master:
            logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % args.save_eval_interval == 0 or epoch == args.epochs:
        if is_master:
            logger.info(f"Saving model to {save_dir}")
            if args.distributed:
                torch.save(best_model.module.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")
            else:
                torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")
            callback(save_dir,adata,best_model,best_model_epoch)
    if debug:
        results = eval_testdata(
            best_model,
            adata_t=adata,
            include_types=["cls"],
        )
        break
    if args.distributed:
        dist.barrier()
    scheduler.step()
    if is_master:
        print(f'invalid datapoint: {train_dataset.invalid_datapoint_count}')
        train_dataset.invalid_datapoint_count=0


# %%
# save the best model
if is_master:
    if args.distributed:
        torch.save(best_model.module.state_dict(), save_dir / "best_model.pt")
    else:
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)

    run.finish()
    wandb.finish()
    gc.collect()

# %% [markdown]
# ## Gene embeddings

