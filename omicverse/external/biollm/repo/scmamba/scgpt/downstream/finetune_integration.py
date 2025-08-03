# %%
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import torch
import scanpy as sc
from anndata import AnnData
from scanpy import read_h5ad
import datetime

import scvi
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from finetune_utils import seed_all
import torch.distributed as dist
import argparse
sys.path.append("../../")
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Integration',choices=['Cell_annotation','Integration'], help='Name of task')#
parser.add_argument("--data_name", type=str, default='PBMC10K', help='Name of dataset')#
parser.add_argument("--data_is_raw", type=bool, default=True, help='whether the data is raw')#
parser.add_argument("--model_name", type=str, default='Debug', help='Finetuned model name.')#
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
parser.add_argument("--log_interval",type=int, default=100,help='interval of log.')#
# parser.add_argument("--forcing_wandb", type=bool, default=False, help='whether open wandb by force')#
parser.add_argument("--save_eval_interval",type=int, default=5,help='interval of evaluation')#
parser.add_argument("--schedule_ratio",type=float, default=0.9,help='ratio of epochs for learning rate schedule')#
parser.add_argument("--amp",type=bool, default=True,help='Automatic Mixed Precision')#
# path-related
parser.add_argument("--data_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/data', help='Path of data for finetune.')#
parser.add_argument("--load_model", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/saves/whole_human', help='Path of pretrained model.')
parser.add_argument("--save_dir", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/saves', help='Directory of checkpoint and result to save.')
# if load model, batch_size, layer_size, nlayers, nhead will be ignored
parser.add_argument("--fast_transformer", type=bool, default=True, help='Using fast-attn or not')#
parser.add_argument("--layer_size", type=int, default=128, help='Size of embedding.')#
parser.add_argument("--nhead", type=int, default=4, help='number of attention head')#
parser.add_argument("--nlayers", type=int, default=4, help='number of transformer layers')#
parser.add_argument("--mask_ratio", type=float, default=0.4, help='ratio of masked token.')#
parser.add_argument("--pre_norm", type=bool, default=False, help='normalize previously')#

# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--n_hvg", type=int,default=1200, help='whether to subset the raw data to highly variable genes. -1: turn off hvg, positive: number of hvg')#
parser.add_argument("--ecs_thres", type=float, default=0.8, help='Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable.')#
parser.add_argument("--dab_weight", type=float, default=1.0, help='weight of dab.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=True, help='whether sort the adata by batch_id')#
parser.add_argument("--DSBN", type=bool, default=True, help='Domain-spec batchnorm')#
parser.add_argument("--explicit_zero_prob", type=bool, default=True, help='whether explicit bernoulli for zeros')#

args = parser.parse_args()
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
assert not (args.single_gpu and args.distributed)
if args.distributed:# multi GPU mode
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
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
    is_master=True
    world_size=1
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)
print(f'current devices: {device}')



hyperparameter_defaults = dict(
    seed=args.seed,
    model='gpt',
    dataset_name=args.data_name,
    distributed=args.distributed,
    do_train=args.do_train,
    load_model=args.load_model,
    task=args.task,
    model_name=args.model_name,
    data_path=args.data_path,
    save_dir=args.save_dir,
    mask_ratio=args.mask_ratio,
    epochs=args.epochs,
    n_bins=args.n_bins,
    GEPC=args.GEPC,  # Masked value prediction for cell embedding
    ecs_thres=args.ecs_thres,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=args.dab_weight,
    lr=args.lr,
    batch_size=args.batch_size,
    layer_size=args.layer_size,
    nlayers=args.nlayers,
    nhead=args.nhead,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=args.dropout,
    schedule_ratio=args.schedule_ratio,  # ratio of epochs for learning rate schedule
    save_eval_interval=args.save_eval_interval,
    log_interval=args.log_interval,
    fast_transformer=args.fast_transformer,
    pre_norm=args.pre_norm,
    amp=args.amp,  # Automatic Mixed Precision
    explicit_zero_prob=args.explicit_zero_prob,
    DSBN=args.DSBN,
    per_seq_batch_sample=args.per_seq_batch_sample,
    data_is_raw=args.data_is_raw
)

if is_master:
    ## wandb setting
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name=f'{args.task}_{args.data_name}_{args.model_name}_hvg{args.n_hvg}_{"fast_" if args.fast_transformer else None}{now}'
    wandb_tags=['Finetune',args.task,'SingleNode' if world_size<=4 else 'MultiNode',
                args.data_name,"fast_attn" if args.fast_transformer else 'nomal_attn',]
    run = wandb.init(
        config=hyperparameter_defaults,
        job_type=args.task,
        project="scGPT",
        name=wandb_name,
        tags=wandb_tags,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    print(hyperparameter_defaults)

# set_seed(args.seed)
# %%
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = args.mask_ratio
n_input_bins = args.n_bins
mask_value = -1
pad_value = -2


n_hvg = args.n_hvg if args.n_hvg!=-1 else False # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = args.per_seq_batch_sample
DSBN = args.DSBN  # Domain-spec batchnorm
explicit_zero_prob = args.explicit_zero_prob  # whether explicit bernoulli for zeros


# %%
if is_master:
    save_dir=os.path.join(args.save_dir,args.task,args.data_name,args.model_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

################## Debug Area ##################
# %% [markdown]
debug=False
# ## Loading and preparing data
data_path=os.path.join(args.data_path,args.task,args.data_name)
if args.data_name == "PBMC10K":
    adata = scvi.data.pbmc_dataset(data_path)  # 11990 × 3346
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True
if args.data_name =='Pro397':
    adata=read_h5ad(os.path.join(data_path,'PRO000000397_EPT000000444_SCT000000409.h5ad'))
    ori_batch_col = 'sample_name'
    ##remove the batch which cellnum<=1
    value_counts=adata.obs[ori_batch_col].value_counts()
    indices_of_duplicates = value_counts[value_counts > 1].index
    adata=adata[adata.obs[ori_batch_col].isin(indices_of_duplicates),:]

    adata.obs['celltype']=adata.obs['celltype'].astype("category")
    adata.var = adata.var.set_index("Symbol")
    data_is_raw = True
################## Debug Area ##################

# %%
# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels

adata.var["gene_name"] = adata.var.index.tolist()

if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    if is_master:
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

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
    embsize = args.layer_size 
    nhead = args.nhead
    nlayers = args.nlayers  
    d_hid = args.layer_size


# %%
# set up the preprocessor, use the args to configs the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=3,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if args.data_is_raw else "cell_ranger",
    binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch" if args.data_name != "heart_cell" else None)

# %%
if per_seq_batch_sample:
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# %% [markdown]
# ## Tokenize input

# %%
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

# %%
if args.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

# %%
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
if is_master:
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

# %%
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    if is_master:
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


# %% [markdown]
# # Create and finetune scGPT

# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
wandb.watch(model)


criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
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
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)# masked-> -1
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=args.GEPC,
                ECS=args.ecs_thres > 0,
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
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
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if args.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + args.dab_weight * loss_dab
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

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if args.GEPC else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if args.GEPC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            if is_master:
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if args.GEPC else "")
                )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()


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
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + args.dab_weight * total_dab)
            / total_num,
            "epoch": epoch,
        },
    )

    return total_loss / total_num, total_error / total_num


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].A
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        if is_master:
            logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=args.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                time_step=0,
                return_np=True,
            )
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


# %%
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=args.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=args.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

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
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            # eval on testdata
            results = eval_testdata(
                best_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata,
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

    scheduler.step()


# %%
# save the best model
if is_master:
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
