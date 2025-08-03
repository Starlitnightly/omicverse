# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:gene_regulatory_network.py
# @Software:PyCharm
# @Created Time:2024/2/1 5:44 PM
import json
import os
from pathlib import Path
import shutil
import sys
import warnings
# from . import asyn
import torch
import scanpy as sc
import wandb

sys.path.append("/home/share/huada/home/jiangwenjian/proj/scGPT/")
import scgpt as scg
from scgpt.model import TransformerModel
from scLLM_utils.dataset import Load_Data,filter_gene
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
import argparse
from finetune_utils import seed_all
import torch.distributed as dist
import datetime
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# most important
parser.add_argument("--task", type=str, default='Cell_annotation',choices=['GRN_inference','Cell_annotation','Integration'], help='Name of task')#
parser.add_argument("--data_name", type=str, choices=['adamson'],default='adamson', help='Name of dataset')#
parser.add_argument("--data_is_raw", type=bool, default=True, help='whether the data is raw')#
parser.add_argument("--model_name", type=str, default='Debug', help='Finetuned model name.')#
parser.add_argument("--distributed", type=bool, default=False, help='debug mode, single gpu device')#
parser.add_argument("--single_gpu", type=bool, default=False, help='single gpu device, but not debug mode')#
parser.add_argument("--do_train", type=bool, default=True, help='Train or inference')#
# general hyperparameters
parser.add_argument("--epochs", type=int, default=50, help='Number of epochs.')#
parser.add_argument("--seed", type=int, default=1927, help='Random seed.')#
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate.')
parser.add_argument("--batch_size", type=int, default=64, help='Number of batch size.')
parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--lr_ADV", type=float, default=1e-3, help='learning rate for discriminator, used when ADV is True.')

parser.add_argument("--log_interval",type=int, default=100,help='interval of log.')
parser.add_argument("--forcing_wandb", type=bool, default=False, help='whether open wandb by force')#
parser.add_argument("--save_eval_interval",type=int, default=5, help='interval of evaluation')#
parser.add_argument("--schedule_ratio",type=float, default=0.9, help='ratio of epochs for learning rate schedule')#
parser.add_argument("--amp",type=bool, default=True,help='Automatic Mixed Precision')#
# path-related
parser.add_argument("--data_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/data', help='Path of data for finetune.')#
parser.add_argument("--load_model", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/saves/Pretraining/panglao/scgpt/fast_attn_3', help='Path of pretrained model.')
parser.add_argument("--save_dir", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/saves', help='Directory of checkpoint and result to save.')
parser.add_argument("--vocab_file", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/graph/vocab.json', help='Path of vocab, available if load_model is None')
parser.add_argument("--gene_array_file", type=str, default='/home/share/huada/home/qiuping1/workspace/llm/data/train_data/vocab/tmp/panglao_gene_ids.pk', help='Path of vocab, available if load_model is None')
parser.add_argument("--graph_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scGPT/graph', help='Path of graph')#

# if load model, batch_size, layer_size, nlayers, nhead will be ignored
parser.add_argument("--fast_transformer", type=bool, default=True, help='Using fast-attn or not')#
parser.add_argument("--layer_size", type=int, default=128, help='Size of embedding.')#
parser.add_argument("--d_hid", type=int, default=512, help='Size of hidden state.')#
parser.add_argument("--nheads", type=int, default=8, help='number of attention head')#
parser.add_argument("--nlayers", type=int, default=12, help='number of transformer layers')#
parser.add_argument("--mask_ratio", type=float, default=0.0, help='ratio of masked token.')#
parser.add_argument("--pre_norm", type=bool, default=False, help='normalize previously')#
parser.add_argument("--freeze", type=bool, default=False, help='freeze')#
parser.add_argument("--DSBN", type=bool, default=False, help='Domain-spec batchnorm')#
parser.add_argument("--ecs_thres", type=float, default=0.0, help='Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable.')#
parser.add_argument("--dab_weight", type=float, default=0.0, help='weight of dab.')#
parser.add_argument("--input_emb_style", type=str, default='continuous',choices=['continuous','category','scaling'], help='the style of input emb')#
parser.add_argument("--cell_emb_style", type=str, default='avg-pool',choices=['cls','avg-pool','w-pol'], help='method for generating cell emb')#
parser.add_argument("--mvc_decoder_style", type=str, default='inner product',choices=['inner product','concat query','sum query'], help='architecture style of the decoder')#


# data preprocessing related hyper-params
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins.')#
parser.add_argument("--per_seq_batch_sample", type=bool, default=False, help='whether sort the adata by batch_id')#
parser.add_argument("--include_zero_gene", type=bool, default=False, help='whether include gene with zero expression value')#
parser.add_argument("--input_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='input representation')#
parser.add_argument("--output_style", type=str, default='binned',choices=['binned','normed_raw','log1p'], help='output representation')#



#################Step1: Specify hyper-parameter setup for cell-type annotation task###################
args = parser.parse_args()
sc.set_figure_params(figsize=(6, 6))
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
    elif not args.forcing_wandb: #debug mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["CUDA_VISIBLE_DEVICES"]='2'
        local_rank = 0
    else:# run by terminal
        debug=False
        local_rank = 0
    is_master=True
    world_size=1
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)
print(f'current devices: {device}')
if is_master:
    ## wandb setting
    model_ckpt_name=[name for name in ['whole_human','panglao'] if name in args.load_model][0]
    assert isinstance(model_ckpt_name,str)
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    wandb_name=f'{args.task}_{args.data_name}_{args.model_name}_(ckpt-{model_ckpt_name})' \
               f'{"_MVC" if args.MVC else ""}{"_ADV" if args.ADV else ""}{"_CCE" if args.CCE else ""}{"_MLM" if args.MLM else ""}{"_CLS" if args.CLS else ""}' \
               f'_{"fast_" if args.fast_transformer else ""}{args.cell_emb_style}_{"wZero_" if args.include_zero_gene else ""}{now}'
    wandb_tags=['Finetune',args.task,'SingleNode' if world_size<=4 else 'MultiNode',f'ckpt-{model_ckpt_name}',
                args.data_name,"fast_attn" if args.fast_transformer else 'nomal_attn',
                "MVC" if args.MVC else "w/o MVC","ADV" if args.ADV else "w/o ADV",
                "CLS" if args.CLS else "w/o CLS","CCE" if args.CCE else "w/o CCE","MLM" if args.MLM else "w/o MLM",args.cell_emb_style]
    run = wandb.init(
        config=args.__dict__,
        job_type=args.task,
        project="scGPT-CA",
        name=wandb_name,
        tags=wandb_tags,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    print(args.__dict__)

# set_seed(args.seed)


# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token,"<cls>", "<eoc>","<mask>","<unk>"]

include_zero_gene = args.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = args.n_bins

# settings for optimizer
lr = args.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = args.lr_ADV  # learning rate for discriminator, used when ADV is True
batch_size = args.batch_size
eval_batch_size = args.batch_size
epochs = args.epochs
schedule_interval = 1

# settings for the model
fast_transformer = args.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = args.layer_size  # embedding dimension
d_hid = args.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = args.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = args.nhead  # number of heads in nn.MultiheadAttention
dropout = args.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = args.save_eval_interval  # epochs
do_eval_scib_metrics = True


if is_master:
    save_dir=os.path.join(args.save_dir,args.task,args.data_name,args.model_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")


#################Step2: Loading model and vocab###################
if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    shutil.copy(vocab_file, save_dir / "vocab.json")
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
    if is_master:
        logger.info(f"Train model from scratch.")
    embsize = args.d_hid
    nhead = args.nheads
    nlayers = args.nlayers
    d_hid = args.d_hid
    vocab_file=args.vocab_file
vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
gene2idx=vocab.get_stoi()

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    n_input_bins=n_bins,
    use_fast_transformer=fast_transformer,
)
try:
    model.load_state_dict(torch.load(model_file))
    print(f"Loading all model params from {model_file}")
except:
    # only load params that are in the model and match the size
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
model.to(device)

#################Step3: Data loading and preprocessing###################
dataset_name = args.data_name
data_path=os.path.join(args.data_path,args.task,dataset_name)
adata,data_is_raw,ori_batch_col,filter_gene_by_counts=Load_Data(data_path=data_path,args=args)
# remove the genes that didn't appear in vocab
adata,_=filter_gene(vocab=vocab,adata=adata,is_master=is_master,logger=logger)

# set up the preprocessor, use the args to configs the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch")