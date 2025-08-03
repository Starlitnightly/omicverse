#!/bin/bash

##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

##Config nnodes node_rank master_addr
NNODES=$1
NODE_RANK=$2
MASTER_IP=$3
#NPROC=$4

##Start trochrun
torchrun --nproc_per_node=4 --master_port=2222 --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} main.py \
  --lmdb_path /home/share/huada/home/qiuping1/workspace/llm/scbert/data/Zheng68K.h5ad \
  --ckpt_dir /home/share/huada/home/qiuping1/workspace/llm/.pycharm_remote/st_performer/ckpt/panglao \
  --gene2vec_file /home/share/huada/home/qiuping1/workspace/llm/.pycharm_remote/st_performer/data/gene2vec_16906.npy \
  --gene_vocab_file /home/share/huada/home/qiuping1/workspace/llm/.pycharm_remote/st_performer/data/panglao_gene_vocab.txt \
  --organ_vocab_file /home/share/huada/home/qiuping1/workspace/llm/.pycharm_remote/st_performer/data/panglao_organ_vocab.txt \
  --model_name st_performer_panglao_ft_zheng68k \
  --pretrained_model /home/share/huada/home/qiuping1/workspace/llm/.pycharm_remote/st_performer/ckpt/panglao/st_performer_25.pth \
  --is_exp_emb --finetune  --distributed

