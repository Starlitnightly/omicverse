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
#HOSTFILE=$2
#HOST=`hostname`
#flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
#MASTER_IP=`head -n 1 ${HOSTFILE}`
#HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
#let NODE_RANK=HOST_RANK-1
NODE_RANK=$2
MASTER_IP=$3
NPROC=$4

##Start trochrun
torchrun --nproc_per_node=${NPROC} --master_port=2222 --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} main.py \
	--lmdb_path /home/share/huada/home/zuolulu/00.project/03.st_st_performer/0.data/lmdb_sc_inner/train.db  --gene_vocab_file ./test/gene_vocab_inner.txt --organ_vocab_file ./test/organ_vocab.txt --organ_num 1 --pretrain  --distributed --max_seq_len 3114 --batch_size 8

