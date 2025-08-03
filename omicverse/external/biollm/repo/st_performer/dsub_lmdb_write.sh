#!/bin/bash
#DSUB -n lmdb_write_organs_4w
#DSUB -N 1
#DSUB -A huadjyin.team_s
#DSUB -R "cpu=16;mem=100000"
#DSUB -oo logs/lmdb_write_organs_6w.out
#DSUB -eo logs/lmdb_write_organs_6w.err

##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

HOST=`hostname`
echo 'run node: '$HOST

source ~/init_env/torch2.sh
JOB_PATH="/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/st_performer/data_parse"

cd ${JOB_PATH};python lmdb_handler.py 0


