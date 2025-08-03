#!/bin/bash
#DSUB -n st_performer_scPretrain_panglao
#DSUB -N 3
#DSUB -A root.huada
#DSUB -R "cpu=128;gpu=4;mem=64000"
#DSUB -oo logs/st_performer_scPretrain_panglao.out
#DSUB -eo logs/st_performer_scPretrain_panglao.err
##DSUB -pn "gongchun002-agent-6 gongchun002-agent-9"
#DSUB -pn "gongchun002-agent-8 gongchun002-agent-9 gongchun002-agent-7"
NNODES=3
NUM_GPU_PER_NODE=4

HOST=`hostname`
HOSTFILE="hostnames.log"
flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_IP=`head -n 1 ${HOSTFILE}`
HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=${HOST_RANK}-1

###Set Start Path
JOB_PATH="/home/share/huada/home/qiuping1/workspace/llm/.pycharm_remote/st_performer"

echo "master ip: "${MASTER_IP}
echo "node rank: "${NODE_RANK}

cd ${JOB_PATH};/usr/bin/bash ./run_sc_pretrain.sh ${NNODES} ${NODE_RANK} ${MASTER_IP}
