set -xe

device_id=6 # which device to run the program, for multi-gpus, set params like device_id=0,2,5,7. [Note that] the device index in python refers to 0,1,2,3 respectively.

# params
data_dir=./data/
data_name=gse90546_k562_63587_19264_10k_log1p
split=simulation
result_dir=./results
seed=1
epochs=15
batch_size=30 #30
test_batch_size=4
hidden_size=128 #128
train_gene_set_size=0.75
lr=0.001 #1e-3

workdir=./
cd $workdir

export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

result_dir=${workdir}/results/${data_name}/${train_gene_set_size}/split_${split}_seed_${seed}_hidden_${hidden_size}_epochs_${epochs}_batch_${batch_size}_lr_${lr}/${TIMESTAMP}/

mkdir -p ${result_dir}

CUDA_VISIBLE_DEVICES=${device_id} python -u train.py \
    --data_dir=${data_dir} \
    --data_name=${data_name} \
    --seed=${seed} \
    --result_dir=${result_dir} \
    --seed=${seed} \
    --epochs=${epochs} \
    --batch_size=${batch_size} \
    --test_batch_size=${test_batch_size} \
    --hidden_size=${hidden_size} \
    --lr=${lr} > ${result_dir}/train.log 2>&1