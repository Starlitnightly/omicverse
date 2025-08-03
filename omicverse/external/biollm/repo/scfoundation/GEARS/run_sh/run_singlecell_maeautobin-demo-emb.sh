set -xe

device_id=7 # which device to run the program, for multi-gpus, set params like device_id=0,2,5,7. [Note that] the device index in python refers to 0,1,2,3 respectively.


# params
data_dir=./data/
data_name=demo
split=simulation
result_dir=./results
seed=1
epochs=1
batch_size=2
accumulation_steps=1
test_batch_size=2
hidden_size=512
train_gene_set_size=0.75
mode=v1
highres=0 # 0
lr=0.0002 #1e-3

model_type=API
bin_set=autobin_resolution_append #autobin_resolution, bin_2, bin_3, no_bin
finetune_method='API' # [None,finetune, 'frozen', 'finetune_lr_1'])


workdir=./
cd $workdir

export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

result_dir=${workdir}/results/${data_name}/${train_gene_set_size}/50m-0.1B_split_${split}_seed_${seed}_hidden_${hidden_size}_bin_${bin_set}_singlecell_${model_type}_finetune_${finetune_method}_epochs_${epochs}_batch_${batch_size}_accmu_${accumulation_steps}_mode_${mode}_highres_${highres}_lr_${lr}/${TIMESTAMP}/

mkdir -p ${result_dir}

CUDA_VISIBLE_DEVICES=${device_id} /usr/bin/python -u train.py \
    --data_dir=${data_dir} \
    --data_name=${data_name} \
    --seed=${seed} \
    --result_dir=${result_dir} \
    --seed=${seed} \
    --epochs=${epochs} \
    --batch_size=${batch_size} \
    --test_batch_size=${test_batch_size} \
    --hidden_size=${hidden_size} \
    --bin_set=${bin_set} \
    --model_type=${model_type} \
    --finetune_method=${finetune_method} \
    --singlecell_model_path=${singlecell_model_path} \
    --mode=${mode} \
    --highres=${highres} \
    --accumulation_steps=${accumulation_steps} \
    --lr=${lr} > ${result_dir}/train.log 2>&1

# --singlecell_model_path=${singlecell_model_path}