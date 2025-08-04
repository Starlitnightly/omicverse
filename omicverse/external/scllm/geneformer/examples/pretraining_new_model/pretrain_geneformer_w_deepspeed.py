#!/usr/bin/env python
# coding: utf-8

# run with:
# deepspeed --num_gpus=12 --num_nodes=3 pretrain_geneformer_w_deepspeed.py --deepspeed ds_config.json

import datetime

# imports
import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments

from geneformer import GeneformerPretrainer

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# set local time/directories
timezone = pytz.timezone("US/Eastern")
rootdir = "/parent_ouput_directory"

# set model parameters
# model type
model_type = "bert"
# max input size
max_input_size = 2**11  # 2048
# number of layers
num_layers = 6
# number of attention heads
num_attn_heads = 4
# number of embedding dimensions
num_embed_dim = 256
# intermediate size
intermed_size = num_embed_dim * 2
# activation function
activ_fn = "relu"
# initializer range, layer norm, dropout
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.02
hidden_dropout_prob = 0.02


# set training parameters
# total number of examples in Genecorpus-30M after QC filtering:
num_examples = 27_406_208
# number gpus
num_gpus = 12
# batch size for training and eval
geneformer_batch_size = 12
# max learning rate
max_lr = 1e-3
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 10_000
# number of epochs
epochs = 3
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.001


# output directories
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"{datestamp}_geneformer_30M_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")


# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")


# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)


# load gene_ensembl_id:token dictionary (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/token_dictionary.pkl)
with open("token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# model configuration
config = {
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": token_dictionary.get("<pad>"),
    "vocab_size": len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
}

config = BertConfig(**config)
model = BertForMaskedLM(config)
model = model.train()

# define the training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": False,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    "save_steps": np.floor(
        num_examples / geneformer_batch_size / 8
    ),  # 8 saves per epoch
    "logging_steps": 1000,
    "output_dir": training_output_dir,
    "logging_dir": logging_dir,
}
training_args = TrainingArguments(**training_args)

print("Starting training.")

# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
    train_dataset=load_from_disk("genecorpus_30M_2048.dataset"),
    # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
    example_lengths_file="genecorpus_30M_2048_lengths.pkl",
    token_dictionary=token_dictionary,
)

# train
trainer.train()

# save model
trainer.save_model(model_output_dir)
