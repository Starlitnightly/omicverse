#!/usr/bin/env python3
# coding: utf-8
"""
@file: main.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/31  create file.
"""
import json
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.data.dataloader import default_collate
from trainer import Trainer
from sct_dataset import SctDataset, ScDataset, ScDatasetLocal,ScDatasetLmdb
from tokenizer import GeneTokenizer, OraganTokenizer, Tokennizer
import os
from utils import save_ckpt, load_config
import pickle


def make_dataset1(args, gene_tokenizer, organ_tokenizer, device):
    """
    策略一
    """
    train_dataset = SctDataset(args.lmdb_path, args.max_seq_len, args.is_st, gene_tokenizer, organ_tokenizer,
                               args.mask_prob, args.finetune)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.n_workers,
                              collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = None
    if args.do_eval:
        test_dataset = SctDataset(args.test_lmdb_path, args.max_seq_len, args.is_st, gene_tokenizer, organ_tokenizer,
                                  args.mask_prob, args.finetune)
        test_sampler = RandomSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 num_workers=args.n_workers,
                                 collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    return train_loader, test_loader


def make_dataset2(args, device):
    """
    原生scbert
    """
    mask_token_ids = [0, args.bin_num + 1]  # 零值及pad、mask token

    train_dataset = ScDataset(args.lmdb_path, args.bin_num, mask_token_ids, args.bin_num + 1, args.bin_num + 1,
                              args.mask_prob,
                              args.keep_replace_prob, args.random_replace_prob)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.n_workers,
                              collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = None
    if args.do_eval:
        test_dataset = ScDataset(args.test_lmdb_path, args.bin_num, [0, args.bin_num + 1], args.bin_num + 1,
                                 args.bin_num + 1, args.mask_prob,
                                 args.keep_replace_prob, args.random_replace_prob)
        test_sampler = RandomSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 num_workers=args.n_workers,
                                 collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    return train_loader, test_loader


def make_dataset(args, device):
    """
    策略二： 基因选择：marker gene； 初始化gene embedding： gene2vec

    lmdb： gene_id、express_x、organ、disease、batches
    """
    with open(args.use_genes_file, 'rb') as fd:
        use_genes = pickle.load(fd)
    train_dataset = ScDatasetLmdb(args.lmdb_path, use_genes)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.n_workers)
    test_loader = None
    if args.do_eval:
        test_dataset = ScDatasetLmdb(args.test_lmdb_path, use_genes)
        test_sampler = RandomSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 num_workers=args.n_workers,
                                 collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    return train_loader, test_loader


def make_finetune_dataset(args, device):
    import scanpy as sc
    import numpy as np
    import pickle as pkl
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


    data = sc.read_h5ad(args.lmdb_path)
    with open(args.vocab_dir + '/gene_vocab.json', 'r') as f:
        gene_vocab = json.load(f)
    gene_index = [gene_vocab[i] for i in data.var.index.values]
    with open(args.use_genes_file, 'rb') as fd:
        use_genes = pickle.load(fd)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    # store the label dict and label for prediction
    with open(f'./data/label_dict.{args.model_name}', 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open(f'./data/label.{args.model_name}', 'wb') as fp:
        pkl.dump(label, fp)
    label = torch.from_numpy(label)
    data = data.X

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2023)
    for index_train, index_val in sss.split(data, label):
        data_train, label_train = data[index_train], label[index_train]
        data_val, label_val = data[index_val], label[index_val]
        train_dataset = ScDatasetLocal(data_train, label_train, args.bin_num, gene_index, use_genes)
        test_dataset = ScDatasetLocal(data_val, label_val, args.bin_num, gene_index, use_genes)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.n_workers)
    test_sampler = RandomSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                             num_workers=args.n_workers)
    return train_loader, test_loader


def main(args):
    device = 'cuda:2'
    # Setup CUDA, GPU & distributed training
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
        args.local_rank = local_rank
    if args.no_cuda:
        device = torch.device('cpu')

    gene_tokenizer = GeneTokenizer(args.vocab_dir + '/gene_vocab.json')
    organ_tokenizer = OraganTokenizer(args.vocab_dir + '/organ_vocab.json')
    sequence_tokenizer = Tokennizer(args.vocab_dir + '/sequence_vocab.json')
    disease_tokenizer = Tokennizer(args.vocab_dir + '/disease_vocab.json')
    assert args.pretrain != args.finetune, ""
    if args.pretrain:
        train_loader, test_loader = make_dataset(args, device)
    else:
        train_loader, test_loader = make_finetune_dataset(args, device)
    # Build Trainer
    trainer = Trainer(args=args, device=device,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      gene_tokenizer=gene_tokenizer,
                      organ_tokenizer=organ_tokenizer,
                      sequence_tokenizer=sequence_tokenizer,
                      disease_tokenizer=disease_tokenizer)
    for epoch in range(1, args.epochs + 1):
        if args.pretrain:
            trainer.pretrain(epoch)
            if args.do_eval:
                trainer.evaluate(epoch)
        else:
            trainer.finetune(epoch)
            trainer.val_finetune(epoch)
        save_ckpt(epoch, trainer.model, trainer.optimizer, trainer.scheduler, 'st_performer', args.ckpt_dir)


if __name__ == "__main__":
    config_file = sys.argv[1]
    args = load_config(config_file)
    main(args)
