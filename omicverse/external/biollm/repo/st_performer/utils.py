#!/usr/bin/env python3
# coding: utf-8
"""
@file: utils.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/27  create file.
"""
import torch
import time
import os
import json
import lmdb
from scipy.sparse import issparse
from torchsummary import summary
from biollm.biollm.repo.st_performer.log_manager import LogManager
import numpy as np
import munch
import toml
from functools import reduce
import math


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)  # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)  # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),
                                                                                                        mask.size(
                                                                                                            -1)).to(
        device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())  # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]  # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask,
                                                                   -1e9)  # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)  # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)  # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)  # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)  # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()  # the final mask, True is mask


def data_mask(data, mask_prob, replace_prob, num_tokens, random_token_prob, mask_token_id,
              pad_token_id, mask_ignore_token_ids):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)  # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)  # get the True/False mask matrix
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)  # get the mask matrix of random token replace
        random_tokens = torch.randint(0, num_tokens, data.shape,
                                      device=data.device)  # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens,
                                          mask_ignore_token_ids)  # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask  # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)  # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]  # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)  # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob,
                                            mask_token_id)  # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)  # the label of masked tokens
    return masked_input, labels


def load_config(config_file):
    return munch.munchify(toml.load(config_file))


def save_config(config, config_file):
    with open(config_file, 'w') as f:
        toml.dump(config, f)


def is_mask_tokens(x, ignore_mask_tokens):
    ignore_mask_tokens = np.array(ignore_mask_tokens)
    no_mask = np.isin(x, ignore_mask_tokens)
    return ~no_mask


def mask_tokens(x, ignore_mask_tokens, mask_token_id, pad_token_id, mask_prob=0.15, keep_mask_pro=0.8,
                random_replace_mask_pro=0.1, token_nums=None):
    if len(ignore_mask_tokens) > 0:
        mask_tokens = is_mask_tokens(x, ignore_mask_tokens)
        mask_index = np.nonzero(mask_tokens)[0]
    else:
        mask_index = np.arange(x.shape[0])
    mask_num = np.int32(np.ceil(len(mask_index) * mask_prob * keep_mask_pro))
    keep_mask_index = np.random.choice(mask_index, mask_num, replace=False)
    mask_x = np.copy(x)
    mask_x[keep_mask_index] = mask_token_id
    labels = np.full_like(x, pad_token_id)
    labels[keep_mask_index] = x[keep_mask_index]
    if random_replace_mask_pro:
        assert token_nums is not None, "error: token_nums must be set if random_replace_mask_pro > 0."
        token_ids = np.arange(token_nums)
        replace_tokens = mask_tokens
        replace_tokens[keep_mask_index] = False
        random_replace_num = np.ceil(len(mask_index) * mask_prob * random_replace_mask_pro).astype(np.int32)
        random_replace_index = np.random.choice(np.nonzero(replace_tokens)[0], random_replace_num, replace=False)
        labels[random_replace_index] = x[random_replace_index]
        random_token_ids = np.random.choice(token_ids, random_replace_num, replace=True)
        mask_x[random_replace_index] = random_token_ids
    return mask_x, labels


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def get_logger(log_path=None):
    logger = LogManager(log_path)
    return logger.logger


def timeit(method):
    def timed(*args, **kw):
        _args = args[0].args

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if _args.distributed:
            if _args.local_rank == 0:
                print('Function Time: {}\t>\t{:.0f} min {:.0f} sec'.format(method.__name__, (te - ts) // 60,
                                                                           (te - ts) % 60))
        else:
            print(
                'Function Time: {}\t>\t{:.0f} min {:.0f} sec'.format(method.__name__, (te - ts) // 60, (te - ts) % 60))

        return result

    return timed


def dist_cat_tensors(tensor):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def get_reduced(tensor, device, dest_device):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / torch.distributed.get_world_size()
    return tensor_mean


def make_dir(ckpt_folder):
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)


def save_ckpt(epoch, model, optimizer, scheduler, model_name, ckpt_folder):
    """
    保存模型checkpoint
    """
    make_dir(ckpt_folder)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },
        f'{ckpt_folder}/{model_name}_{epoch}.pth'
    )
    print('save dir: ', f'{ckpt_folder}/{model_name}_{epoch}.pth')


def save_model(model, model_name, ckpt_folder):
    """
    保存模型checkpoint
    """
    make_dir(ckpt_folder)
    torch.save(
        {
            'model_state_dict': model.module.state_dict()
        },
        f'{ckpt_folder}/{model_name}.pth'
    )


def save_best_ckpt(epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder):
    """
    保存模型checkpoint
    """
    make_dir(ckpt_folder)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses,
        },
        f'{ckpt_folder}/{model_name}_best.pth'
    )


def get_gene_vocab(id_gene_json, output):
    with open(id_gene_json, 'r') as f, open(output, 'w') as w:
        id_gene = json.load(f)
        for i in range(len(id_gene)):
            w.write(id_gene[str(i)] + '\n')


def data2lmdb(adata, lmdb_path, index, finetune=True, cell_label=None, write_frequency=10000, celltype_count=None):
    print("Generate LMDB to %s" % lmdb_path)
    train_db = lmdb.open(lmdb_path, map_size=int(50000e9))
    txn = train_db.begin(write=True)
    length = index
    data_id = str(int(time.time()))
    data = adata.X
    meta = {'organ': 'brain', 'gene_list': adata.var['gene_id'].tolist()}
    if finetune:
        celltype = {} if cell_label is None else cell_label
        for i in range(data.shape[0]):
            x = data[i].A.tolist() if issparse(data) else [data[i].tolist()]
            celltype_count[adata.obs['ontology_name'][i]] += 1
            if adata.obs['ontology_name'][i] in celltype:
                celltype_id = celltype[adata.obs['ontology_name'][i]]
            else:
                celltype_id = len(celltype)
                celltype[adata.obs['ontology_name'][i]] = celltype_id
            value = {'x': x, 'celltype': celltype_id, 'meta': 'm' + str(data_id)}
            txn.put(str(length).encode(), json.dumps(value).encode())
            length += 1
            if (length + 1) % write_frequency == 0:
                print('write: ', length)
                txn.commit()
                txn = train_db.begin(write=True)
        txn.put(('m' + str(data_id)).encode(), json.dumps(meta).encode())
    print(length)
    # finish iterating through dataset
    txn.commit()
    with train_db.begin(write=True) as txn:
        txn.put(b'__len__', str(length).encode())
    print("Flushing database ...")
    train_db.sync()
    train_db.close()
    return cell_label, celltype_count, length


def cal_model_params(model, input_size):
    summary(model, input_size)


if __name__ == '__main__':
    get_gene_vocab('/home/share/huada/home/zuolulu/00.project/03.st_st_performer/0.data/gene_vocab.json',
                   './test/gene_vocab.txt')
