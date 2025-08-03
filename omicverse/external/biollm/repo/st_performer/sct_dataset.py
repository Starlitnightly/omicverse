#!/usr/bin/env python3
# coding: utf-8
"""
@file: st_dataset.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/27  create file.
"""
import numpy as np
from torch.utils.data import Dataset
import lmdb
import json
import torch
from utils import mask_tokens


class SctDataset(Dataset):
    def __init__(self, lmdb_path, max_seq_len, is_st, gene_tokenizer, organ_tokenizer, mask_prob=0.15, finetune=False):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.is_st = is_st
        self.ignore_mask_tokens = [gene_tokenizer.pad_token_id, gene_tokenizer.cls_token_id, gene_tokenizer.unk_token_id]
        self.mask_prob = mask_prob
        self.gene_tokenizer = gene_tokenizer
        self.organ_tokenizer = organ_tokenizer
        self.max_seq_len = max_seq_len
        self.finetune = finetune

        self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.length = self.get_length()
        self.env = None
        self.txn = None

    def get_length(self):
        # length = int(self.txn.get(b'__len__').decode())
        length = 10000
        return length

    def get_lmdb_data(self, index):
        value = json.loads(self.txn.get(str(index).encode()))
        exp_x = value['x']
        meta_info = json.loads(self.txn.get(value['meta'].encode()))
        gene_x = meta_info['gene_list']
        organ = meta_info['organ']
        if self.finetune:
            celltype = value['celltype']
            return gene_x, exp_x, organ, celltype
        else:
            neighbor_labels = value['labels'] if self.is_st else np.array([])
            return gene_x, exp_x, organ, neighbor_labels

    def __getitem__(self, index):
        if self.txn is None:
            self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
            self.txn = self.env.begin(write=False)
        x = self.get_lmdb_data(index)
        celltype, neighbor_labels = None, None
        if self.finetune:
            gene_x, exp_x, organ, celltype = x
        else:
            gene_x, exp_x, organ, neighbor_labels = x
        # add [CLS] token at the head
        exp_x = np.hstack((np.array([[0], [0]]), exp_x)) if self.is_st else np.hstack((np.array([[0]]), exp_x))
        gene_x = np.hstack((np.array([self.gene_tokenizer.cls_token_id]), gene_x))
        gene_x = np.array(self.gene_tokenizer.padding(gene_x, self.max_seq_len))
        if exp_x.shape[-1] >= self.max_seq_len:
            exp_x = exp_x[:, 0: self.max_seq_len]
        else:
            exp_x = np.hstack((exp_x, np.zeros((2, self.max_seq_len-exp_x.shape[-1])))) if self.is_st else np.hstack((exp_x, np.zeros((1, self.max_seq_len-exp_x.shape[-1]))))
        organ = self.organ_tokenizer.convert_token_to_id(organ)
        if self.finetune:
            return gene_x, exp_x[0], organ, celltype
        else:
            if self.is_st:
                mask_exp_x = np.zeros((2, self.max_seq_len))
                labels = np.zeros((2, self.max_seq_len))
                for i in range(exp_x.shape[0]):
                    res = self.mask_exp_x(exp_x[i], mask_prob=self.mask_prob, pad_value=self.gene_tokenizer.pad_token_id)
                    mask_exp_x[i] = res[0]
                    labels[i] = res[1]
            else:
                mask_exp_x, labels = self.mask_exp_x(exp_x[0], mask_prob=self.mask_prob, pad_value=self.gene_tokenizer.pad_token_id)
            return gene_x, mask_exp_x, organ, labels, neighbor_labels

    def __len__(self):
        return self.length

    def mask_exp_x(self, x, pad_value, mask_value=0, mask_prob=0.8, mask_zero=False):
        if mask_zero:
            mask_num = np.int32(np.ceil(x.shape[0] * mask_prob))
            keep_mask_index = np.random.randint(0, x.shape[0], mask_num)
        else:
            mask_index = np.nonzero(x)[0]
            mask_num = np.int32(np.ceil(len(mask_index) * mask_prob))
            keep_mask_index = np.random.choice(mask_index, mask_num, replace=False)
        mask_x = np.copy(x)
        mask_x[keep_mask_index] = mask_value
        labels = np.full_like(x, pad_value)
        labels[keep_mask_index] = x[keep_mask_index]
        return mask_x, labels


class ScDataset(Dataset):
    """
    origin scbert dataset.
    """
    def __init__(self, lmdb_path, bin_num, ignore_mask_tokens, mask_token_id, pad_token_id, mask_prob=0.15,
                 keep_mask_pro=0.8, random_replace_mask_pro=0.1):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.bin_num = bin_num
        self.ignore_mask_tokens = ignore_mask_tokens
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_prob = mask_prob
        self.keep_mask_pro = keep_mask_pro
        self.random_replace_pro = random_replace_mask_pro

        self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.length = self.get_length()
        self.env = None
        self.txn = None

    def get_length(self):
        length = int(self.txn.get(b'__len__').decode("utf-8"))
        return length

    def get_lmdb_data(self, index):
        value = self.txn.get(u'{}'.format(index).encode('ascii'))
        value = np.frombuffer(value)
        return value

    def __getitem__(self, index):
        if self.txn is None:
            self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
            self.txn = self.env.begin(write=False)
        x = self.get_lmdb_data(index)
        x = x.copy()
        x[x > self.bin_num] = self.bin_num
        full_seq = np.floor(x).astype(np.int32)
        # add [CLS] token at the end
        full_seq = np.hstack((full_seq, np.array([0])))
        mask_x, labels = mask_tokens(full_seq, self.ignore_mask_tokens, self.mask_token_id, self.pad_token_id,
                                   self.mask_prob, self.keep_mask_pro, self.random_replace_pro, self.bin_num + 1)
        gene_index = np.arange(mask_x.shape[0])
        return gene_index, mask_x, labels

    def __len__(self):
        return self.length

    def exp_bin(self, x):
        nonzero_x = x[np.nonzero(x)[0]]
        bins = np.linspace(nonzero_x.min(), nonzero_x.max() + 0.0001, self.bin_num + 1)
        indices = np.digitize(x, bins)
        return indices


class ScDatasetLocal(Dataset):
    def __init__(self, data, label, bin_num, gene_index, use_gene_ids):
        super().__init__()
        self.data = data
        self.label = label
        self.bin_num = bin_num
        self.gene_index = torch.tensor(gene_index)
        self.use_gene_ids = use_gene_ids

    def __getitem__(self, index):
        # rand_start = np.random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > self.bin_num] = self.bin_num
        full_seq = torch.from_numpy(full_seq).long()
        # full_seq = torch.cat((full_seq, torch.tensor([0])))
        seq_label = self.label[index]
        # gene_index = np.arange(full_seq.shape[0])
        gene_x, express_x = self.make_train_data(self.gene_index, full_seq, self.use_gene_ids)
        return gene_x, express_x, seq_label

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def make_train_data(gene_ids, express_x, use_genes_ids):
        use_genes_ids = np.array(use_genes_ids)
        gene_ids, unique_index = np.unique(gene_ids, return_index=True)
        express_x = np.array(express_x)[unique_index]
        indices = np.where(np.isin(gene_ids, use_genes_ids))[0]
        gene_subs = gene_ids[indices]
        express_x = express_x[indices]
        supl_genes = np.setdiff1d(use_genes_ids, gene_subs)
        supl_express_x = np.zeros_like(supl_genes)
        gene_x = np.concatenate([gene_subs, supl_genes])
        express_x = np.concatenate([express_x, supl_express_x])
        return gene_x, express_x


class ScDatasetLmdb(Dataset):
    """
    origin scbert dataset.
    """
    def __init__(self, lmdb_path, use_genes_ids):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.length = self.get_length()
        self.env = None
        self.txn = None
        self.use_genes_ids = use_genes_ids

    def get_length(self):
        length = int(self.txn.get(b'__len__').decode("utf-8"))
        return length

    def get_lmdb_data(self, index):
        """
        get lmdb data
        :param index: the index of cell in lmdb
        :return: {"express_x": list, "organ": int, "sequence": int, "disease": int, "celltype": int}
        """
        res = json.loads(self.txn.get(str(index).encode()))
        gene_id = np.frombuffer(self.txn.get(res['gene_index'].encode()), dtype=np.int64)
        res['gene_id'] = gene_id
        res['batch_id'] = int(res['gene_index'].strip('g'))
        res.pop('gene_index')
        return res

    def __getitem__(self, index):
        if self.txn is None:
            self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
            self.txn = self.env.begin(write=False)
        result = self.get_lmdb_data(index)
        result = self.make_train_data(result, self.use_genes_ids)
        return result

    def __len__(self):
        return self.length

    @staticmethod
    def make_train_data(result, use_genes_ids):
        use_genes_ids = np.array(use_genes_ids)
        gene_ids, unique_index = np.unique(result['gene_id'], return_index=True)
        express_x = np.array(result['express_x'])[unique_index]
        indices = np.where(np.isin(gene_ids, use_genes_ids))[0]
        gene_subs = gene_ids[indices]
        express_x = express_x[indices]
        supl_genes = np.setdiff1d(use_genes_ids, gene_subs)
        supl_express_x = np.zeros_like(supl_genes)
        gene_x = np.concatenate([gene_subs, supl_genes])
        express_x = np.concatenate([express_x, supl_express_x])
        result['gene_id'] = gene_x
        result['express_x'] = express_x
        return result
