#!/usr/bin/env python3
# coding: utf-8
"""
@file: tokenizer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/27  create file.
"""
from typing import List
from collections import OrderedDict
import json
import numpy as np
import os


def build_vocab(vocab_file):
    # build vocab and ids_to_organs
    vocab = OrderedDict()
    with open(vocab_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            token = line.strip().split()[0]
            vocab[token] = i
    return vocab


def build_ids_to_tokens(vocab):
    ids_to_tokens = OrderedDict()
    for token, id in vocab.items():
        ids_to_tokens[id] = token
    return ids_to_tokens


def load_vocab(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            vocab = json.load(f)
        return vocab
    return {}


class Tokennizer:
    def __init__(self, vocab_file, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', cls_token='<cls>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.cls_token = cls_token
        if vocab_file is None:
            self.vocab = {}
        elif vocab_file.endswith('txt'):
            self.vocab = build_vocab(vocab_file)
        else:
            self.vocab = load_vocab(vocab_file)
        num = len(self.vocab)
        for i in ['<pad>', '<cls>', '<unk>', '<mask>']:
            if i not in self.vocab:
                self.vocab[i] = num
                num += 1
        self.ids_to_tokens = build_ids_to_tokens(self.vocab)
        self.vocab_json_path = vocab_file.replace('txt', 'json')

    def update_vocab(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)

    def convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) in an id (integer) using the vocab.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_id_to_token(self, id: int) -> str:
        """Convert an id (integer) in a token (str) using the vocab.
        """
        return self.ids_to_tokens.get(id, self.unk_token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens in list of ids using the vocab.
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids in list of tokens using the vocab.
        """
        return [self.convert_id_to_token(id) for id in ids]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size.
        """
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Id of pad_token in the vocab.
        """
        return self.convert_token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Id of unk_token in the vocab.
        """
        return self.convert_token_to_id(self.unk_token)

    @property
    def mask_token_id(self) -> int:
        """Id of mask_token in the vocab.
        """
        return self.convert_token_to_id(self.mask_token)

    @property
    def cls_token_id(self) -> int:
        """Id of mask_token in the vocab.
        """
        return self.convert_token_to_id(self.cls_token)

    def padding(self, x, max_seq_len):
        if len(x) >= max_seq_len:
            return x[0: max_seq_len]
        else:
            if isinstance(x, np.ndarray):
                x = list(x)
            x += [self.pad_token_id] * (max_seq_len - len(x))
            x = np.array(x)
        return x

    def vocab_to_json(self):
        with open(self.vocab_json_path, 'w') as f:
            json.dump(self.vocab, f)


class DiseaseTokenizer(Tokennizer):
    def __init__(self, vocab_file):
        super(DiseaseTokenizer, self).__init__(vocab_file)


class OraganTokenizer(Tokennizer):
    def __init__(self, vocab_file):
        super(OraganTokenizer, self).__init__(vocab_file)


class GeneTokenizer(Tokennizer):
    def __init__(self, vocab_file):
        super(GeneTokenizer, self).__init__(vocab_file=vocab_file)


class ExpBinTokenizer(Tokennizer):
    def __init__(self, vocab_file):
        super(ExpBinTokenizer, self).__init__(vocab_file=vocab_file)
