#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_lmdb.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/13  create file.
"""
import numpy as np
import lmdb
import json


def get_lmdb_data(dpath):
    db = lmdb.open(dpath)
    with db.begin() as txn:
        print(txn.get(b'__len__'))
        print(txn.get(b'__len__').decode("utf-8"))
        print(int(txn.get(b'__len__').decode("utf-8")))
        for i in range(1, 11):
            value = txn.get(u'{}'.format(i).encode('ascii'))

            if value:
                value = np.frombuffer(value)
                print(np.min(value), np.max(value))


def test_lmdb_put_dict(dpath):
    db = lmdb.open(dpath)
    x = np.random.randint(0, 100, (1000, 50))
    genes = ['g' + str(i) for i in range(50)]
    organ = 'brain'
    with db.begin(write=True) as txn:
        data = {'x': [1, 2, 4], 'gene': 'a'}
        a = '1'.encode()
        b = str('g1').encode()
        txn.put(a, json.dumps(data).encode())
        txn.put(b, np.array([2, 3, 4]))

        print(json.loads(txn.get(a)))
        print(txn.get(b))
        print(np.fromstring(txn.get(b), dtype=int))
    db.close()


def make_pretrain_data(dpath):
    db = lmdb.open(dpath)
    with open('./gene_vocab.txt', 'w') as w:
        for i in range(50):
            w.write('g' + str(i) + '\n')
    organ = 'brain'
    with db.begin(write=True) as txn:
        for i in range(1000):
            index_meta = list(range(0, 50))
            np.random.shuffle(index_meta)
            label = int(np.random.choice([0, 1]))
            meta = {'organ': organ, 'gene_x': list(index_meta)}
            # meta = {'organ': organ, 'gene_x': list(index_meta), 'label': -}
            meta_key = 'm' + str(i)
            txn.put(meta_key.encode(), json.dumps(meta).encode())
            x_fun = lambda _: [int(j) for j in np.random.randint(0, 10, 50)]
            data = {'x': x_fun(1), 'meta': meta_key}
            # data = {'x': [x_fun(1), x_fun(1)], 'meta': meta_key}
            txn.put(str(i).encode(), json.dumps(data).encode())
            print(meta_key)
            print(meta)
            print(i)
            print(data)
            print('get data:')
            print(json.loads(txn.get(meta_key.encode())))
            print(json.loads(txn.get(str(i).encode())))
        txn.put(b'__len__', str(1000).encode())
        print(txn.get(b'__len__').decode())
    db.close()


make_pretrain_data('./test_sc.db')
