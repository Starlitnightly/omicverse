#!/usr/bin/env python3
# coding: utf-8
"""
@file: fit_bugs_lmdb.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/01/29  create file.
"""
import lmdb
from st_performer.utils import get_logger

logger = get_logger()

env = lmdb.open('/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/cellxgene/all.db',
                map_size=int(10000e9))
env1 = lmdb.open('/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/cellxgene/all.db.bak')
i = 0
txn = env.begin(write=True)
with env1.begin() as txn1:
    cursor = txn1.cursor()
    for key, value in cursor:
        if not key.decode().startswith('g'):
            txn.put(str(i).encode(), value)
            i += 1
        else:
            txn.put(key, value)
            logger.info(f'key: {key.decode()}')
        if i % 100000 == 0:
            logger.info('put len: {}'.format(i))
            txn.commit()
            env.close()
            env = lmdb.open(
                '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/cellxgene/all.db',
                map_size=int(10000e9))
            txn = env.begin(write=True)
    txn.put(b'__len__', str(i).encode())
    logger.info('db len: {}'.format(i))
    txn.commit()
env.close()
env1.close()
