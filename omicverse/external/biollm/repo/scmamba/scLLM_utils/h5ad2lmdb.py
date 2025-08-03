# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:h5ad2lmdb.py
# @Software:PyCharm
# @Created Time:2024/1/12 1:57 PM
import lmdb
from scipy import sparse
import sys
sys.path.append("/home/share/huada/home/jiangwenjian/proj/scGPT/")
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import os
import pickle
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from tqdm import tqdm
import warnings
def folder2lmdb(dpath, write_frequency=100000,vocab_file='/home/share/huada/home/jiangwenjian/proj/scGPT/saves/whole_human/vocab.json'):
    vocab = GeneVocab.from_file(vocab_file)
    print("Generate LMDB to %s" % dpath)
    train_db = lmdb.open(dpath + '/train_test.db', map_size=536870912000*4, readonly=False, meminit=False, map_async=True)
    val_db = lmdb.open(dpath + '/val_test.db', map_size=536870912000, readonly=False, meminit=False, map_async=True)

    txn = train_db.begin(write=True)
    val_txn = val_db.begin(write=True)
    train_idx = 0
    val_idx = 0
    for f in tqdm([file for file in os.listdir(dpath) if file.endswith('.h5ad') ]):
        adata = sc.read_h5ad(os.path.join(dpath, f))
        print(f'\n\twriting {f}')
        val_index = np.random.randint(0, adata.n_obs, np.ceil(adata.n_obs*0.05).astype(np.int32))
        for i in range(adata.n_obs):
            binned=np.array(adata[i].layers['X_binned']).reshape(-1)
            values=np.array(adata[i].X).reshape(-1)
            gene_names=adata[i].var_names.tolist()
            gene_ids=np.array(vocab(gene_names), dtype=int)
            datapoint={'binned':binned,'values':values,'gene_ids':gene_ids}
            if i in val_index:
                val_txn.put(u'{}'.format(val_idx).encode('ascii'), pickle.dumps(datapoint))
                val_idx += 1
                if (val_idx + 1) % write_frequency == 0:
                    print('val write: ', val_idx)
                    val_txn.commit()
                    val_txn = val_db.begin(write=True)
            else:
                txn.put(u'{}'.format(train_idx).encode('ascii'), pickle.dumps(datapoint))
                train_idx += 1
            if (train_idx + 1) % write_frequency == 0:
                print('write: ', train_idx)
                txn.commit()
                txn = train_db.begin(write=True)

        print(train_idx, val_idx)
        break
    # finish iterating through dataset
    txn.commit()
    val_txn.commit()
    with train_db.begin(write=True) as txn, val_db.begin(write=True) as val_txn:
        txn.put(b'__len__', str(train_idx).encode())
        val_txn.put(b'__len__', str(train_idx).encode())
    print("Flushing database ...")
    train_db.sync()
    train_db.close()
    val_db.sync()
    val_db.close()

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    folder2lmdb(dpath="/home/share/huada/home/jiangwenjian/proj/scGPT/data/Pretraining/panglao/binned")