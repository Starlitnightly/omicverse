# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:pretrain_testdataset.py
# @Software:PyCharm
# @Created Time:2024/1/11 3:39 PM
import os.path
import sys

import scanpy

sys.path.append("../../")
import numpy as np
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.preprocess import Preprocessor
import os,scanpy
from scipy.sparse import issparse
def prepare_test(test_path,vocab,is_master,args,logger,test_name='pancreas'):
    if test_name=='pbmc10k':
        adata = scanpy.read_h5ad(os.path.join(test_path,'pbmc_10k.h5ad'))#11990 × 19099
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        #adata.var = adata.var.set_index("gene_symbols")
        data_is_raw = False

        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels

        adata.var["gene_name"] = adata.var.index.tolist()
        # set up the preprocessor, use the args to configs the workflow
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=3,  # step 1
            filter_cell_by_counts=3,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=args.n_hvg if args.n_hvg!=-1 else False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if args.data_is_raw else "cell_ranger",
            binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata, batch_key="str_batch" if args.data_name != "heart_cell" else None)

        # %%
        if args.per_seq_batch_sample:
            # sort the adata by batch_id in advance
            adata = adata[adata.obs["batch_id"].argsort()].copy()

        input_layer_key = "X_binned"
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        genes = adata.var["gene_name"].tolist()

        celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
        num_types = len(set(celltypes_labels))
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)
        adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        if is_master:
            logger.info(
                f"TestSet: match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
                f"in vocabulary of size {len(vocab)}."
            )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)

        return adata,gene_ids,gene_ids_in_vocab
    elif test_name=='pancreas':
        adata = scanpy.read_h5ad(os.path.join(test_path,'pancreas.h5ad'))
        ori_batch_col = "batch_id"
        adata.obs["celltype"] = adata.obs["ontology_name"].astype("category")
        # adata.var = adata.var.set_index("gene_symbols")
        data_is_raw = False

        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels

        adata.var["gene_name"] = adata.var['Symbol'].tolist()
        # set up the preprocessor, use the args to configs the workflow
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=3,  # step 1
            filter_cell_by_counts=3,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=args.n_hvg if args.n_hvg != -1 else False,
            # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata, batch_key="str_batch" if args.data_name != "heart_cell" else None)

        # %%
        if args.per_seq_batch_sample:
            # sort the adata by batch_id in advance
            adata = adata[adata.obs["batch_id"].argsort()].copy()

        input_layer_key = "X_binned"
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        genes = adata.var["gene_name"].tolist()

        celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
        num_types = len(set(celltypes_labels))
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)
        adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        if is_master:
            logger.info(
                f"TestSet: match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
                f"in vocabulary of size {len(vocab)}."
            )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)

        return adata, gene_ids, gene_ids_in_vocab