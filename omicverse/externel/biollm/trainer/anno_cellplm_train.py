#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: anno_cellplm_train.py
@time: 2024/3/3 15:02
"""
import pandas as pd
import scanpy as sc
import json
import os
from ..repo.CellPLM.pipeline.cell_type_annotation import (CellTypeAnnotationDefaultModelConfig,
                                                               CellTypeAnnotationPipeline,
                                                               CellTypeAnnotationDefaultPipelineConfig)


def train(data, args):
    """
    Train the model for one epoch.
    """
    data.var = data.var.set_index(args.var_key)
    data.obs['split'] = 'test'
    train_num = data.shape[0]
    tr = np.random.permutation(train_num)
    data.obs['split'][tr[:int(train_num * 0.9)]] = 'train'
    data.obs['split'][tr[int(train_num * 0.9):train_num]] = 'valid'
    pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()

    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config['out_dim'] = data.obs['celltype'].nunique()
    pipeline = CellTypeAnnotationPipeline(pretrain_prefix=args.pretrain_version, overwrite_config=model_config,
                                          pretrain_directory=args.pretrain_directory)
    pipeline.fit(data,  # An AnnData object
                 pipeline_config,  # The config dictionary we created previously, optional
                 split_field='split',  # Specify a column in .obs that contains split information
                 train_split='train',
                 valid_split='valid',
                 label_fields=[args.obs_key])  # Specify a column in .obs that contains cell type labels
    # Save the model
    torch.save(pipeline.model.state_dict(), f'{args.output_dir}/cellplm_best_model.pth')
    return pipeline


def predict(pipeline, data, args):
    data.obs['split'] = 'test'
    data.var = data.var.set_index(args.var_key)
    pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config['out_dim'] = data.obs[args.obs_label].nunique()
    pred = pipeline.predict(
        data,  # An AnnData object
        pipeline_config,  # The config dictionary we created previously, optional
    )
    metric = pipeline.score(data,  # An AnnData object
                   pipeline_config,  # The config dictionary we created previously, optional
                   split_field='split',  # Specify a column in .obs to specify train and valid split, optional
                   target_split='test',  # Specify a target split to predict, optional
                   label_fields=[args.obs_label])  # Specify a column in .obs that contains cell type labels
    with open(f'{args.output_dir}/metric.json', 'w') as fd:
        json.dump(metric, fd)
    with open(f'{args.output_dir}/predict.pk', 'wb') as fd:
        json.dump(pred.detach().cpu.numpy(), fd)

