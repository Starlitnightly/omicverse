#!/usr/bin/env python3
# coding: utf-8
"""
@file: bm_metrices_gears.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/03/21  create file.
"""
from biollm.repo.gears import GEARS, PertData
import os
from biollm.repo.gears.inference import evaluate, compute_metrics, deeper_analysis
import pickle


def load_data(data_path, data_name='norman'):
    pert_data = PertData(data_path)
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split='simulation', seed=1, train_gene_set_size=0.75)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    return pert_data


def load_model(model_dir, device):
    model = GEARS(pert_data, device=device)
    model.load_pretrained(model_dir)
    return model


def calculate_metrices_gears(pert_data, model, output=None):
    print('test_subgroup:', pert_data.subgroup['test_subgroup'].keys())
    print('condition:', pert_data.adata.obs.condition.unique())
    test_res = evaluate(model.dataloader['test_loader'], model.best_model,
                           model.config['uncertainty'], model.device)
    test_metrics, test_pert_res = compute_metrics(test_res)
    print(test_metrics)
    if output is not None:
        with open(output, 'wb') as f:
            pickle.dump([test_res, test_metrics, test_pert_res], f)
    return test_res, test_metrics, test_pert_res


if __name__ == '__main__':
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/'
    data_name = 'norman'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/evaluate/'
    model_path = data_path + '/model_best.pt'
    pert_data = load_data(data_path, data_name)
    for i in ['norman', 'norman_scbert', 'norman_scgpt', 'norman_mamba_gst_ori_initemb']:
        model_path = data_path + '{}/model_best.pt'.format(i)
        calculate_metrices_gears(pert_data, model_path, output_path + '{}.pk'.format(i))
