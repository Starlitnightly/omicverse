#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :bm_metrices_anno
# @Time      :2024/3/5 16:20
# @Author    :Luni Hu

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def compute_metrics(y_true, y_pred):

    # calculate accuracy and macro f1 using sklearn's function
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')

    return {
        'accuracy': accuracy,
        'macro_f1': f1,
        'recall': recall,
        'precision': precision
    }