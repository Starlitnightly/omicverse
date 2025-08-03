#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: bm_cell_embedding.py
@time: 2025/3/12 10:11
"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score


def cluster_metrics(X, y_pred, y_true):
    """
    embedding 是 X，聚类标签是 y_pred，真实细胞类型是 y_true
    """
    silhouette = silhouette_score(X, y_pred)
    ch_index = calinski_harabasz_score(X, y_pred)
    db_index = davies_bouldin_score(X, y_pred)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    print(f"Silhouette Score: {silhouette}")
    print(f"CH Index: {ch_index}")
    print(f"DB Index: {db_index}")
    print(f"ARI: {ari}")
    print(f"NMI: {nmi}")
    print(f"AMI: {ami}")
    return {'silhouette': silhouette, 'ch_index': ch_index, 'db_index': db_index, 'ari': ari, 'nmi': nmi, 'ami': ami}
