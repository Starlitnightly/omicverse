#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: cell_embedding.py
@time: 2025/3/12 10:21
"""
from biollm.tasks.bio_task import BioTask
import scanpy as sc
import gc
from biollm.evaluate.bm_cell_embedding import cluster_metrics
import matplotlib.pyplot as plt
import pickle
from scib_metrics.benchmark import Benchmarker


class CellEmbedding(BioTask):
    def __init__(self, config_file):
        super(CellEmbedding, self).__init__(config_file)
        self.adata = sc.read_h5ad(self.args.input_file)
        if 'max_seq_len' in self.args and self.args.max_seq_len < 0:
            self.args.max_seq_len = self.adata.shape[1]
        self.logger.info(self.args)

    def run(self):
        # get cell embedding
        adata = self.adata
        if self.args.var_key:
            adata.var_names = adata.var[self.args.var_key].values

        emb = self.llm_embedding(emb_type=self.args.emb_type, adata=adata)
        adata.obsm['X_emb'] = emb
        with open(self.args.output_dir + '/cell_emb.pk', 'wb') as fd:
            pickle.dump(adata.obsm['X_emb'], fd)
        scores = self.eval_cluster(adata)
        if 'batch_key' in self.args:
            self.batch_effect(adata)

        del adata
        gc.collect()
        return scores

    def eval_cluster(self, adata):
        sc.pp.neighbors(adata, use_rep="X_emb")  # 基于 embedding 计算 KNN 图
        sc.tl.leiden(adata)  # Leiden 聚类
        y_pred = adata.obs["leiden"].astype(int).to_numpy()  # 获取聚类标签
        y_true = adata.obs[self.args.obs_key]
        scores = cluster_metrics(adata.obsm['X_emb'], y_pred=y_pred, y_true=y_true)
        with open(self.args.output_dir + '/cluster_metrics.pk', 'wb') as fd:
            pickle.dump(scores, fd)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc.tl.umap(adata, min_dist=0.3)
        # UMAP with Leiden clusters
        sc.pl.umap(adata, color="leiden", ax=axes[0], show=False, legend_loc="on data")
        axes[0].set_title("Leiden Clustering")
        # UMAP with True Cell Types
        sc.pl.umap(adata, color=self.args.obs_key, ax=axes[1], show=False, legend_loc=None)
        axes[1].set_title("True Cell Types")
        plt.tight_layout()
        plt.savefig(self.args.output_dir + '/cluster_umap.pdf')
        return scores

    def batch_effect(self, adata):
        bm1 = Benchmarker(
            adata,
            batch_key=self.args.batch_key,
            label_key=self.args.obs_key,
            embedding_obsm_keys=['X_emb'],
            n_jobs=6)
        bm1.benchmark()
        df = bm1.get_results(min_max_scale=True)
        df.to_csv(self.args.output_dir + '/batch_effect_bench.csv')
        bm1.plot_results_table(show=False, save_dir=self.args.output_dir)


if __name__ == "__main__":
    import sys

    # config_file = sys.argv[1]

    files = [
        '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM1/biollm/config/embeddings/cell_emb/cellplm/gse155468.toml',
    ]

    for i in files:
        config_file = i
        obj = CellEmbedding(config_file)
        obj.run()
