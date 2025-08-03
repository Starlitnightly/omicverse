#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: grn_task.py
@time: 2024/3/12 11:33
"""
from .bio_task import BioTask
import scanpy as sc
import gc
from ..evaluate.bm_metrices_grn import GeneEmbedding
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


class GrnTask(BioTask):
    def __init__(self, config_file):
        super(GrnTask, self).__init__(config_file)
        self.adata = sc.read_h5ad(self.args.input_file)
        if 'max_seq_len' in self.args and self.args.max_seq_len < 0:
            self.args.max_seq_len = self.adata.shape[1]
        self.logger.info(self.args)

    def run(self):
        adata = self.adata
        if self.args.var_key:
            adata.var_names = adata.var[self.args.var_key].values
        emb = self.llm_embedding(emb_type=self.args.emb_type, adata=adata)
        gene_emb = {emb['gene_names'][i]: emb['gene_emb'][i] for i in range(len(emb['gene_names']))}
        go_files = ['mf',  'cc', 'bp']
        embed = GeneEmbedding(gene_emb)
        for go in go_files:
            go_file = f'{self.args.go_gene_set_dir}/c5.go.{go}.v2024.1.Hs.symbols.gmt'
            result = embed.go_enrich(go_file)
            df_enrichment_combined = pd.concat(result, ignore_index=True)
            output_path = f"{self.args.output_dir}/GO_{go}.csv"
            df_enrichment_combined.to_csv(output_path, index=False)
        del adata
        gc.collect()
        self.plot(self.args.output_dir)

    @staticmethod
    def plot(path_prefix):
        def count_genes(gene_string):
            if pd.isna(gene_string):
                return 0
            return len(gene_string.split(';'))
        go_types = ['MF', 'BP', 'CC']
        fig, axes = plt.subplots(1, len(go_types), figsize=(5 * len(go_types), 6), sharex=True)
        if len(go_types) == 1:
            axes = [axes]

        for ax, go_type in zip(axes, go_types):
            file_path = os.path.join(path_prefix, f'GO_{go_type.lower()}.csv')
            if not os.path.exists(file_path):
                print(f'File not found: {file_path}')
                continue
            df = pd.read_csv(file_path)
            df_filtered = df[df['Adjusted P-value'] < 0.01].copy()
            df_filtered['gene_count'] = df_filtered['Genes'].apply(count_genes)
            df_filtered = df_filtered[df_filtered['gene_count'] > 25]
            resolution_data = df_filtered.groupby('resolution').agg(
                total_term_count=('Term', 'nunique')
            ).reset_index()
            ax.plot(resolution_data['resolution'], resolution_data['total_term_count'],
                    alpha=0.8, linewidth=2.5)
            ax.scatter(resolution_data['resolution'], resolution_data['total_term_count'],
                       s=30, alpha=0.8, edgecolors='none')
            ax.set_title(f'GO Enrichment ({go_type})')
            ax.set_xlabel('Resolution')
            ax.set_ylabel('Number of Enriched GO Pathways')
            ax.legend(title="Model", frameon=False)
        plt.tight_layout()
        plt.rcParams['pdf.fonttype'] = 42
        plt.savefig(os.path.join(path_prefix, f'go_comparison.pdf'), bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    import sys
    # config_file = sys.argv[1]
    config_file = '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM1/biollm/config/embeddings/gene_exp_emb/scgpt .toml'
    obj = GrnTask(config_file)
    obj.run()

