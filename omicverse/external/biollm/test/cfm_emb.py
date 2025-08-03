#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: cfm_emb.py
@time: 2025/3/13 17:38
"""
from biollm.tasks.bio_task import BioTask
import scanpy as sc



class CellEmbTask(BioTask):
    def __init__(self, config_file):
        super(CellEmbTask, self).__init__(config_file)
        self.logger.info(self.args)

    def run(self):
        # get cell embedding
        adata = sc.read_h5ad(self.args.input_file)[0: 100].copy()
        if self.args.gene_symbol_key:
            adata.var_names = adata.var[self.args.gene_symbol_key].values
        emb = self.llm_embedding(emb_type=self.args.emb_type, adata=adata)
        if self.args.emb_type == 'cell':
            print(emb.shape)
        else:
            print(len(emb['gene_names']), emb['gene_emb'].shape)


if __name__ == "__main__":
    import sys

    # config_file = sys.argv[1]
    for i in [
    '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM2/biollm/config/embeddings/cell_emb/cfm/blood_v1.toml',
    ]:
        config_file = i
        obj = CellEmbTask(config_file)
        obj.run()
