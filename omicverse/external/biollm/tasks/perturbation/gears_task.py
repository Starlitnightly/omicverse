#!/usr/bin/env python3
# coding: utf-8
"""
@file: gears_task.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/03/18  create file.
"""
from biollm.tasks.bio_task import BioTask
import numpy as np
from os.path import join as pjoin
from biollm.repo.gears import PertData, GEARS
import torch
import scanpy as sc
from biollm.utils.utils import gene2vec_embedding
from biollm.repo.scfoundation.GEARS.gears import GEARS as scFoundationGEARS
from biollm.repo.scfoundation.GEARS.gears import PertData as scFoundationPertData


class GearsTask(BioTask):
    def __init__(self, cfs_file):
        super(GearsTask, self).__init__(cfs_file)
        self.gene2ids = self.load_obj.get_gene2idx() if self.load_obj is not None else {}
        if self.load_obj is not None:
            self.load_obj.freezon_model(keep_layers=[-2])
        if self.model is not None:
            self.model = self.model.to(self.device)

    def make_pert_data(self):
        pert_data = scFoundationPertData(self.args.data_dir) if (self.args.model_used == 'scfoundation' and
                                                                 self.args.emb_type == 'gene-expression') else PertData(
            self.args.data_dir)
        if self.args.model_used == 'scmamba' and self.args.emb_type == 'gene-expression':
            pert_data.llm_loader = self.load_obj
        gene_subset = list(self.vocab.get_stoi().keys()) if self.args.emb_type == 'gene-expression' else None
        # load dataset in paper: norman, adamson, dixit.
        if self.args.data_name in ['norman', 'adamson', 'dixit']:
            pert_data.load(data_name=self.args.data_name, gene_subset=gene_subset)
        else:
            adata = self.read_h5ad(pjoin(self.args.data_dir, self.args.data_name + '.h5ad'), filter_gene=True)
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            pert_data.new_data_process(dataset_name=self.args.data_name, adata=adata)
        # specify data split
        pert_data.prepare_split(split=self.args.split, seed=self.args.seed,
                                train_gene_set_size=self.args.train_gene_set_size)
        # get dataloader with batch size
        pert_data.get_dataloader(batch_size=self.args.batch_size, test_batch_size=self.args.test_batch_size)
        return pert_data

    def universal_gene_embdedding(self, pert_data):
        gene_list = pert_data.gene_names.values.tolist()
        self.logger.info('len of gene_list: {}'.format(len(gene_list)))
        gene_emb_weight = torch.nn.Embedding(len(gene_list),
                                             self.args.pretrained_emb_size).weight.detach().numpy() if self.args.use_pretrained else None
        if self.args.use_pretrained and self.args.model_used != 'gene2vec':
            gene_in_vocab = [gene for gene in pert_data.adata.var.gene_name if gene in self.gene2ids]
            gene_ids = np.array([self.gene2ids[gene] for gene in gene_in_vocab])
            gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(self.device)
            # get genes embedding from the pretrained model.
            ids_embedding = self.load_obj.get_gene_embedding(gene_ids)
            ids_embedding = ids_embedding if isinstance(ids_embedding,
                                                        np.ndarray) else ids_embedding.detach().cpu().numpy()
            for i in range(len(gene_in_vocab)):
                gene_emb_weight[gene_list.index(gene_in_vocab[i])] = ids_embedding[i]
        if self.args.model_used == 'gene2vec':
            g2v_emb_dict = gene2vec_embedding(self.args.g2v_file, self.args.g2v_genes)
            for i in g2v_emb_dict:
                if i in gene_list:
                    gene_emb_weight[gene_list.index(i)] = g2v_emb_dict[i]
        gene_emb_weight = torch.from_numpy(gene_emb_weight).to(self.device) if self.args.use_pretrained else None
        return gene_emb_weight

    def run(self):
        self.logger.info(self.args)
        pert_data = self.make_pert_data()
        self.logger.info('adata shape: {}'.format(pert_data.adata))
        self.args.max_seq_len = pert_data.adata.shape[1]
        # set up and train a model

        if self.args.emb_type == 'universal':
            gene_emb_weight = self.universal_gene_embdedding(pert_data)
            gears_model = GEARS(pert_data, device=self.device, model_output=self.args.result_dir)
            gears_model.model_initialize(hidden_size=self.args.hidden_size,
                                         use_pretrained=self.args.use_pretrained,
                                         pretrain_freeze=self.args.get('pretrain_freeze', False),
                                         gene_emb_weight=gene_emb_weight,
                                         pretrained_emb_size=self.args.get('pretrained_emb_size', 512),
                                         pretrain_emb_type=self.args.emb_type)
            if self.args.finetune:
                gears_model.train(epochs=self.args.epochs, lr=self.args.lr)

                # # save model
                gears_model.save_model(self.args.result_dir)
        elif self.args.model_used == 'scfoundation':
            gears_model = scFoundationGEARS(pert_data,
                                            device=self.device,
                                            weight_bias_track=self.args.weight_bias_track,
                                            proj_name=self.args.proj_name,
                                            exp_name=self.args.exp_name,
                                            )
            gears_model.model_initialize(hidden_size=self.args.hidden_size,
                                         model_type=self.args.model_type,
                                         bin_set=self.args.bin_set,
                                         load_path=self.args.model_file,
                                         finetune_method=self.args.finetune_method,
                                         accumulation_steps=self.args.accumulation_steps,
                                         mode=self.args.mode,
                                         highres=self.args.highres)
            if self.args.finetune:
                gears_model.train(epochs=self.args.epochs, lr=self.args.lr, result_dir=self.args.result_dir)
                gears_model.save_model(self.args.result_dir)
        else:
            gene_emb_weight = self.universal_gene_embdedding(pert_data)
            gears_model = GEARS(pert_data,
                                device=self.device,
                                model_output=self.args.result_dir,
                                weight_bias_track=self.args.weight_bias_track,
                                proj_name=self.args.proj_name,
                                exp_name=self.args.exp_name)
            gears_model.model_initialize(hidden_size=self.args.hidden_size, use_pretrained=self.args.use_pretrained,
                                         pretrain_freeze=self.args.pretrain_freeze, gene_emb_weight=gene_emb_weight,
                                         pretrained_emb_size=self.args.pretrained_emb_size, model_loader=self.load_obj,
                                         pretrain_emb_type=self.args.emb_type)
            if self.args.finetune:
                gears_model.train(epochs=self.args.epochs, lr=self.args.lr)
                gears_model.save_model(self.args.result_dir)


if __name__ == '__main__':
    import sys

    # config_file = '../../configs/pert/gears_mamba_gene-express.toml'
    # config_file = '../../configs/pert/gears_mamba.toml'
    config_file = sys.argv[1]
    obj = GearsTask(config_file)
    obj.run()
