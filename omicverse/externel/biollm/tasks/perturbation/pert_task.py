#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: pert_task.py
@time: 2024/3/3 15:02
"""
import os
import gc
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from ..bio_task import BioTask
from ...repo.gears import PertData
from ...repo.scgpt.loss import masked_mse_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ...trainer.pert_scgpt_trainer import PertScgptTrainer
from ...trainer.pert_scmamba_trainer import PertScmambaTrainer
import warnings
import pickle
from ...evaluate.perturbation import plot_perturbation


class PertTask(BioTask):
    def __init__(self, config_file):
        super(PertTask, self).__init__(config_file)
        if self.args.model_used == 'scgpt':
            if self.args.finetune:
                self.args['load_param_prefixs'] = ["encoder", "value_encoder", "transformer_encoder"]
            self.model = self.load_obj.load_pert_model()
            self.model = self.load_obj.init_model(self.model)
            self.model.to(self.device)
        if self.args.model_used == 'scmamba':
            self.model.do_pert = True
        self.load_obj.freezon_model(keep_layers=[-2])
        self.criterion = masked_mse_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.schedule_interval, gamma=0.9)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def make_dataset(self):
        pert_data = PertData(self.args.data_dir)
        if self.args.model_used == 'scmamba':
            pert_data.llm_loader = self.load_obj
        gene_subset = list(self.vocab.get_stoi().keys()) if self.args.emb_type == 'gene-expression' else None
        try:
            if self.args.data_name in ['norman', 'adamson', 'dixit']:
                pert_data.load(data_name=self.args.data_name, gene_subset=gene_subset)
            else:
                pert_data.load(data_path=self.args.data_dir, gene_subset=gene_subset)
        except Exception as e:
            adata = self.read_h5ad(os.path.join(self.args.data_dir, self.args.data_name + '.h5ad'), filter_gene=True)
            pert_data.new_data_process(dataset_name=self.args.data_name, adata=adata)
        # specify data split
        pert_data.prepare_split(split=self.args.split, seed=self.args.seed,
                                train_gene_set_size=self.args.train_gene_set_size)
        # get dataloader with batch size
        pert_data.get_dataloader(batch_size=self.args.batch_size, test_batch_size=self.args.eval_batch_size)
        # check how many genes in vocab
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        self.logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()
        gene_ids = np.array(
            [self.vocab[gene] if gene in self.vocab else self.vocab["<pad>"] for gene in genes], dtype=int
        )
        return pert_data, gene_ids

    def run(self):
        best_val_loss = float("inf")
        best_model = None
        patience = 0
        pert_data, gene_ids = self.make_dataset()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]
        if self.args.model_used == 'scgpt':
            trainer = PertScgptTrainer(self.args, self.model, train_loader, valid_loader, self.optimizer, self.scheduler,
                                  self.scaler, masked_mse_loss)
        else:
            trainer = PertScmambaTrainer(self.args, self.model, train_loader, valid_loader, self.optimizer, self.scheduler,
                                  self.scaler, masked_mse_loss)
        if self.args.model_used in ['scgpt', 'scmamba']:
            if self.args.finetune:
                epoch_start_time = time.time()
                for epoch in range(1, self.args.epochs + 1):
                    trainer.train(epoch, gene_ids)
                    val_loss, val_mre = trainer.evaluate(gene_ids)
                    elapsed = time.time() - epoch_start_time
                    self.logger.info("-" * 89)
                    self.logger.info(
                        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                        f"valid loss/mse {val_loss:5.4f} |"
                    )
                    self.logger.info("-" * 89)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(self.model)
                        self.logger.info(f"Best model with score {best_val_loss:5.4f}")
                        patience = 0
                    else:
                        patience += 1
                        if patience >= self.args.early_stop:
                            self.logger.info(f"Early stop at epoch {epoch}")
                            break

                    torch.save(
                        self.model.state_dict(),
                        f"{self.args.save_dir}/model_{epoch}.pt",
                    )

                    self.scheduler.step()
                torch.save(best_model.state_dict(), self.args.save_dir + "/best_model.pt")
                self.model = best_model
            if self.args.predict:
                query = self.args.query
                pert_list = [[i for i in query.split("+") if i != 'ctrl']]
                pred = trainer.predict(pert_data, pert_list, gene_ids, self.args.predict_sample_num)
                with open(os.path.join(self.args.save_dir, f"pred_{query.replace('+', '_')}.pk"), 'wb') as w:
                    pickle.dump(pred, w)
                jpg_path = os.path.join(self.args.save_dir, f"pred_degene_{query.replace('+', '_')}.jpg")
                plot_perturbation(pert_data, pred, query, jpg_path)


if __name__ == "__main__":
    # config_file = sys.argv[1]
    config_file = '../../config/pert/scmamba_pert.toml'
    obj = PertTask(config_file)
    obj.run()
