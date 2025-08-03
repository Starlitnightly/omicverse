#!/usr/bin/env python3
# coding: utf-8
"""
@file: pert_task.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/02  create file.
"""

from biollm.tasks.bio_task import BioTask
import os
import scanpy as sc
import torch
from torch import nn
from biollm.repo.scgpt.loss import masked_mse_loss
from biollm.dataset.scgpt_dataset import make_train_data
from biollm.trainer.integration_scgpt_trainer import train, predict
from biollm.repo.scgpt.utils import eval_scib_metrics


class IntergrationTaskScgpt(BioTask):
    def __init__(self, config_file):
        super(IntergrationTaskScgpt, self).__init__(config_file, load_model=False)
        self.check_parameters()
        # init the func for the trainer
        self.criterion = nn.CrossEntropyLoss()

        self.logger.info(self.args)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.args.device = torch.device(self.args.device)
        self.criterion = self.criterion.to(self.args.device)
        print(self.args)

    def check_parameters(self):
        assert self.args.input_style in ["normed_raw", "log1p", "binned"]
        assert self.args.output_style in ["normed_raw", "log1p", "binned"]
        assert self.args.input_emb_style in ["category", "continuous", "scaling"]
        if self.args.input_style == "binned":
            if self.args.input_emb_style == "scaling":
                raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif self.args.input_style == "log1p" or self.args.input_style == "normed_raw":
            if self.args.input_emb_style == "category":
                raise ValueError(
                    "input_emb_style `category` is not supported for log1p or normed_raw input."
                )
        if self.args.input_emb_style == "category":
            self.args.mask_value = self.args.n_bins + 1
            self.args.pad_value = self.args.n_bins  # for padding gene expr values
            self.args.n_input_bins = self.args.n_bins + 2
        else:
            self.args.mask_value = -1
            self.args.pad_value = -2
            self.args.n_input_bins = self.args.n_bins

    def make_dataloader(self, adata):
        train_loader, val_loader = self.load_obj.get_dataloader(adata, self.args.do_preprocess, self.args.sort_seq_batch)

        return train_loader, val_loader

    def run(self):
        # make the data loader for the trainer
        adata = sc.read_h5ad(self.args.input_file)
        self.args.num_batch_labels = len(set(adata.obs['batch_id']))
        self.model = self.load_model()
        self.vocab = self.load_obj.vocab
        self.criterion = masked_mse_loss
        self.criterion_dab = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, eps=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=self.args.schedule_ratio)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        # for the finetune setting
        best_model = None
        if self.args.finetune:
            train_loader, val_loader = self.make_dataloader(adata)
            self.load_obj.freezon_model(keep_layers=[-2])
            self.model = self.model.to(self.args.device)
            best_model = train(
                model=self.model,
                loader=train_loader,
                val_loader=val_loader,
                scaler=self.scaler,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.args.device,
                args=self.args,
                criterion=self.criterion,
                criterion_dab=self.criterion_dab)
            torch.save(best_model.state_dict(), os.path.join(self.args.output_dir, 'anno_scgpt_best_model.pt'))
        if self.args.predicted:
            if self.args.do_preprocess:
                adata = self.load_obj.preprocess_adata(adata)
            adata = adata[adata.obs["batch_id"].argsort()]
            if best_model is None:
                best_model = self.model.to(self.args.device)
            train_data = make_train_data(adata, self.vocab, self.args)
            # data_loader = prepare_dataloader(train_data, self.args.batch_size)
            cell_embeddings = predict(best_model, train_data, self.args)
            adata.obsm["X_scGPT"] = cell_embeddings
            res = eval_scib_metrics(adata)
            print(res)


if __name__ == "__main__":
    # config_file = sys.argv[1]
    config_file = '../../config/intergration/scgpt.toml'
    obj = IntergrationTaskScgpt(config_file)
    obj.run()
