#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: pert_scmamba_trainer.py
@time: 2024/3/3 15:02
"""
import copy
import gc
import json
import time
import warnings

import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from ..repo.scgpt.utils import map_raw_id_to_vocab_id
from ..repo.scgpt.loss import masked_relative_error
from ..repo.gears.utils import create_cell_graph_dataset_for_prediction
from torch_geometric.loader import DataLoader


class PertScmambaTrainer(Trainer):
    def __init__(self, args, model, train_loader, val_loader, optimizer, scheduler, scaler, criterion):
        super(PertScmambaTrainer, self).__init__(args, model, train_loader)
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.criterion = criterion

    def train(self, epoch, gene_ids):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss, total_mse = 0.0, 0.0
        start_time = time.time()
        # n_genes = len(gene_ids)
        num_batches = len(self.train_loader)
        for batch, batch_data in enumerate(self.train_loader):
            batch_size = len(batch_data.y)
            batch_data.to(self.device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 1)
            ori_gene_values = x[:, 0].view(batch_size, -1)
            n_genes = ori_gene_values.shape[-1]
            pert_flags = torch.Tensor(np.stack(batch_data.pert_flag)).long().to(self.device)
            sorted_layer_idx = batch_data.sorted_layer_idx
            target_gene_values = batch_data.y  # (batch_size, n_genes)
            input_gene_ids = torch.arange(n_genes, device=self.device, dtype=torch.long)
            # if self.args.include_zero_gene == "all":
            #     input_gene_ids = torch.arange(n_genes, device=self.device, dtype=torch.long)
            # else:
            #     input_gene_ids = (
            #         ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            #     )
            # # sample input_gene_id
            # if len(input_gene_ids) > self.args.max_seq_len:
            #     input_gene_ids = torch.randperm(len(input_gene_ids), device=self.device)[
            #                      :self.args.max_seq_len
            #                      ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]
            sorted_layer_idx = sorted_layer_idx[:, input_gene_ids]
            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=self.device
            )
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = self.model(
                    mapped_input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    sorted_layer_idx=sorted_layer_idx,
                    input_pert_flags=input_pert_flags,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool
                )  # Use all
                loss = loss_mse = self.criterion(output_values, target_values, masked_positions)

            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    self.logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # torch.cuda.empty_cache()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            if batch % self.args.log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / self.args.log_interval
                cur_loss = total_loss / self.args.log_interval
                cur_mse = total_mse / self.args.log_interval
                # ppl = math.exp(cur_loss)
                self.logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                )
                total_loss = 0
                total_mse = 0
                start_time = time.time()

    def evaluate(self, gene_ids) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        self.model.eval()
        total_loss = 0.0
        total_error = 0.0
        # n_genes = len(gene_ids)
        with torch.no_grad():
            for batch, batch_data in enumerate(self.val_loader):
                batch_size = len(batch_data.y)
                batch_data.to(self.device)
                x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
                ori_gene_values = x[:, 0].view(batch_size, -1)
                n_genes = ori_gene_values.shape[-1]
                pert_flags = torch.Tensor(np.stack(batch_data.pert_flag)).long().to(self.device)
                target_gene_values = batch_data.y  # (batch_size, n_genes)
                sorted_layer_idx = batch_data.sorted_layer_idx
                input_gene_ids = torch.arange(n_genes, device=self.device, dtype=torch.long)
                # if self.args.include_zero_gene in ["all", "batch-wise"]:
                #     if self.args.include_zero_gene == "all":
                #         input_gene_ids = torch.arange(n_genes, device=self.device)
                #     else:  # when batch-wise
                #         input_gene_ids = (
                #             ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                #         )
                #     # sample input_gene_id
                #     if len(input_gene_ids) > self.args.max_seq_len:
                #         input_gene_ids = torch.randperm(len(input_gene_ids), device=self.device)[
                #                          :self.args.max_seq_len
                #                          ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                sorted_layer_idx = sorted_layer_idx[:, input_gene_ids]

                print(input_pert_flags.sum())
                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = self.model(
                    mapped_input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    sorted_layer_idx=sorted_layer_idx,
                    input_pert_flags=input_pert_flags,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = self.criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
        return total_loss / len(self.val_loader), total_error / len(self.val_loader)

    def predict(self, pert_data, pert_list, gene_ids, pool_size=None):
        """
        Predict the gene expression values for the given perturbations.

        Args:
            model (:class:`torch.nn.Module`): The model to use for prediction.
            pert_list (:obj:`List[str]`): The list of perturbations to predict.
            pool_size (:obj:`int`, optional): For each perturbation, use this number
                of cells in the control and predict their perturbation results. Report
                the stats of these predictions. If `None`, use all control cells.
        """
        adata = pert_data.adata
        ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
        if pool_size is None:
            pool_size = len(ctrl_adata.obs)
        gene_list = pert_data.gene_names.values.tolist()
        for pert in pert_list:
            for i in pert:
                if i not in gene_list:
                    raise ValueError(
                        f"The gene {i} is not in the perturbation graph. Please select from GEARS.gene_list!"
                    )

        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            results_pred = {}
            for pert in pert_list:
                cell_graphs = create_cell_graph_dataset_for_prediction(
                    pert, ctrl_adata, gene_list, device, num_samples=pool_size
                )
                loader = DataLoader(cell_graphs, batch_size=self.args.eval_batch_size, shuffle=False)
                preds = []
                for _, batch_data in enumerate(loader):
                    pred_gene_values = self.model.pred_perturb(
                        batch_data, self.args.include_zero_gene, gene_ids=gene_ids, amp=True
                    )
                    preds.append(pred_gene_values)
                preds = torch.cat(preds, dim=0)
                results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

        return results_pred
