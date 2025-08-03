#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: integration_scgpt_trainer.py
@time: 2024/3/3 15:02
"""
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import copy
from ..repo.scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)


def train(model, loader, val_loader, scaler, optimizer, scheduler, device, args, criterion,
          criterion_dab):
    """
    Train the model for one epoch.
    """
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = args.log_interval

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(args.pad_value)
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels,
                    MVC=True,
                    ECS=True,
                )

                masked_positions = input_values.eq(args.mask_value)  # the postions to predict
                loss = loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metrics_to_log = {"train/mse": loss_mse.item()}
                if args.explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if args.do_mvc:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                    metrics_to_log.update({"train/mvc": loss_gepc.item()})
                if args.do_mvc and args.explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                if args.ecs_threshold > 0:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + args.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    print(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if args.do_mvc else 0.0
            total_error += mre.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if args.do_mvc else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if args.do_mvc else "")
                )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
        val_loss, val_mre = evaluate(epoch, model, val_loader, args, criterion, criterion_dab)
        print(
            f"| end of epoch {epoch:3d} | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            print(f"Best model with score {best_val_loss:5.4f}")
            torch.save(best_model.state_dict(), args.output_dir + f"/best_model.pt")
        scheduler.step()
    return best_model


def evaluate(epoch, model, loader, args, criterion, criterion_dab):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(args.device)
            input_values = batch_data["values"].to(args.device)
            target_values = batch_data["target_values"].to(args.device)
            batch_labels = batch_data["batch_labels"].to(args.device)

            src_key_padding_mask = input_gene_ids.eq(args.pad_value)
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(args.mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    print(
        {
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + args.dab_weight * total_dab)
                                 / total_num,
            "epoch": epoch,
        },
    )

    return total_loss / total_num, total_error / total_num


def predict(model, tokenized_data, args):
    model.eval()
    all_gene_ids, all_values, batch_ids = tokenized_data["gene_ids"], tokenized_data["values"], tokenized_data['batch_labels']
    all_gene_ids, all_values, batch_ids = all_gene_ids.to(args.device), all_values.to(args.device), batch_ids.to(args.device)
    src_key_padding_mask = all_gene_ids.eq(args.pad_value).to(args.device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        cell_embeddings = model.encode_batch(
            all_gene_ids,
            all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=args.batch_size,
            batch_labels=batch_ids.long(),
            time_step=0,
            return_np=True,
        )
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    return cell_embeddings
