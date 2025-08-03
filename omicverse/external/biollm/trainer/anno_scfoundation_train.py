#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: anno_scfoundation_train.py
@time: 2024/3/3 15:02
"""
import torch
import torch.optim as optim
from ..utils.utils import get_reduced, distributed_concat, cal_model_params
from ..repo.scfoundation.load import gatherData
import os
import torch.nn as nn
from ..utils.log_manager import LogManager
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score

logger = LogManager().logger

def train(model, config, train_loader, val_loader, args=None, wandb=None):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_model = None
    params = cal_model_params(model)
    print('all params: ', params)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0
        num_batches = len(train_loader)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch, (inputs, targets) in enumerate(train_loader):
            x = inputs  # (B, L)
            value_labels = x > 0
            #
            x, x_padding = gatherData(x, value_labels, config['pad_token_id'])
            data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                              config['pad_token_id'])

            x = x.to(args.device)
            x_padding = x_padding.to(args.device)
            position_gene_ids = position_gene_ids.to(args.device)
            targets = targets.to(args.device)
            local_rank = int(os.environ['LOCAL_RANK']) if args.distributed else 0
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(local_rank)
            with torch.cuda.amp.autocast():
                logits = model(x, position_gene_ids, x_padding)
                loss = criterion(logits, targets)

                # Accumulate the loss for monitoring the training progress
                epoch_loss += loss.item()
                epoch_acc += (logits.argmax(1) == targets).sum().item() / targets.size(0)
                optimizer.zero_grad()
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
        if args.distributed:
            epoch_loss = get_reduced(epoch_loss, args.device, 0)
            epoch_acc = get_reduced(epoch_acc, args.device, 0)

        is_master = int(os.environ['RANK']) == 0 if args.distributed else True
        if is_master:
            logger.info(
                f"Epoch {epoch}: train/loss: {epoch_loss / num_batches}, train/acc: {epoch_acc / num_batches}")
            if wandb:
                wandb.log({
                    "train/loss": epoch_loss / num_batches,
                    "train/acc": epoch_acc / num_batches,
                })

        if args.distributed:
            dist.barrier()
        val_loss = evaluate(model, val_loader, epoch, args, is_master, wandb, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            if is_master:
                logger.info(f"Best model with score {best_val_loss:5.4f}")
        if is_master:
            torch.save(
                model.state_dict(),
                f"{args.output_dir}/model_{epoch}.pt",
            )
    return best_model


def evaluate(model, loader, epoch, args, is_master, wandb, config):
    model.eval()
    total_loss = 0.0
    predictions = []
    trues = []
    num_batches = len(loader)
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(loader):
            x = inputs  # (B, L)
            value_labels = x > 0
            x, x_padding = gatherData(x, value_labels, config['pad_token_id'])
            data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                              config['pad_token_id'])
            x = x.to(args.device)
            x_padding = x_padding.to(args.device)
            position_gene_ids = position_gene_ids.to(args.device)
            # inputs = inputs.to(self.device)
            targets = targets.to(args.device)
            with torch.cuda.amp.autocast():
                logits = model(x, position_gene_ids, x_padding)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, targets)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
            predictions.append(preds)
            trues.append(targets)
        if args.distributed:
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(loader.dataset), args.world_size)
            trues = distributed_concat(torch.cat(trues, dim=0), len(loader.dataset), args.world_size)
        else:
            predictions = torch.cat(predictions, dim=0)
            trues = torch.cat(trues, dim=0)
    predictions = predictions.cpu().numpy()
    trues = trues.cpu().numpy()
    accuracy = accuracy_score(trues, predictions)
    f1 = f1_score(trues, predictions, average='macro')
    if args.distributed:
        accuracy = get_reduced(accuracy, args.device, 0)
        f1 = get_reduced(f1, args.device, 0)
    if is_master:
        logger.info(
            f"Epoch {epoch}: eval/loss: {total_loss / num_batches:5.4f}, "
            f"eval/acc: {accuracy:5.3f}, f1-score: {f1:5.3f}")
        if wandb:
            wandb.log({
                'eval/loss': total_loss / num_batches,
                'eval/acc': accuracy,
                'eval/f1': f1
            })
    return total_loss / num_batches

def predict(model, data_loader, args, config=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, (inputs) in enumerate(data_loader):
            # Move the batch data to the appropriate device
            x = inputs[0]
            value_labels = x > 0
            #
            x, x_padding = gatherData(x, value_labels, config['pad_token_id'])
            data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                              config['pad_token_id'])

            x = x.to(args.device)
            x_padding = x_padding.to(args.device)
            position_gene_ids = position_gene_ids.to(args.device)
            with torch.cuda.amp.autocast():
                logits = model(x, position_gene_ids, x_padding)
                # Get the predicted labels
                batch_predictions = logits.argmax(1)
                # Append the predictions to the list
                predictions.extend(batch_predictions.cpu().numpy())
    return predictions