#!/usr/bin/env python3
# coding: utf-8
"""
@file: trainer.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/03  create file.
"""
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
import torch.distributed as dist
from ..repo.scbert.utils import *
from datetime import datetime
import copy


def train(model, train_loader, val_loader, args, wandb=None):
    world_size = args.world_size
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_master = local_rank == 0 if args.distributed else True
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    loss_fn = nn.CrossEntropyLoss(weight=None)
    if args.distributed:
        loss_fn = loss_fn.to(args.local_rank)
        dist.barrier()
    if is_master:
        print(datetime.now().strftime("%y-%m-%d %H:%M:%S"), " start to train....")
    best_val_loss = float("inf")
    best_model = None
    patience = 0
    for i in range(1, args.epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(i)
        model.train()
        running_loss = 0.0
        cum_acc = 0.0
        for index, (data, labels) in enumerate(train_loader):
            index += 1
            data, labels = data.to(device), labels.to(device)
            if args.distributed:
                if index % args.GRADIENT_ACCUMULATION != 0:
                    with model.no_sync():
                        logits = model(data)
                        loss = loss_fn(logits, labels)
                        loss.backward()
                if index % args.GRADIENT_ACCUMULATION == 0:
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
            # print(index,  loss)
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final = softmax(logits)
            final = final.argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            if is_master:
                if wandb:
                    wandb.log({
                        "train/step_loss": loss.item()
                    })

        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        if is_master:
            print(f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
            if wandb:
                wandb.log({
                    "train/loss": epoch_loss,
                    "train/acc": epoch_acc,
                })
        if args.distributed:
            dist.barrier()
        scheduler.step()
        val_loss = evaluate(i, model, val_loader, loss_fn, args, wandb, False)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            if is_master:
                print(f"Best model with score {best_val_loss:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                if is_master:
                    print(f"Early stop at epoch {i}")
                break
        torch.save(
            model.state_dict(),
            f"{args.output_dir}/model_{i}.pt",
        )

    if best_model is not None:
        torch.save(best_model.state_dict(), args.output_dir + "/best_model.pt")
    model = best_model
    return model


def evaluate(epoch, model, val_loader, loss_fn, args, wandb=None, return_pred=False):
    model.eval()
    world_size = args.world_size
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.distributed:
        dist.barrier()
    running_loss = 0.0
    predictions = []
    truths = []
    is_master = local_rank == 0
    with torch.no_grad():
        for index, (data_v, labels_v) in enumerate(val_loader):
            index += 1
            data_v, labels_v = data_v.to(device), labels_v.to(device)
            logits = model(data_v)
            loss = loss_fn(logits, labels_v)
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.cpu()), axis=-1) < args.UNASSIGN_THRES] = -1
            predictions.append(final)
            truths.append(labels_v)
        del data_v, labels_v, logits, final_prob, final
        # gather
        if args.distributed:
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_loader.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_loader.dataset), world_size)
        else:
            predictions = torch.cat(predictions, dim=0)
            truths = torch.cat(truths, dim=0)
        no_drop = predictions != -1
        predictions = np.array((predictions[no_drop]).cpu())
        truths = np.array((truths[no_drop]).cpu())
        cur_acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, average='macro')
        val_loss = running_loss / index
        if args.distributed:
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}    ==  Epoch: {epoch} | Validation Loss: {val_loss:.6f} | Accuracy: {cur_acc:.6f} | F1 Score: {f1:.6f}  ==')
            if wandb:
                wandb.log({
                    'eval/loss': val_loss,
                    'eval/acc': cur_acc,
                    'eval/f1': f1
                })
    if return_pred:
        return predictions

    return val_loss


def predict(model, data_loader, args):
    world_size = args.world_size
    device = args.device
    model.eval()
    if args.distributed:
        dist.barrier()
    predictions = []
    with torch.no_grad():
        for index, (data_v, labels_v) in enumerate(data_loader):
            index += 1
            data_v, labels_v = data_v.to(device), labels_v.to(device)
            logits = model(data_v)
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.cpu()), axis=-1) < args.UNASSIGN_THRES] = -1
            predictions.append(final)
        del data_v, labels_v, logits, final_prob, final
        # gather
        if args.distributed:
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(data_loader.dataset), world_size)
        else:
            predictions = torch.cat(predictions, dim=0)
        no_drop = predictions != -1
        predictions = np.array((predictions[no_drop]).cpu())
    return predictions
