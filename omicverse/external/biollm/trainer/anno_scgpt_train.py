#!/usr/bin/env python3
# coding: utf-8

import warnings
import torch

from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist
from biollm.utils.utils import get_reduced


def train(model, train_loader, val_loader, args, wandb=None):
    """
    Train the model for one epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.schedule_interval, gamma=args.schedule_ratio
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_master = args.is_master
    best_val_loss = float("inf")
    best_model = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=None)
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        total_acc = 0.0
        epoch_loss = 0
        epoch_acc = 0
        num_batches = len(train_loader)
        for batch, batch_data in enumerate(train_loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(args.pad_value)
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                loss = 0.0
                loss_cls = loss_fn(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                acc = (output_dict["cls_output"].argmax(1) == celltype_labels).sum().item() / celltype_labels.size(0)
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
                    if is_master:
                        print(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_acc += acc
            epoch_loss += loss.item()
            epoch_acc += acc
            if batch % args.log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / args.log_interval
                cur_acc = total_acc / args.log_interval
                if args.distributed:
                    cur_loss = get_reduced(cur_loss, device, 0)
                    cur_acc = get_reduced(cur_acc, device, 0)
                if is_master:
                    print(
                        f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"lr {lr:5.5f} | "
                        f"acc {cur_acc:5.2f} | "
                        f"loss {cur_loss:5.2f} | "

                    )
                total_loss = 0
                total_acc = 0
        if args.distributed:
            epoch_loss = get_reduced(epoch_loss, device, 0)
            epoch_acc = get_reduced(epoch_acc, device, 0)

        if is_master:
            if wandb:
                wandb.log({
                    "train/loss": epoch_loss / num_batches,
                    "train/acc": epoch_acc / num_batches,
                })
            print({
                "train/loss": epoch_loss / num_batches,
                "train/acc": epoch_acc / num_batches,
            })

        if args.distributed:
            dist.barrier()
        val_loss = evaluate(epoch, model, val_loader, loss_fn, args, wandb=wandb, return_pred=False)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            if is_master:
                print(f"Best model with score {best_val_loss:5.4f}")
        scheduler.step()
        if is_master:
            torch.save(
                model.state_dict(),
                f"{args.output_dir}/model_{epoch}.pt",
            )
    return best_model


def evaluate(epoch, model, val_loader, loss_fn, args, wandb=None, return_pred=False):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_num = 0
    predictions = []
    trues = []
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_master = args.is_master
    with torch.no_grad():
        for batch_data in val_loader:
            input_gene_ids = batch_data["gene_ids"].to(args.device)
            input_values = batch_data["values"].to(args.device)
            celltype_labels = batch_data["celltype_labels"].to(args.device)
            src_key_padding_mask = input_gene_ids.eq(args.pad_value)
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                output_values = output_dict["cls_output"]
                loss = loss_fn(output_values, celltype_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(dim=-1)
            predictions.append(preds)
            trues.append(celltype_labels)

        if args.distributed:
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_loader.dataset), args.world_size)
            trues = distributed_concat(torch.cat(trues, dim=0), len(val_loader.dataset), args.world_size)
        else:
            predictions = torch.cat(predictions, dim=0)
            trues = torch.cat(trues, dim=0)
    predictions = predictions.cpu().numpy()
    trues = trues.cpu().numpy()
    accuracy = accuracy_score(trues, predictions)
    f1 = f1_score(trues, predictions, average='macro')
    if args.distributed:
        accuracy = get_reduced(accuracy, device, 0)
        f1 = get_reduced(f1, device, 0)
    if is_master:
        print(
            f"| end of epoch {epoch:3d} | "
            f"valid loss/mse {total_loss/total_num:5.4f} "
            f'accuracy: {accuracy:5.3f}, f1-score: {f1:5.3f}')
        if wandb:
            wandb.log({
                'eval/loss': total_loss/total_num,
                'eval/acc': accuracy,
                'eval/f1': f1
            })
    if return_pred:
        return predictions
    return total_loss/total_num


def predict(model, data_loader, args):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in data_loader:
            input_gene_ids = batch_data["gene_ids"].to(args.device)
            input_values = batch_data["values"].to(args.device)
            src_key_padding_mask = input_gene_ids.eq(args.pad_value)
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                output_values = output_dict["cls_output"]
            preds = output_values.argmax(1)
            predictions.append(preds)
        predictions = torch.cat(predictions, dim=0)
    predictions = predictions.detach().cpu().numpy()
    return predictions


def distributed_concat(tensor, num_total_examples, world_size):
    """
    合并不同进程的inference结果
    """
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]