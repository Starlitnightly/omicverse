#!/usr/bin/env python3
# coding: utf-8
"""
@file: trainer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/27  create file.
"""
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from biollm.repo.st_performer.model.st_performer import StPerformer, StPerformerLM, StPerformerCLS
from utils import get_reduced, dist_cat_tensors, data_mask
from biollm.repo.st_performer.model.learn_rate import CosineAnnealingWarmupRestarts
from collections import OrderedDict
from sklearn.metrics import classification_report, f1_score
import pickle
import torch.distributed as dist
from st_performer.utils import get_logger, count_parameters


class Trainer:
    def __init__(self, args, device, train_loader, test_loader, gene_tokenizer, organ_tokenizer=None,
                 disease_tokenizer=None, sequence_tokenizer=None):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gene_tokenizer = gene_tokenizer
        self.organ_tokenizer = organ_tokenizer
        self.sequence_tokenizer = sequence_tokenizer
        self.disease_tokenizer = disease_tokenizer
        self.gene_vocab_size = gene_tokenizer.vocab_size  # gene vocab size
        self.gene_pad_id = gene_tokenizer.pad_token_id
        self.gene_mask_id = gene_tokenizer.mask_token_id
        self.device = device
        self.logger = get_logger()
        self.is_master = int(os.environ['RANK']) == 0 if args.distributed else True
        self.writer = SummaryWriter() if self.is_master else None
        assert args.pretrain != args.finetune
        if self.is_master:
            self.logger.info(args)
        # self.performer = StPerformer(gene_num=self.vocab_size, gene_pad_idx=self.pad_id, organ_num=self.args.organ_num,
        #                              max_seq_len=self.args.max_seq_len, dim=self.args.dim, depth=self.args.depth,
        #                              heads=self.args.heads, g2v_position_emb=False, g2v_file=None) # 策略一：表达量加权，预测真实表达值
        self.performer = StPerformer(gene_num=self.gene_vocab_size,
                                     gene_pad_idx=self.gene_pad_id,
                                     max_seq_len=self.args.max_seq_len,
                                     dim=self.args.dim,
                                     depth=self.args.depth,
                                     heads=self.args.heads,
                                     is_exp_emb=args.is_exp_emb,
                                     exp_bins=args.bin_num + 2,
                                     is_sequence_emb=args.is_sequence_emb,
                                     sequence_num=self.sequence_tokenizer.vocab_size,
                                     is_organ_emb=self.args.is_organ_emb,
                                     organ_num=self.organ_tokenizer.vocab_size,
                                     g2v_position_emb=self.args.g2v_position_emb,
                                     g2v_file=args.gene2vec_file,
                                     )
        self.exp_bin_pad_id = args.bin_num + 1
        self.exp_bin_mask_id = args.bin_num + 1
        if args.pretrained_model:
            ckpt = torch.load(args.pretrained_model)['model_state_dict']
            rename_ckpt = OrderedDict()
            for i in ckpt:
                if i.startswith('model'):
                    rename_ckpt[i.split('model.')[-1]] = ckpt[i]
            self.performer.load_state_dict(rename_ckpt)
        if args.pretrain:
            gene_tokens_num = self.gene_vocab_size
            exp_bins_num = args.bin_num + 2
            disease_tokens_num = self.disease_tokenizer.vocab_size
            sequence_tokens_num = self.sequence_tokenizer.vocab_size
            self.model = StPerformerLM(self.performer, gene_tokens_num, exp_bins_num, disease_tokens_num,
                                       sequence_tokens_num, args.batch_tokens_num, args.is_st)
        else:
            self.frozen()
            if args.finetune:
                with open(f'./data/label_dict.{args.model_name}', 'rb') as f:
                    self.cls_label = pickle.load(f).tolist()
            self.model = StPerformerCLS(self.performer, len(self.cls_label), gene_tokenizer.cls_token_id, max_seq_len=args.max_seq_len)

        self.model.to(self.device)
        # cal_model_params(self.model, ((8, 3114, 200), (8, 3114, 200)))
        if args.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                 output_device=args.local_rank, find_unused_parameters=True)
        # self.optimizer = RAdam(self.model.parameters(), args.lr)
        self.optimizer = Adam(self.model.parameters(), args.lr, eps=1e-4)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=args.lr,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        self.scaler = torch.cuda.amp.GradScaler()
        # self.l1_criterion = nn.SmoothL1Loss().to(self.device) # 策略一，预测表达值
        # self.cls_criterion = nn.CrossEntropyLoss(ignore_index=self.exp_bin_pad_id).to(self.device)
        self.cls_criterion = nn.CrossEntropyLoss().to(self.device)
        # self.cls_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id).to(self.device)
        params_num = count_parameters(self.model)
        if self.is_master:
            self.logger.info('Model params: {}'.format(params_num))

    def frozen(self):
        for param in self.performer.parameters():
            param.requires_grad = False
        for param in self.performer.norm.parameters():
            param.requires_grad = True
        for param in self.performer.performer.net.layers[-2].parameters():
            param.requires_grad = True

    def mask_input(self, gene_x, exp_x):
        exp_x[exp_x > self.args.bin_num] = self.args.bin_num
        exp_x = exp_x.long()
        mask_exp_x_non_zero, mask_exp_labels = data_mask(exp_x, self.args.mask_non_zero_prob,
                                                         self.args.keep_replace_prob,
                                                         self.args.bin_num + 1, self.args.random_replace_prob,
                                                         self.exp_bin_mask_id,
                                                         self.exp_bin_pad_id, mask_ignore_token_ids=[0])
        mask_exp_x, mask_exp_labels_zero = data_mask(mask_exp_x_non_zero, self.args.mask_zero_prob,
                                                     self.args.keep_replace_prob,
                                                     self.args.bin_num + 1, self.args.random_replace_prob,
                                                     self.exp_bin_mask_id,
                                                     self.exp_bin_pad_id,
                                                     mask_ignore_token_ids=list(range(1, self.args.bin_num + 2)))
        mask_exp_labels[mask_exp_labels_zero != self.exp_bin_pad_id] = mask_exp_labels_zero[
            mask_exp_labels_zero != self.exp_bin_pad_id]
        pad_index = (mask_exp_labels == self.exp_bin_pad_id)
        mask_gene_labels = gene_x.masked_fill(pad_index, self.gene_pad_id)
        gene_x[~pad_index] = self.gene_mask_id
        mask_gene_x = gene_x
        return mask_gene_x, mask_gene_labels, mask_exp_x, mask_exp_labels

    def pretrain(self, epoch):
        mask_gene_sum = 0
        mask_gene_true_sum = 0
        mask_exp_bin_sum = 0
        mask_exp_bin_true_sum = 0
        neighbor_sum = 0
        neighbor_true_sum = 0
        batch_sum = 0
        batch_true_sum = 0
        sequence_sum = 0
        sequence_true_sum = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        if self.is_master:
            print(f'sample number: {n_samples}, n_batches: {n_batches}')
        self.model.train()
        ecpoch_loss = 0
        for i, batch in enumerate(self.train_loader):
            losses = 0
            gene_x = batch['gene_id'].to(self.device)
            exp_x = batch['express_x'].to(self.device)
            organ_x = torch.tensor(self.organ_tokenizer.convert_tokens_to_ids(batch['organ'])).to(self.device)
            sequence_x = torch.tensor(self.sequence_tokenizer.convert_tokens_to_ids(batch['sequence'])).to(self.device)
            disease_x = torch.tensor(self.sequence_tokenizer.convert_tokens_to_ids(batch['disease'])).to(self.device)
            batch_x = batch['batch_id'].to(self.device)
            gene_x, mask_gene_labels, exp_x, mask_exp_labels = self.mask_input(gene_x, exp_x)
            # mask_labels: [batch_size, max_seq_len]
            exp_gt_index = mask_exp_labels != self.exp_bin_pad_id
            gene_gt_index = mask_gene_labels != self.gene_pad_id
            with torch.cuda.amp.autocast():
                mask_logits, neighbor_logits, _ = self.model(gene_x, exp_x, organ_x, sequence_x)
                if 'gene_logit' in mask_logits:
                    mask_gene_loss = self.cls_criterion(mask_logits['gene_logit'][gene_gt_index],
                                                        mask_gene_labels[gene_gt_index])
                    losses += mask_gene_loss
                    mask_tokens = gene_gt_index.sum().item()
                    mask_gene_sum += mask_tokens
                    final = mask_logits['gene_logit'].argmax(dim=-1)
                    correct_num = (final == mask_gene_labels)[gene_gt_index].sum()
                    mask_gene_true_sum += correct_num.item()
                    if self.is_master:
                        self.writer.add_scalar('Loss/Gene', mask_gene_loss.item(), ((epoch - 1) * n_batches) + i)
                        if i % 100 == 0 and i != 0:
                            self.logger.info(
                                'Loss/Gene: {:.4f}, mask tokens: {}, acc: {:.4f}'.format(mask_gene_loss.item(),
                                                                                         mask_tokens,
                                                                                         correct_num.item() / mask_tokens))
                if 'exp_bin_logit' in mask_logits:
                    mask_exp_bin_loss = self.cls_criterion(mask_logits['exp_bin_logit'][exp_gt_index],
                                                           mask_exp_labels[exp_gt_index])
                    losses += mask_exp_bin_loss
                    mask_exp_bin_sum += exp_gt_index.sum().item()
                    final = mask_logits['exp_bin_logit'].argmax(dim=-1)
                    correct_num = (final == mask_exp_labels)[exp_gt_index].sum()
                    mask_exp_bin_true_sum += correct_num.item()
                    if self.is_master:
                        self.writer.add_scalar('Loss/exp_bin', mask_exp_bin_loss.item(), ((epoch - 1) * n_batches) + i)
                        if i % 100 == 0 and i != 0:
                            self.logger.info('Loss/exp_bin: {:.4f}, acc: {:.4f}'.format(mask_exp_bin_loss.item(),
                                                                                        correct_num / exp_gt_index.sum().item()))
                if 'disease_logit' in mask_logits:
                    disease_loss = self.cls_criterion(mask_logits['disease_logi'], disease_x)
                    losses += disease_loss
                    if self.is_master:
                        self.writer.add_scalar('Loss/disease', disease_loss.item(), ((epoch - 1) * n_batches) + i)
                        if i % 100 == 0 and i != 0:
                            self.logger.info('Loss/disease: {:.4f}'.format(disease_loss.item()))
                if 'sequence_logit' in mask_logits:
                    sequence_loss = self.cls_criterion(mask_logits['sequence_logit'], sequence_x)
                    losses += sequence_loss
                    sequence_sum += sequence_x.shape[0]
                    final = mask_logits['sequence_logit'].argmax(dim=-1)
                    correct_num = (final == sequence_x).sum().item()
                    sequence_true_sum += correct_num
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/sequence', sequence_loss.item(),
                                               ((epoch - 1) * n_batches) + i)
                        if i % 100 == 0 and i != 0:
                            self.logger.info('Loss/sequence: {:.4f}'.format(sequence_loss.item()))
                if 'batch_logit' in mask_logits:
                    batch_loss = self.cls_criterion(mask_logits['batch_logit'], batch_x)
                    losses += batch_loss
                    batch_sum += batch_x.shape[0]
                    final = mask_logits['batch_logit'].argmax(dim=-1)
                    batch_true_sum += (final == batch_x).sum().item()
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/batch', batch_loss.item(), ((epoch - 1) * n_batches) + i)
                        if i % 100 == 0 and i != 0:
                            self.logger.info('Loss/batch: {:.4f}'.format(batch_loss.item()))
            self.optimizer.zero_grad()
            ecpoch_loss += losses.item()
            # losses.backward()
            # self.optimizer.step()
            self.scaler.scale(losses).backward()
            self.scaler.unscale_(self.optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0,
                                               error_if_nonfinite=False if self.scaler.is_enabled() else True)
                if len(w) > 0:
                    self.logger.info("warning: find infinite gradient. This may be caused by the "
                                     "gradient scaler. The scaler is {}".format(self.scaler.get_scale()))
            self.scaler.step(optimizer=self.optimizer)
            self.scaler.update()
            if self.is_master:
                self.writer.add_scalar('Loss/all', losses.item(), ((epoch - 1) * n_batches) + i)
                if i % 100 == 0 and i != 0:
                    self.logger.info('Iteration {} ({}/{})\tlosses: {:.4f}, mask genes sum: {}'.
                                     format(i, i, n_batches, ecpoch_loss / i, gene_gt_index.sum().item()))
        if self.args.distributed:
            mask_gene_sum += get_reduced(mask_gene_sum, self.device, 0)
            mask_gene_true_sum += get_reduced(mask_gene_true_sum, self.device, 0)
            mask_exp_bin_true_sum += get_reduced(mask_exp_bin_true_sum, self.device, 0)
            batch_true_sum += get_reduced(batch_true_sum, self.device, 0)
            batch_sum += get_reduced(batch_sum, self.device, 0)
            sequence_true_sum += get_reduced(sequence_true_sum, self.device, 0)

            if self.args.is_st:
                neighbor_sum += get_reduced(neighbor_sum, self.device, 0)
                neighbor_true_sum += get_reduced(neighbor_true_sum, self.device, 0)
            ecpoch_loss += get_reduced(ecpoch_loss, self.device, 0)
        if self.is_master:
            mask_gene_acc = 100 * mask_gene_true_sum / mask_gene_sum
            mask_exp_bin_acc = 100 * mask_exp_bin_true_sum / mask_gene_sum
            batch_acc = 100 * batch_true_sum / batch_sum
            sequence_acc = 100 * sequence_true_sum / batch_sum
            neighbor_acc = 100 * neighbor_true_sum / neighbor_sum if self.args.is_st else 0
            self.writer.add_scalar('Epoch Loss', ecpoch_loss / n_batches, epoch)
            self.logger.info(
                'Train Epoch {} [rank: {}] > Loss: {:.4f}; Mask Gene Acc: {:.2f}%;'
                ' Mask ExpBin Acc: {:.2f}%; neighbor Acc: {:.2f}%; Batch Acc: {:.2f}%; Sequence Acc: {:.2f}%'.format(
                    epoch, self.args.local_rank, ecpoch_loss / n_batches, mask_gene_acc, mask_exp_bin_acc,
                    neighbor_acc, batch_acc, sequence_acc))
        self.scheduler.step()

    def evaluate(self, epoch):
        mask_gene_sum = 0
        mask_gene_true_sum = 0
        mask_exp_bin_sum = 0
        mask_exp_bin_true_sum = 0
        neighbor_sum = 0
        neighbor_true_sum = 0
        ecpoch_loss = 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                losses = 0
                gene_x = batch['gene_index'].to(self.device)
                exp_x = batch['express_x'].to(self.device)
                organ_x = batch['organ'].to(self.device)
                sequence_x = batch['sequence'].to(self.device)
                disease_x = batch['disease'].to(self.device)
                batch_x = batch['batch_id'].to(self.device)
                mask_exp_labels = batch['mask_exp_labels'].to(self.device)
                mask_gene_labels = batch['mask_gene_labels'].to(self.device)
                # mask_labels: [batch_size, max_seq_len]
                exp_gt_index = mask_exp_labels != self.exp_bin_pad_id
                gene_gt_index = mask_gene_labels != self.gene_pad_id
                mask_logits, neighbor_logits, _ = self.model(gene_x, exp_x, organ_x, sequence_x)
                if 'gene_logit' in mask_logits:
                    mask_gene_loss = self.cls_criterion(mask_logits['gene_logit'][gene_gt_index],
                                                        mask_gene_labels[gene_gt_index])
                    losses += mask_gene_loss
                    mask_gene_sum += gene_gt_index.sum().item()
                    final = mask_logits['gene_logit'].argmax(dim=-1)
                    correct_num = (final == mask_gene_labels)[gene_gt_index].sum()
                    mask_gene_true_sum += correct_num.item()
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/Gene', mask_gene_loss.item(), ((epoch - 1) * n_batches) + i)
                if 'exp_bin_logit' in mask_logits:
                    mask_exp_bin_loss = self.cls_criterion(mask_logits['exp_bin_logit'][exp_gt_index],
                                                           mask_exp_labels[exp_gt_index])
                    losses += mask_exp_bin_loss
                    mask_exp_bin_sum += exp_gt_index.sum().item()
                    final = mask_logits['exp_bin_logit'].argmax(dim=-1)
                    correct_num = (final == mask_exp_labels)[exp_gt_index].sum()
                    mask_exp_bin_true_sum += correct_num.item()
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/exp_bin', mask_exp_bin_loss.item(),
                                               ((epoch - 1) * n_batches) + i)
                if 'disease_logit' in mask_logits:
                    disease_loss = self.cls_criterion(mask_logits['disease_logit'], disease_x)
                    losses += disease_loss
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/disease', disease_loss.item(), ((epoch - 1) * n_batches) + i)
                if 'sequence_logit' in mask_logits:
                    sequence_loss = self.cls_criterion(mask_logits['sequence_logit'], sequence_x)
                    losses += sequence_loss
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/sequence', sequence_loss.item(),
                                               ((epoch - 1) * n_batches) + i)
                if 'batch_logit' in mask_logits:
                    batch_loss = self.cls_criterion(mask_logits['batch_logit'], batch_x)
                    losses += batch_loss
                    if self.is_master:
                        self.writer.add_scalar('Eval Loss/batch', batch_loss.item(), ((epoch - 1) * n_batches) + i)
                ecpoch_loss += losses.item()
                if self.is_master:
                    self.writer.add_scalar('Eval Loss/all', losses.item(), ((epoch - 1) * n_batches) + i)
            if self.args.distributed:
                mask_gene_sum += get_reduced(mask_gene_sum, self.device, 0)
                mask_gene_true_sum += get_reduced(mask_gene_true_sum, self.device, 0)
                mask_exp_bin_true_sum += get_reduced(mask_exp_bin_true_sum, self.device, 0)
                if self.args.is_st:
                    neighbor_sum += get_reduced(neighbor_sum, self.device, 0)
                    neighbor_true_sum += get_reduced(neighbor_true_sum, self.device, 0)
                ecpoch_loss += get_reduced(ecpoch_loss, self.device, 0)
            if self.is_master:
                mask_gene_acc = 100 * mask_gene_true_sum / mask_gene_sum
                mask_exp_bin_acc = 100 * mask_exp_bin_true_sum / mask_gene_sum
                neighbor_acc = 100 * neighbor_true_sum / neighbor_sum if self.args.is_st else 0
                self.writer('Eval Epoch Loss', ecpoch_loss / n_batches, epoch)
                self.logger.info(
                    'Eval Epoch {} [rank: {}] > Loss: {:.4f} / Mask Gene Acc: {:.2f}%;'
                    ' Mask Gene Acc: {:.2f}%; neighbor Acc: {}'.format(
                        epoch, self.args.local_rank, ecpoch_loss / n_batches, mask_gene_acc, mask_exp_bin_acc,
                        neighbor_acc))

    def finetune(self, epoch):
        losses = 0
        sample_sum = 0
        sample_true_sum = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            gene_x, exp_x, cls_labels = batch
            gene_x = gene_x.to(self.device)
            exp_x = exp_x.to(self.device)
            cls_labels = cls_labels.to(self.device)
            # cls_labels: [batch_size]
            cls_logits, _ = self.model(gene_x, exp_x)
            loss = self.cls_criterion(cls_logits, cls_labels)
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            sample_sum += cls_labels.shape[0]
            sample_true_sum += (cls_logits.argmax(dim=-1) == cls_labels).sum().item()
            if self.is_master:
                # self.writer.add_scalar('Loss/finetune', loss.item(), ((epoch - 1) * n_batches) + i)
                if i % 100 == 0 and i != 0:
                    self.writer.add_scalar('Loss/finetune', loss.item(), ((epoch - 1) * n_batches) + i)
                    print('Iteration {} ({}/{})\tLoss: {:.4f}'.format(i, i, n_batches, losses / i))
        if self.args.distributed:
            sample_sum += get_reduced(sample_sum, self.device, 0)
            sample_true_sum += get_reduced(sample_true_sum, self.device, 0)
            losses += get_reduced(losses, self.device, 0)
        if self.is_master:
            acc = 100 * sample_true_sum / sample_sum
            self.writer.add_scalar('Loss epoch/train', losses / n_batches, epoch)
            self.writer.add_scalar('Acc/train', acc, epoch)
            print('Train Epoch {} [rank: {}] > Loss: {:.4f} / Acc: {:.2f}%'
                  .format(epoch, self.args.local_rank, losses / n_batches, acc))
        dist.barrier()
        self.scheduler.step()

    def val_finetune(self, epoch):
        losses = 0
        sample_sum = 0
        sample_true_sum = 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        predictions = []
        truths = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                gene_x, exp_x, cls_labels = batch
                gene_x = gene_x.to(self.device)
                exp_x = exp_x.to(self.device)
                cls_labels = cls_labels.to(self.device)
                # cls_labels: [batch_size]
                cls_logits, _ = self.model(gene_x, exp_x)
                loss = self.cls_criterion(cls_logits, cls_labels)
                losses += loss.item()
                sample_sum += cls_labels.shape[0]
                sample_true_sum += (cls_logits.argmax(dim=-1) == cls_labels).sum().item()
                predict_labels = cls_logits.argmax(dim=-1)
                predictions.append(predict_labels)
                truths.append(cls_labels)
                if self.is_master:
                    self.writer.add_scalar('Loss/finetune Eval', loss.item(), ((epoch - 1) * n_batches) + i)

            if self.args.distributed:
                sample_sum += get_reduced(sample_sum, self.device, 0)
                sample_true_sum += get_reduced(sample_true_sum, self.device, 0)
                losses += get_reduced(losses, self.device, 0)
                predictions = dist_cat_tensors(torch.cat(predictions, dim=0))
                truths = dist_cat_tensors(torch.cat(truths, dim=0))
            if self.is_master:
                acc = 100 * sample_true_sum / sample_sum
                print('Eval  Epoch {} [rank: {}] > Loss: {:.4f} / Acc: {:.2f}%'
                      .format(epoch, self.args.local_rank, losses / n_batches, acc))
                predictions = np.array(predictions.cpu())
                truths = np.array(truths.cpu())
                self.writer.add_scalar('Loss epoch/eval', losses / n_batches, epoch)
                self.writer.add_scalar('Acc/eval', acc, epoch)
                f1 = f1_score(truths, predictions, average='macro')
                self.writer.add_scalar('F1(macro)/eval', f1, epoch)
                print(classification_report(truths, predictions, target_names=self.cls_label, digits=4))

    def pretrain_bak(self, epoch):
        losses = 0
        mask_sum = 0
        mask_true_sum = 0
        neighbor_sum = 0
        neighbor_true_sum = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            gene_x, exp_x, mask_labels = batch
            # gene_x, exp_x, organ_x, mask_labels, neighbor_lables = batch

            if self.args.is_st:
                exp_x = torch.cat((exp_x[:, 0, :], exp_x[:, 1, :]), 0)
                mask_labels = torch.cat((mask_labels[:, 0, :], mask_labels[:, 1, :]), 0)
            # mask_labels: [batch_size, max_seq_len]
            # neighbor_labels: [batch_size]
            gt_index = mask_labels != self.exp_bin_pad_id
            # gt_index = mask_labels != self.tokenizer.pad_token_id
            mask_logits, neighbor_logits, _ = self.model(gene_x, exp_x)
            # mask_logits, neighbor_logits, _ = self.model(gene_x, exp_x, organ_x, self.args.is_st)
            mask_loss = self.cls_criterion(mask_logits[gt_index], mask_labels[gt_index])
            # mask_loss = self.l1_criterion(mask_logits[gt_index], mask_labels[gt_index]) # 策略一
            loss = mask_loss
            # if self.args.is_st:
            #     neighbor_loss = self.cls_criterion(neighbor_logits, neighbor_lables)
            #     loss += neighbor_loss
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mask_sum += gt_index.sum().item()
            mask_exp_diff = torch.abs(mask_logits[gt_index] - mask_labels[gt_index])
            mask_true_sum += (mask_exp_diff < 0.001).sum().item()
            if self.args.is_st:
                neighbor_sum += neighbor_lables.shape[0]
                neighbor_true_sum += (neighbor_logits.argmax(dim=-1) == neighbor_lables).sum().item()
            if self.is_master:
                self.writer.add_scalar('Loss/pre-train', loss.item(), ((epoch - 1) * n_batches) + i)
                if i % 10 == 0 and i != 0:
                    if self.args.is_st:
                        print(
                            'Iteration {} ({}/{})\tmask loss: {:.4f}, neighbor loss: {:.4f}, mask genes sum: {} '.format(
                                i, i, n_batches,
                                mask_loss.item(),
                                neighbor_loss.item(), gt_index.sum().item()))
                    else:
                        print('Iteration {} ({}/{})\tmask loss: {:.4f}, mask genes sum: {}'.format(i, i, n_batches,
                                                                                                   mask_loss.item(),
                                                                                                   gt_index.sum().item()))
                    # print('Iteration {} ({}/{})\tLoss: {:.4f}'.format(i, i, n_batches, losses / i))
        if self.args.distributed:
            mask_sum += get_reduced(mask_sum, self.device, 0)
            mask_true_sum += get_reduced(mask_true_sum, self.device, 0)
            if self.args.is_st:
                neighbor_sum += get_reduced(neighbor_sum, self.device, 0)
                neighbor_true_sum += get_reduced(neighbor_true_sum, self.device, 0)
            losses += get_reduced(losses, self.device, 0)
        if self.is_master:
            mask_acc = 100 * mask_true_sum / mask_sum
            neighbor_acc = 100 * neighbor_true_sum / neighbor_sum if self.args.is_st else 0
            print('Epoch {} [rank: {}] > Loss: {:.4f} / Mask Acc: {:.2f}%; neighbor Acc: {}'
                  .format(epoch, self.args.local_rank, losses / n_batches, mask_acc, neighbor_acc))
        self.scheduler.step()

    def finetune_bak(self, epoch):
        losses = 0
        sample_sum = 0
        sample_true_sum = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        print('sample number: ', n_samples)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            gene_x, exp_x, organ_x, cls_labels = batch
            # cls_labels: [batch_size]
            cls_logits, _ = self.model(gene_x, exp_x, organ_x, self.args.is_st)
            loss = self.cls_criterion(cls_logits, cls_labels)
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sample_sum += cls_labels.shape[0]
            sample_true_sum += (cls_logits.argmax(dim=-1) == cls_labels).sum().item()
            if self.is_master:
                self.writer.add_scalar('Loss/finetune', loss.item(), ((epoch - 1) * n_batches) + i)
                if i % 20 == 0 and i != 0:
                    print('Iteration {} ({}/{})\tLoss: {:.4f}'.format(i, i, n_batches, losses / i))
        if self.args.distributed:
            sample_sum += get_reduced(sample_sum, self.device, 0)
            sample_true_sum += get_reduced(sample_true_sum, self.device, 0)
            losses += get_reduced(losses, self.device, 0)
        if self.is_master:
            acc = 100 * sample_true_sum / sample_sum
            print('Epoch {} [rank: {}] > Loss: {:.4f} / Acc: {:.2f}%'
                  .format(epoch, self.args.local_rank, losses / n_batches, acc))

    def evaluate_bak(self, epoch):
        losses = 0
        mask_sum = 0
        mask_true_sum = 0
        neighbor_sum = 0
        neighbor_true_sum = 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                gene_x, exp_x, organ_x, mask_labels, neighbor_lables = batch
                # mask_labels: [batch_size, max_seq_len]
                # neighbor_labels: [batch_size]
                mask_logits, neighbor_logits, _ = self.model(gene_x, exp_x, organ_x, self.args.is_st)
                mask_loss = self.cls_criterion(mask_logits, mask_labels)
                neighbor_loss = self.cls_criterion(neighbor_logits, neighbor_lables)
                loss = mask_loss + neighbor_loss
                losses += loss.item()
                gt_index = mask_labels != self.exp_bin_pad_id
                mask_sum += gt_index.sum().item()
                mask_true_sum += (mask_logits.argmax(dim=-1) == mask_labels)[gt_index].sum().item()
                if self.args.is_st:
                    neighbor_sum += neighbor_lables.shape[0]
                    neighbor_true_sum += (neighbor_logits.argmax(dim=-1) == neighbor_lables).sum().item()
                if self.is_master:
                    self.writer.add_scalar('Loss/pre-train(eval)', loss.item(), ((epoch - 1) * n_batches) + i)
            if self.args.distributed:
                mask_sum += get_reduced(mask_sum, self.device, 0)
                mask_true_sum += get_reduced(mask_true_sum, self.device, 0)
                if self.args.is_st:
                    neighbor_sum += get_reduced(neighbor_sum, self.device, 0)
                    neighbor_true_sum += get_reduced(neighbor_true_sum, self.device, 0)
                losses += get_reduced(losses, self.device, 0)
            if self.is_master:
                mask_acc = 100 * mask_true_sum / mask_sum
                neighbor_acc = 100 * neighbor_true_sum / neighbor_sum if self.args.is_st else 0
                print('Eval Epoch {} [rank: {}] > Loss: {:.4f} / Mask Acc: {:.2f}%; neighbor Acc: {}'
                      .format(epoch, self.args.local_rank, losses / n_batches, mask_acc, neighbor_acc))
