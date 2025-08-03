#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: cell_annotation.py
@time: 2025/3/27 15:53
"""
from .bio_task import BioTask
import scanpy as sc
import json
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ..trainer import anno_scgpt_train, anno_scbert_train, anno_geneformer_train, anno_scfoundation_train, anno_cellplm_train
from ..evaluate.bm_metrices_anno import compute_metrics
import pickle
from ..algorithm.annotation import ScbertClassification, LinearProbingClassifier
from collections import Counter


class CellAnnotation(BioTask):
    def __init__(self, config_file):
        super(CellAnnotation, self).__init__(config_file)
        self.logger.info(self.args)
        # init the ddp
        self.is_master = int(os.environ['RANK']) == 0 if self.args.distributed else True
        self.args.is_master = self.is_master
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', '0')) if self.args.distributed else int(
                self.args.device.lstrip('cuda:'))
        except ValueError:
            local_rank = 0
        torch.cuda.set_device(local_rank)
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            self.args.device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
        self.args.local_rank = local_rank
        self.world_size = torch.distributed.get_world_size() if self.args.distributed else 1
        if self.is_master:
            self.logger.info(self.args)
        self.args['world_size'] = self.world_size

    def split_adata(self, adata):
        train_adata, val_adata = self.data_handler.split_adata(adata, train_ratio=0.9)
        return train_adata, val_adata

    def split_dataset(self, dataset, nproc=16, train_test_split=0.9):
        def classes_to_ids(example):
            example["label"] = target_dict[example["label"]]
            return example
        def if_trained_label(example):
            return example["label"] in trained_labels
        if self.args.finetune:
            ids = torch.tensor([i for i in range(len(dataset))])
            # Convert the ids tensor to a list
            ids_list = ids.tolist()
            # Add a new column to store the ids
            tokenized_dataset = dataset.add_column("id", ids_list)
            tokenized_dataset = tokenized_dataset.shuffle(seed=42)
            tokenized_dataset = tokenized_dataset.rename_column(self.args.label_key, "label")
            target_names = list(Counter(tokenized_dataset["label"]).keys())
            self.label_dict = target_names
            target_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
            labeled_trainset = tokenized_dataset.map(classes_to_ids, num_proc=nproc)

            # create train/eval splits
            labeled_train_split = labeled_trainset.select([i for i in range(0, round(len(labeled_trainset)*train_test_split))])
            labeled_eval_split = labeled_trainset.select(
                [i for i in range(round(len(labeled_trainset)*train_test_split), len(labeled_trainset))])

            # filter dataset for cell types in corresponding training set
            trained_labels = list(Counter(labeled_train_split["label"]).keys())
            labeled_eval_split = labeled_eval_split.filter(if_trained_label, num_proc=nproc)
            return labeled_train_split, labeled_eval_split

    def get_dataloader(self, adata, obs_key=None, shuffle=False, ddp_train=False, drop_last=False, obs_id_output=None, data_path = None, cell_type_key="celltype", nproc=16, add_length=False, label_dict=None, for_train=False):
        data_loader = None
        if self.args.model_used == 'scgpt':
            data_loader = self.load_obj.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=obs_key,
                                                       n_hvg=self.args.n_hvg, bin_num=self.args.n_bins,
                                                       batch_size=self.args.batch_size,
                                                       obs_id_output=obs_id_output,
                                                       ddp_train=ddp_train, shuffle=shuffle, drop_last=drop_last)
        elif self.args.model_used == 'scbert':
            data_loader = self.load_obj.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=obs_key, n_hvg=0,
                                                       bin_num=self.args.n_bins, batch_size=self.args.batch_size,
                                                       ddp_train=ddp_train, obs_id_output=obs_id_output,
                                                       shuffle=shuffle, drop_last=drop_last)
        elif self.args.model_used == 'geneformer':
            data_loader = self.load_obj.get_dataloader(adata=adata, data_path=data_path, cell_type_key=cell_type_key, nproc=nproc, add_length=add_length)
        elif self.args.model_used == 'scfoundation':
            data_loader = self.load_obj.get_dataloader(adata=adata, label_dict=label_dict, finetune=self.args.finetune,
                                                       label_key=self.args.label_key, for_train=for_train,
                                                       batch_size=self.args.batch_size, ddp_train=ddp_train, shuffle=shuffle,
                                                       drop_last=drop_last)
        return data_loader

    def init_model_for_finetune(self, labels_num):
        if self.args.model_used == 'scgpt':
            self.args.n_cls = labels_num
            self.model = self.load_model()
        if self.args.model_used == 'scbert':
            self.model.to_out = ScbertClassification(h_dim=128,
                                                     class_num=labels_num,
                                                     max_seq_len=self.args.max_seq_len, dropout=0.)
        if self.args.model_used == 'geneformer':
            self.labels_num = labels_num
            self.model = self.load_model()
        if self.args.model_used == 'scfoundation':
            self.model = LinearProbingClassifier(self.load_obj.model, self.load_obj.config, frozenmore=True)
            self.model.build(num_classes=labels_num)
        if self.args.model_used == 'scgpt' or self.args.model_used == 'scbert':
            if self.args.distributed:
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                     output_device=self.args.local_rank, find_unused_parameters=True)
        if self.args.model_used == 'scgpt' or self.args.model_used == 'scbert':
            self.load_obj.freezon_model(keep_layers=[-2])
        self.model = self.model.to(self.args.device)
        return self.model

    def init_model_for_infer(self, labels_num):
        if self.args.model_used == 'scgpt':
            self.args.n_cls = labels_num
            self.model = self.load_model()
        return self.model

    def infer(self, model, data_loader, trainer_module, true_labels, label2id=None, label_dict=None):
        if self.args.model_used != 'scfoundation':
            predictions = trainer_module[self.args.model_used].predict(model, data_loader, self.args)
        else:
            predictions = trainer_module[self.args.model_used].predict(model, data_loader, self.args, config=self.load_obj.config)
        if self.args.model_used != 'geneformer':
            id2label = {v: k for k, v in label2id.items()}
        else:
            id2label = label_dict
        predicted_label = [id2label[i] for i in predictions]
        with open(self.args.output_dir + f'/predict_list.pk', 'wb') as w:
            pickle.dump(predicted_label, w)
        metrics = compute_metrics(true_labels, predicted_label)
        with open(self.args.output_dir + f'/metrics.json', 'w') as w:
            json.dump(metrics, w)
        print("Metrics in test data:", metrics)

    def train(self, adata, trainer_module):
        self.logger.info("start to split data for training...")
        label2id = None
        label_dict = None
        if self.args.model_used == 'scgpt' or self.args.model_used == 'scbert':
            train_adata, val_adata = self.split_adata(adata)
            self.logger.info("start to get train_dataloader")
            train_loader = self.get_dataloader(train_adata, obs_key=self.args.label_key, shuffle=True,
                                               obs_id_output=f"{self.args.output_dir}/label2id.json",
                                               ddp_train=self.args.distributed, drop_last=True)
            with open(f"{self.args.output_dir}/label2id.json", 'r') as f:
                label2id = json.load(f)
                label_num = len(label2id)
            model = self.init_model_for_finetune(label_num)

            self.logger.info("start to get val and test dataloader")
            val_loader = self.get_dataloader(val_adata, obs_key=self.args.label_key, shuffle=False,
                                             ddp_train=self.args.distributed, drop_last=False)

            self.logger.info("start to training...")
            best_model = trainer_module[self.args.model_used].train(model=model, train_loader=train_loader,
                                                                    val_loader=val_loader,
                                                                    args=self.args)
            if self.is_master:
                torch.save(best_model.state_dict(), os.path.join(self.args.output_dir, 'anno_scgpt_best_model.pt'))

        if self.args.model_used == 'geneformer':
            dataset = self.get_dataloader(adata, cell_type_key = self.args.label_key, add_length=True)
            train_dataset, eval_dataset = self.split_dataset(dataset)
            if os.path.exists(f'{self.args.output_dir}/label_dict.pk'):
                with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                    label_list = pickle.load(fp)
                label_num = len(label_list)
            else:
                adata = sc.read_h5ad(self.args.input_file)
                label_num = len(adata.obs[self.args.label_key].unique())
            model = self.init_model_for_finetune(label_num)
            self.logger.info("start to training...")
            best_model = trainer_module[self.args.model_used].train(model=model, train_set=train_dataset,
                                                                    eval_set=eval_dataset,
                                                                    args=self.args, label_dict=self.label_dict)
            with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                label_list = pickle.load(fp)
                label_dict = dict([(i, label_list[i]) for i in range(len(label_list))])

        if self.args.model_used == 'scfoundation':
            self.for_train = True
            train_adata, val_adata = self.split_adata(adata)
            self.logger.info("start to get train_dataloader")
            if os.path.exists(f'{self.args.output_dir}/label_dict.pk'):
                with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                    celltype = pickle.load(fp)
            else:
                celltype = adata.obs[self.args.label_key].unique().tolist()
                with open(f'{self.args.output_dir}/label_dict.pk', 'wb') as fp:
                    pickle.dump(celltype, fp)
            label2id = {celltype[i]: i for i in range(len(celltype))}
            label_num = len(celltype)
            train_loader = self.get_dataloader(adata=train_adata, label_dict=label2id, for_train=self.for_train,
                                               ddp_train=self.args.distributed, shuffle=True,
                                               drop_last=True)
            model = self.init_model_for_finetune(label_num)
            self.logger.info("start to get val and test dataloader")
            eval_loader = self.get_dataloader(adata=val_adata, label_dict=label2id, for_train=self.for_train,
                                              ddp_train=self.args.distributed, shuffle=False,
                                              drop_last=True)
            self.logger.info("start to training...")
            best_model = trainer_module[self.args.model_used].train(model=model, config=self.load_obj.config,
                                                                    train_loader=train_loader, val_loader=eval_loader,
                                                                    args=self.args)
            if self.is_master:
                torch.save(
                    best_model.state_dict(),
                    f"{self.args.output_dir}/model_best.pt",
                )
            if os.path.exists(f'{self.args.output_dir}/model_best.pt'):
                best_model = self.load_obj.load_pretrain_model(f"{self.args.output_dir}/model_best.pt", model)  #看上面训练后生成的模型最佳路径
                best_model = best_model.to(self.args.device)

        if 'test_file' in self.args:
            self.for_train = False
            test_adata = sc.read_h5ad(self.args.test_file)
            if self.args.model_used == 'scgpt' or self.args.model_used == 'scbert':
                test_loader = self.get_dataloader(test_adata, self.args.label_key, shuffle=False,
                                                  ddp_train=self.args.distributed, drop_last=False)
            elif self.args.model_used == 'geneformer':
                test_loader = self.get_dataloader(test_adata, cell_type_key=None, add_length=False)
            elif self.args.model_used == 'scfoundation':
                test_loader = self.get_dataloader(adata=test_adata, label_dict=label2id, for_train=self.for_train,
                                                  ddp_train=False, shuffle=False, drop_last=False)
            true_labels = test_adata.obs[self.args.label_key]

            self.infer(best_model, test_loader, trainer_module, true_labels, label2id, label_dict)

    def cellplm_train(self, adata):
        pipeline = anno_cellplm_train.train(adata, self.args)
        if 'test_file' in self.args:
            test_adata = sc.read_h5ad(self.args.test_file)
            anno_cellplm_train.predict(pipeline, test_adata, self.args)

    def run(self):
        adata = sc.read_h5ad(self.args.input_file)
        trainer_module = {
            'scgpt': anno_scgpt_train,
            'scbert': anno_scbert_train,
            'geneformer': anno_geneformer_train,
            'scfoundation': anno_scfoundation_train
        }
        if self.args.finetune:
            if self.args.model_used == 'cellplm':
                self.cellplm_train(adata)
            else:
                self.train(adata, trainer_module)
        else:
            label2id = None
            label_dict = None
            if self.args.model_used == 'scgpt' or self.args.model_used == 'scbert':
                loader = self.get_dataloader(adata, self.args.label_key, shuffle=False,
                                             ddp_train=self.args.distributed, drop_last=False)
                with open(f"{self.args.output_dir}/label2id.json", 'r') as f:
                    label2id = json.load(f)
                    label_num = len(label2id)
            elif self.args.model_used == 'geneformer':
                loader = self.get_dataloader(adata, cell_type_key=None)
                with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                    label_list = pickle.load(fp)
                    label_num = len(label_list)
                    label_dict = dict([(i, label_list[i]) for i in range(len(label_list))])
                print(label_num)
            elif self.args.model_used == 'scfoundation':
                with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                    celltype = pickle.load(fp)
                label2id = {celltype[i]: i for i in range(len(celltype))}
                label_num = len(celltype)
                loader = self.get_dataloader(adata=adata, label_dict=label2id,
                                             ddp_train=False,
                                             shuffle=False,
                                             drop_last=False)
            if self.args.model_used != 'scfoundation':
                best_model = self.init_model_for_finetune(label_num)
            else:
                model = self.init_model_for_finetune(label_num)
                best_model = self.load_obj.load_pretrain_model(f"{self.args.output_dir}/model_best.pt", model)
                best_model = best_model.to(self.args.device)
            true_labels = adata.obs[self.args.label_key]
            self.infer(best_model, loader, trainer_module, true_labels, label2id, label_dict)


if __name__ == '__main__':
    config_file = '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM1/biollm/config/anno/cellplm_ms.toml'
    obj = CellAnnotation(config_file)
    obj.run()
