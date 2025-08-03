#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :anno_geneformer_train.py
# @Time      :2025/4/15 17:40
# @Author    :Qiansqian Chen


from transformers.training_args import TrainingArguments
from transformers import Trainer
from biollm.repo.geneformer.collator_for_classification import DataCollatorForCellClassification
from sklearn.metrics import accuracy_score, f1_score
import pickle as pkl
from torch.utils.data import DataLoader
import torch

def train(model, train_set, eval_set, args, label_dict=None, lr_schedule_fn="linear", warmup_steps=500):

    geneformer_batch_size = args.batch_size

    # set logging steps
    logging_steps = round(len(train_set) / geneformer_batch_size / 10)

    training_args = {
        "learning_rate": args.lr if 'lr' in args else 5e-5,
        "do_train": True,
        "do_eval": True,
        "logging_steps": logging_steps,
        "group_by_length": True,
        "prediction_loss_only": True,
        "evaluation_strategy": "epoch",
        "length_column_name": "length",
        "disable_tqdm": False,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "fp16": True,
        "save_total_limit": 1,
        "lr_scheduler_type": lr_schedule_fn,
        "save_strategy": "epoch",
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "num_train_epochs": args.epochs,
        "load_best_model_at_end": True,
        "output_dir": args.output_dir
    }

    training_args_init = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics
    )

    if label_dict is not None:
        with open(f'{args.output_dir}/label_dict.pk', 'wb') as fp:
            pkl.dump(label_dict, fp)
    trainer.train()
    best_model = trainer.model
    return best_model


def predict(model, test_dataset=None, args=None):
    # if test_dataset is None:
    #     eval_set = self.prepare_data(cell_type_key=None)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=DataCollatorForCellClassification(),
                             shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for index, inputs in enumerate(test_loader):
            for i in inputs:
                inputs[i] = inputs[i].to(args.device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = logits.argmax(1)
            predictions.append(preds)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.detach().cpu().numpy()
    return predictions


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }