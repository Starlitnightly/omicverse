 # -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:KGbased_finetune.py
# @Software:PyCharm
# @Created Time:2023/11/20 5:31 PM
# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
import sys
sys.path.append('/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/')
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from KG.GRL import HeteroGAT
import dgl
import wandb

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=False, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/scBERT/data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/scBERT/ckpts/panglao_pretrain.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/scBERT/ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='Debug', help='Finetuned model name.')
parser.add_argument("--distributed", type=bool, default=False, help='debug mode, single gpu device')

parser.add_argument("--graph_feat", type=str, default='kg',choices=['gene2vec','kg'], help='the source of graph feature')
parser.add_argument("--gnn_conv", type=str, default='GAT', choices=['GAT','SAGE'],help='convolution layer of gnn')
parser.add_argument("--graph_path", type=str, default='/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/data', help='path of graph')
parser.add_argument("--graph_data", type=str, default='Public_DB_1115', help='dataset name')
parser.add_argument("--num_head", type=int, default=2, help='number of attention head')
parser.add_argument("--fanout", type=list, default=[], help='neighborhood size')
parser.add_argument("--layers", type=int, default=2, help='layer of graph neural network')
parser.add_argument("--n_neighbors", type=int, default=5,help='numbers of sampled neighbors each layer')

args = parser.parse_args()
if not args.distributed:
    os.environ["WANDB_MODE"] = "offline"

if args.distributed:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    is_master = local_rank == 0
else:
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    local_rank = 0
    is_master=True

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir
FT_ckpt_dir=args.ckpt_dir+args.model_name+'/'
if is_master and not os.path.exists(FT_ckpt_dir):
    os.mkdir(FT_ckpt_dir)

if args.distributed:
    dist.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(SEED + torch.distributed.get_rank())
else:
    world_size=1
    device = torch.device("cuda", local_rank)
    seed_all(SEED)
print(device)

# graph data
g = dgl.load_graphs(args.graph_path + f'/{args.graph_data}/Preprocessed/{args.graph_data}_graph.wfeat.dgl')[0][0]
g.nodes['N'].data['_ID']=g.nodes('N')
g=g.edge_type_subgraph(['regulate','interact_with'])
g=dgl.node_subgraph(g,{'N':torch.range(0,args.gene_num-1,1,dtype=torch.int64)})
if args.graph_feat=='gene2vec':
    gene2vec=np.load('/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/scBERT/data/gene2vec_16906.npy')
    g.nodes['N'].data['feat']=torch.tensor(gene2vec,dtype=torch.float32)
    gnn_feat_dim=200
else:
    gnn_feat_dim=128
if not args.fanout:
    args.fanout=[{k[1]:args.n_neighbors for k in g.canonical_etypes}]*args.layers
    #args.fanout=[10,10]
g=g.to(device)

def define_wandb_metrcis():
    wandb.define_metric("train/loss", summary="min", step_metric="epoch")
    wandb.define_metric("train/acc", summary="max", step_metric="epoch")
    wandb.define_metric("valid/loss", summary="min", step_metric="epoch")
    wandb.define_metric("valid/f1", summary="max", step_metric="epoch")
    wandb.define_metric("'valid/acc'", summary="max", step_metric="epoch")
    # wandb.define_metric("test/avg_bio", summary="max")
if is_master:
    hyperparameter_wandb=dict(
        task='Cell Annotation',
        dataset_name=args.data_path.split('/')[-1],
        pretrain_model=args.model_path.split('/')[-1],
        seed=args.seed,
        epochs=args.epoch,
        pos_embed=args.pos_embed,
        distributed=args.distributed,
        gene_num=args.gene_num,
        bin_num=args.bin_num,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        fanout=args.fanout,
        grad_acc=args.grad_acc,
        ckpt_dir=args.ckpt_dir+args.model_name,
        dropout=args.dropout,
        graph_feat=args.graph_feat,
        num_head=args.num_head,
        gnn_conv=args.gnn_conv
    )
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    KG_type='gene'
    exp_name=f'CA_KG{KG_type}_dpo{args.dropout}_feat_{args.graph_feat}_{args.gnn_conv}+MLP_l{args.layers}n{args.n_neighbors}_{now}'
    tags=['Finetune',hyperparameter_wandb['task'],
          hyperparameter_wandb['dataset_name'],'KG_'+KG_type+'_'+args.graph_data,
          'SingleNode' if world_size<=4 else 'MultiNode',
          f'feat_{args.graph_feat}',args.gnn_conv]
    run=wandb.init(
        config=hyperparameter_wandb,
        project='scBert_KG',
        name=exp_name,
        reinit=True,
        save_code=True,
        job_type='FineTune',
        tags=tags,
        settings=wandb.Settings(start_method="fork"),
    )
    define_wandb_metrcis()
    print(f'Config: {hyperparameter_wandb}')


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]



class Emb2CellType(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Emb2CellType, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x): # [batch_size,gene_num,emb_size(200)]
        x = x[:,None,:,:] # [batch_size,1,gene_num,emb_size(200)]
        x = self.conv1(x)   # [batch_size,1,gene_num,1]
        x = self.act(x)
        x = x.view(x.shape[0],-1)# [batch_size,gene_num]

        x = self.fc1(x)# [batch_size,512]
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def generator_cat(generators):
    for gen in generators:
        yield from gen


data = sc.read_h5ad(args.data_path)

label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
#store the label dict and label for prediction
with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X

acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data.shape[0])


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

if args.distributed:
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
else:
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING,
    emb_dropout = args.dropout,
    ff_dropout =args.dropout,
    attn_dropout = args.dropout,
)
gnn=HeteroGAT(canonical_etypes=g.canonical_etypes,
                      in_feat=gnn_feat_dim, hidden_feat=512, out_feat=200,
                      num_heads=args.num_head, layers=args.fanout.__len__(),dropout=args.dropout,conv_type=args.gnn_conv)
emb2celltype=Emb2CellType(dropout=args.dropout,h_dim=128, out_dim=label_dict.shape[0])


if not os.path.exists(FT_ckpt_dir+model_name+'Performer_best.pth'):
    path = args.model_path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to_out = nn.Identity()
    start_epoch=0
    start_epoch=0
else:
    model.to_out = nn.Identity()
    performer_ckpt = torch.load(FT_ckpt_dir+model_name+'Performer_best.pth')#checkpoints
    start_epoch = performer_ckpt['epoch']
    model.load_state_dict(performer_ckpt['model_state_dict'])
    # Graph Module
    gnn_ckpt=torch.load(FT_ckpt_dir+model_name+'GNN_best.pth')#checkpoints
    gnn.load_state_dict(gnn_ckpt['model_state_dict'])
    cls_ckpt = torch.load(FT_ckpt_dir + model_name + 'Classifier_best.pth')  # checkpoints
    emb2celltype.load_state_dict(cls_ckpt['model_state_dict'])
    if is_master:
        print(f'Restart training from checkpoint:{FT_ckpt_dir + model_name}, Epoch: {start_epoch}, Loss: {performer_ckpt["losses"]}')
model = model.to(device)
emb2celltype=emb2celltype.to(device)
gnn=gnn.to(device)
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True

if args.distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    gnn = DDP(gnn, device_ids=[local_rank], output_device=local_rank)
    emb2celltype = DDP(emb2celltype, device_ids=[local_rank], output_device=local_rank)




# optimizer
params = generator_cat([model.parameters(), emb2celltype.parameters(), gnn.parameters()])
optimizer = Adam(params, lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)

if args.distributed:
    loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
    dist.barrier()
else:
    loss_fn = nn.CrossEntropyLoss(weight=None).to(device)
trigger_times = 0
max_acc = 0.0
if is_master:
    wandb.watch(model,log='all',idx=0,log_freq=100)
    wandb.watch(gnn,log='all',idx=1,log_freq=100)
    wandb.watch(emb2celltype,log='all',idx=2,log_freq=100)
neighbor_sampler=dgl.dataloading.NeighborSampler(args.fanout)
for i in range(start_epoch, EPOCHS+1):
    model.train()
    gnn.train()
    emb2celltype.train()
    if args.distributed:
        train_loader.sampler.set_epoch(i)
        dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, data in enumerate(tqdm(train_loader)):
        index += 1
        data, labels = data[0].to(device),data[1].to(device)
        expressed_gene_idx=[torch.where(cell!=0)[0] for cell in data]
        # expressed_gene_uni=torch.cat(expressed_gene_idx).unique()


        if index % GRADIENT_ACCUMULATION != 0:
            if args.distributed:
                with model.no_sync():
                    sc_emb = model(data)
                    for sc_idx, node_idx in enumerate(expressed_gene_idx):
                        _, _, blocks = neighbor_sampler.sample_blocks(g, {'N': node_idx})
                        kg_emb = gnn(blocks)
                        sc_emb[sc_idx][node_idx] += kg_emb

                    logits = emb2celltype(sc_emb)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            else:
                sc_emb = model(data)
                for sc_idx,node_idx in enumerate(expressed_gene_idx):
                    _, _, blocks = neighbor_sampler.sample_blocks(g, {'N': node_idx})
                    kg_emb=gnn(blocks)
                    sc_emb[sc_idx][node_idx]+=kg_emb

                logits = emb2celltype(sc_emb)
                loss = loss_fn(logits, labels)
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            sc_emb = model(data)
            for sc_idx, node_idx in enumerate(expressed_gene_idx):
                _, _, blocks = neighbor_sampler.sample_blocks(g, {'N': node_idx})
                kg_emb = gnn(blocks)
                sc_emb[sc_idx][node_idx] += kg_emb

            logits = emb2celltype(sc_emb)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        if not args.distributed:
            break
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    if args.distributed:
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        wandb.log(
            {
                'epoch':i,
                'train/loss':epoch_loss,
                'train/acc':epoch_acc,
            }
        )
    if args.distributed:
        dist.barrier()
    scheduler.step()


    if i % VALIDATE_EVERY == 0:
        model.eval()
        gnn.eval()
        emb2celltype.eval()
        if args.distributed:
            dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                expressed_gene_idx = [torch.where(cell != 0)[0] for cell in data_v]
                sc_emb = model(data_v)
                for sc_idx, node_idx in enumerate(expressed_gene_idx):
                    _, _, blocks = neighbor_sampler.sample_blocks(g, {'N': node_idx})
                    kg_emb = gnn(blocks)
                    sc_emb[sc_idx][node_idx] += kg_emb

                logits = emb2celltype(sc_emb)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
                if not args.distributed:
                    break
            del data_v, labels_v, logits, final_prob, final
            # gather
            if args.distributed:
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)

            ### debug mode ###
            if args.distributed: # ori code
                no_drop = predictions != -1
                predictions = np.array((predictions[no_drop]).cpu())
                truths = np.array((truths[no_drop]).cpu())
            else:# test code
                predictions = np.array(predictions[0].cpu())
                truths = np.array(truths[0].cpu())

            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / index
            if args.distributed:
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
                #print(confusion_matrix(truths, predictions))

                print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))
                #metric_dict=classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4,output_dict=True)
                wandb.log(
                    {
                        'epoch': i,
                        'valid/loss': val_loss,
                        'valid/f1': f1,
                        'valid/acc':cur_acc
                    }
                )
            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
                if not args.distributed:
                    save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name+'Performer', FT_ckpt_dir)
                    save_best_ckpt(i, gnn, optimizer, scheduler, val_loss, model_name+'GNN', FT_ckpt_dir)
                    save_best_ckpt(i, emb2celltype, optimizer, scheduler, val_loss, model_name + 'Classifier', FT_ckpt_dir)
                else:

                    save_best_ckpt(i, model.module, optimizer, scheduler, val_loss, model_name + 'Performer',
                                   FT_ckpt_dir)
                    save_best_ckpt(i, gnn.module, optimizer, scheduler, val_loss, model_name + 'GNN', FT_ckpt_dir)
                    save_best_ckpt(i, emb2celltype.module, optimizer, scheduler, val_loss, model_name + 'Classifier',
                                   FT_ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
    del predictions, truths
if is_master:
    run.finish()
    wandb.finish()