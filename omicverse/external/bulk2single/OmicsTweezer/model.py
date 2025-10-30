import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
import torch
from tqdm.auto import trange


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("seed is fixed, seed is {}".format(seed))
set_seed()

def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss
    
class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

        
class OmicsTweezer(object):
    def __init__(self, architectures, epochs, batch_size, target_type, learning_rate, device=None):
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.target_type = target_type
        self.learning_rate = learning_rate
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021
        self.outdir = './result_data/'
        self.architectures_dim = architectures[0]
        self.architectures_drop = architectures[1]
        cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if device is None:
            self.device = torch.device('cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def OmicsTweezer_model(self, celltype_num):
        feature_num = len(self.used_features)


        self.encoder_da = nn.Sequential(EncoderBlock(feature_num, self.architectures_dim[0], self.architectures_drop[0]), 
                                        EncoderBlock(self.architectures_dim[0], self.architectures_dim[1], self.architectures_drop[1]),  
                                     )

        self.predictor_da = nn.Sequential(EncoderBlock(self.architectures_dim[1], self.architectures_dim[2],self.architectures_drop[2]), 
                                          EncoderBlock(self.architectures_dim[2], self.architectures_dim[3], self.architectures_drop[3]), 
                                          nn.Linear(self.architectures_dim[3], celltype_num), 
                                          nn.Softmax(dim=1))
        



        model_da = nn.ModuleList([])
        print("no initial weight")

        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)

        return model_da

    def prepare_dataloader(self, source_data, target_data, batch_size):
        ### Prepare data loader for training ###
        # Source dataset
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
      
        if self.target_type == "simulated":
            target_ratios = [target_data.obs[ctype] for ctype in self.labels]
            self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        elif self.target_type == "real":
            self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def train(self, source_data, target_data, loss_weight):

        ### prepare model structure ###
        self.prepare_dataloader(source_data, target_data, self.batch_size)
        self.model_da = self.OmicsTweezer_model(self.celltype_num)
        self.encoder_da.to(self.device)
        self.predictor_da.to(self.device)
        self.model_da.to(self.device)
        self.loss_weight= loss_weight
        ### setup optimizer ###


        optimizer_da1 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                  {'params': self.predictor_da.parameters()}], lr=self.learning_rate)

        
        metric_logger = defaultdict(list) 

        epoch_iter = trange(self.num_epochs, desc='Training', unit='epoch')
        for epoch in epoch_iter:
            self.model_da.train()

            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, disc_loss_epoch = 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                # get batch item of target
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                source_x = source_x.to(self.device)
                source_y = source_y.to(self.device)
                target_x = target_x.to(self.device)

                embedding_source = self.encoder_da(source_x)
                embedding_target = self.encoder_da(target_x)
                frac_pred = self.predictor_da(embedding_source)


                # caculate loss 
                pred_loss = L1_loss(frac_pred, source_y)       
                pred_loss_epoch += pred_loss.data.item()

                w_distance = embedding_source.mean() - embedding_target.mean()
                loss = pred_loss + self.loss_weight*abs(w_distance) 

                # update weights
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_da1.step()




            pred_loss_epoch = pred_loss_epoch/(batch_idx + 1)
            metric_logger['pred_loss'].append(pred_loss_epoch)
            epoch_iter.set_postfix(pred_loss=f"{pred_loss_epoch:.4f}")


            
    def prediction(self):
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            x = x.to(self.device)
            logits = self.predictor_da(self.encoder_da(x)).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth
    
