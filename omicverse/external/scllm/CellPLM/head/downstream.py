import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ..decoder import MLPDecoder, ResMLPDecoder
from ..latent import GMVAELatentLayer
from ..objective import ReconstructionLoss
from ..utils.data import XDict
from ..encoder.transformer import TransformerEncoder

def buildNetwork(layers, dropouts, activation=nn.ReLU()):
    net = []
    for i in range(1, len(layers)):
        if dropouts[i-1] > 0:
            net.append(nn.Dropout(dropouts[i-1]))
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if i < len(layers) - 1:
            net.append(activation)
    net = nn.Sequential(*net)
    return net

class AnnotationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, dropout, norm, batch_num, **kwargs):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        layers = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes] 
        dropouts = [dropout] * len(layers)
        self.mlp = buildNetwork(layers, dropouts)

    def forward(self, x_dict):
        logits = self.mlp(x_dict['h'][x_dict['loss_mask']])
        pred = logits.argmax(1)
        if 'label' in x_dict:
            y = x_dict['label'][x_dict['loss_mask']].long()
            loss = self.ce_loss(logits, y)
            return {'pred': pred, 'latent': x_dict['h'], 'label': y}, loss
        else:
            return {'pred': pred, 'latent': x_dict['h']}, torch.tensor(float('nan'))

class PatientClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, dropout, norm=None, batch_num=None, **kwargs):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.cls = nn.Parameter(torch.randn((1, in_dim)) * 0.01)
        self.output_layer = nn.Linear(in_dim, num_classes)

    def classify(self, x_dict):

        return self.output_layer(torch.mean(x_dict['h'], 0, keepdim=True))

    def forward(self, x_dict):
        prob = self.classify(x_dict)
        pred = prob.argmax(1)
        y = x_dict['label'].long()
        loss = self.ce_loss(prob, y)
        return {'pred': pred, 'latent': x_dict['h']}, loss

class DenoisingHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num, lib_size=1e4, 
                 log_norm=True, **kwargs):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lib_size = lib_size
        self.log_norm = log_norm
        layers = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim] 
        dropouts = [dropout] * len(layers)
        self.mlp = buildNetwork(layers, dropouts)

    def forward(self, x_dict):
        pred = self.mlp(x_dict['h']) #* x_dict['input_mask']
        if self.training:
            y = x_dict['x_seq'].to_dense()
            if self.lib_size is not None:
                y = y/y.sum(1)[:, None] * self.lib_size
            if self.log_norm:
                y = torch.log(y+1)
            loss = self.mse_loss(pred * x_dict['input_mask'], y * x_dict['input_mask']) + self.mse_loss(pred, y) 
        else:
            loss = torch.zeros(1)
        return {'pred': pred, 'latent': x_dict['h']}, loss


class EmbedderHead(nn.Module):
    def __init__(self, in_dim=None, hidden_dim=None, out_dim=None, num_layers=None,
                 dropout=None, norm=None, batch_num=None, lib_size=None,
                 log_norm=False, **kwargs):
        super().__init__()

    def forward(self, x_dict):
        pred = x_dict['h']
        return {'pred': pred, 'latent': x_dict['h']}, torch.tensor(0.).to(x_dict['h'].device)

class ImputationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num, **kwargs):
        super().__init__()
        self.mse_loss = nn.MSELoss()
#         self.mse_loss = lambda x, y: torch.mean(((x-y) * (y/5+1))**2)
        layers = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim] 
        dropouts = [dropout] * len(layers)
        self.mlp = buildNetwork(layers, dropouts)

    def forward(self, x_dict):
        pred = self.mlp(x_dict['h'])[:, x_dict['gene_mask']]
        y = x_dict['label'][:, x_dict['gene_mask']]
        loss = self.mse_loss(pred, y)
        return {'pred': pred, 'latent': x_dict['h']}, loss

class PerturbationPredictionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num, lib_size=None, 
                 log_norm=False, **kwargs):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lib_size = lib_size
        self.log_norm = log_norm
        layers = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim] 
        dropouts = [dropout] * len(layers)
        self.mlp = buildNetwork(layers, dropouts)

    def forward(self, x_dict):
        pred = self.mlp(x_dict['h']) * x_dict['input_mask']
        y = x_dict['label'].to_dense()
        if self.lib_size is not None:
            y = y/y.sum(1)[:, None] * self.lib_size
        if self.log_norm:
            y = torch.log1p(y)
        loss = self.mse_loss(pred, y * x_dict['input_mask'])
        return {'pred': pred, 'latent': x_dict['h']}, loss

def get_normalized_expression(model, seq_list, batch_list, coord_list=None, device='cuda',
                              transform_batch=None, library_size=None, n_samples=1, return_mean=True):
    transform_batch = range(len(seq_list)) if transform_batch is None else transform_batch
    exprs = []
    for i in tqdm(range(len(seq_list))):
        input_dict = {'x_seq': seq_list[i].to(device)}
        if coord_list is not None:
            input_dict['coord'] = coord_list[i].to(device)
        x_dict = XDict(input_dict)
        per_batch_exprs = []
        for batch in transform_batch:
            per_sample_exprs = []
            input_dict['batch'] = batch * torch.ones(len(seq_list[i])).float().to(device)
            for sample in range(len(n_samples)):
                out_dict, _ = model(x_dict)
                output = out_dict['pred']
                if library_size is not None:
                    output = output / torch.sum(output, 1, keepdim=True) * library_size
                output = output.cpu().numpy()
                per_sample_exprs.append(output)
            per_batch_exprs.append(np.stack(per_sample_exprs))
        per_batch_exprs = np.stack(per_batch_exprs, axis=1)
        exprs.append(per_batch_exprs.mean(1))

    if n_samples > 1:
        # The -2 axis correspond to cells.
        exprs = np.concatenate(exprs, axis=-2)
    else:
        exprs = np.concatenate(exprs, axis=0)
    if n_samples > 1 and return_mean:
        exprs = exprs.mean(0)

    return exprs
