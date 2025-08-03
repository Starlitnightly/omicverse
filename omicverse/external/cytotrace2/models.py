import torch
import torch.nn as nn
import torch.autograd as autograd
import math
import torch.nn.functional as F
#import pdb

class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class Binary_module(nn.Module):
    def __init__(self, input_size: int = 12000, hidden_size: int = 8, dropout: float = 0.2,
                  *args, **kwargs):
        super(Binary_module, self).__init__()
        
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_labels=1
        self.drop=dropout
        self.dropout = nn.Dropout(self.drop)
        self.maxrank = nn.Parameter(torch.FloatTensor(1,1).uniform_(0,1))
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size, self.hidden_size).normal_(mean=-0.1,std=0.055))
        self.ste = StraightThroughEstimator()
        enr = 2
        self.batchnorm = nn.BatchNorm1d(self.hidden_size*enr, affine=False)
        self.out = nn.Linear(self.hidden_size*enr, self.num_labels)
        
    def forward(self, x):
        alpha = self.ste(self.weight)
        n = alpha.sum(0).unsqueeze(0)
        maxrank = n.max()+10+torch.clamp(self.maxrank,min=0)*1000
        
        x = torch.minimum(x,maxrank)
        x_counts = (x<maxrank).float()
        R = torch.matmul(x,alpha)

        # Gene count score
        R_counts = torch.matmul(x_counts,alpha)/((n/x_counts.shape[1])*x_counts.sum(1).unsqueeze(1))

        # Ucell score
        R_out = 1 - ((R - (n*(n+1))/2)/(n*maxrank))

        # Concatenated gene set scores
        R_all = torch.cat((R_out, R_counts),1)

        # Batch normalization to transfer scores into shared space
        R_out_norm = self.batchnorm(R_all)

        # Apply dropout
        R_out_norm = self.dropout(R_out_norm)
        pred = self.out(R_out_norm)
  
        return R_out_norm, pred

class BinaryEncoder(nn.Module):

    def __init__(self, num_layers,**block_args):
        super().__init__()
        self.ste = StraightThroughEstimator()
        self.layers = nn.ModuleList([Binary_module(**block_args) for i in range(num_layers)])

    def forward(self, x):
        ucell_list = []
        pred_list = []
        for l in self.layers:
            ucell, pred = l(x)
            ucell_list.append(ucell)
            pred_list.append(pred.unsqueeze(1))
        ucell_out = torch.cat(ucell_list,1)
        pred_out = torch.cat(pred_list,1)
        
 
        return ucell_out, pred_out 


