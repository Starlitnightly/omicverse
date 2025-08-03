r"""
Graph and GAN networks in SLAT
"""
import math
from typing import Optional, List
import torch
import torch.nn.functional as F

from .graphconv import CombUnweighted

        
class LGCN(torch.nn.Module):
    r"""
    Lightweight GCN which remove nonlinear functions and concatenate the embeddings of each layer:
        (:math:`Z = f_{e}(A, X) = Concat( [X, A_{X}, A_{2X}, ..., A_{KX}])W_{e}`)
    
    Parameters
    ----------
    input_size
        input dim
    K
        LGCN layers
    """
    def __init__(self, input_size:int, K:Optional[int]=8):
        super(LGCN, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor):
        x = self.conv1(feature, edge_index)
        return x


class LGCN_mlp(torch.nn.Module):
    r"""
    Add one hidden layer MLP in LGCN
    
    Parameters
    ----------
    input_size
        input dim
    output_size
        output dim
    K
        LGCN layers
    hidden_size
        hidden size of MLP
    dropout
        dropout ratio
    """
    def __init__(self, input_size:int, output_size:int, K:Optional[int]=8,
                 hidden_size:Optional[int]=512, dropout:Optional[float]=0.2):
        super(LGCN_mlp, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor):
        x = self.conv1(feature, edge_index)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class WDiscriminator(torch.nn.Module):
    r"""
    WGAN Discriminator
    
    Parameters
    ----------
    hidden_size
        input dim
    hidden_size2
        hidden dim
    """
    def __init__(self, hidden_size:int, hidden_size2:Optional[int]=512):
        super(WDiscriminator, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)
    def forward(self, input_embd):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))


class transformation(torch.nn.Module):
    r"""
    Transformation in LGCN
    
    Parameters
    ----------
    hidden_size
        input dim
    """
    def __init__(self, hidden_size:Optional[int]=512):
        super(transformation, self).__init__()
        self.trans = torch.nn.Parameter(torch.eye(hidden_size))
    def forward(self, input_embd):
        return input_embd.mm(self.trans)


class notrans(torch.nn.Module):
    r"""
    LGCN without transformation
    """
    def __init__(self):
        super(notrans, self).__init__()
    def forward(self, input_embd:torch.Tensor):
        return input_embd


class ReconDNN(torch.nn.Module):
    r"""
    Data reconstruction network
    
    Parameters
    ----------
    hidden_size
        input dim
    feature_size
        output size (feature input size)
    hidden_size2
        hidden size
    """
    def __init__(self, hidden_size:int, feature_size:int, hidden_size2:Optional[int]=512):
        super(ReconDNN, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, feature_size)
    def forward(self, input_embd:torch.Tensor):
        return self.output(F.relu(self.hidden(input_embd)))