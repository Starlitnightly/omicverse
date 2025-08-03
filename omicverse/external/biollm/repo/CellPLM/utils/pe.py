import torch
from torch import nn
from ..utils import create_norm
import math

def select_pe_encoder(pe):
    if pe in ['sin', 'sinu', 'sinusoidal']:
        return Sinusoidal2dPE
    elif pe in ['learnable', 'bin']:
        return Learnable2dPE
    elif pe in ['naive', 'mlp']:
        return NaivePE
    elif pe in ['lap', 'graphlap', 'lappe']:
        return GraphLapPE
    else:
        raise NotImplementedError(f'Unsupported positional encoding type: {pe}')

class Sinusoidal2dPE(nn.Module):
    def __init__(self, d_model, height=100, width=100):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        self.d_model = d_model
        self.height = height
        self.width = width
        self.pe_key = 'coord'
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)

        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.pe_enc = nn.Embedding.from_pretrained(pe.flatten(1).T)

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        x = ((x*1.02-0.01) * self.width).long()
        y = ((y*1.02-0.01) * self.height).long()
        x[x >= self.width] = self.width - 1
        y[y >= self.height] = self.height - 1
        x[x < 0] = 0
        y[y < 0] = 0
        pe_input = x * self.width + y
        return self.pe_enc(pe_input)

class Learnable2dPE(nn.Module):
    def __init__(self, d_model, height=100, width=100):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        """
        super().__init__()
        self.pe_enc = nn.Embedding(height * width, d_model)
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'coord'

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        x = ((x*1.02-0.01) * self.width).long()
        y = ((y*1.02-0.01) * self.height).long()
        x[x >= self.width] = self.width
        y[y >= self.height] = self.height
        x[x < 0] = 0
        y[y < 0] = 0
        pe_input = x * self.width + y
        return self.pe_enc(pe_input)

class NaivePE(nn.Module):
    def __init__(self, d_model, coord_dim = 2, height=None, width=None):
        """
        :param d_model: dimension of the model
        :param coord_dim: dimension of coordinates
        :param height: placeholder
        :param width: placeholder
        """
        super().__init__()
        self.pe_enc = nn.Sequential(
                            nn.Linear(coord_dim, d_model),
                            nn.PReLU(),
        )
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'coord'

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        return self.pe_enc(coordinates)

class GraphLapPE(nn.Module):
    def __init__(self, d_model, k = 10, height=None, width=None):
        """
        :param d_model: dimension of the model
        :param k: top k
        :param height: placeholder
        :param width: placeholder
        """
        super().__init__()
        self.k = k
        self.pe_enc = nn.Sequential(
                            nn.Linear(k, d_model),
                            nn.PReLU(),
        )
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'eigvec'

    def forward(self, eigvec):
        if eigvec[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(eigvec.shape[0], -1)
        eigvec = eigvec * (torch.randint(0, 2, (self.k, ), dtype=torch.float, device=eigvec.device)[None, :]*2-1)
        return self.pe_enc(eigvec)

