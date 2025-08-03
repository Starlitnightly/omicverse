import torch
from torch import nn
import torch.nn.functional as F
from ..utils import create_activation, create_norm

class MeanAct(nn.Module):
    """Mean activation class."""

    def __init__(self, softmax):
        super().__init__()
        self.softmax = softmax

    def forward(self, x):
        if not self.softmax:
            return torch.clamp(torch.exp(x), min=1e-5, max=1e6)
        else:
            return torch.softmax(x, 1)

class DispAct(nn.Module):
    """Dispersion activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-4, max=1e4)

class ZINB(nn.Module):
    """ZINB Decoder.
    Parameters
    ----------
    input_dim : int
        dimension of input feature.
    n_z : int
        dimension of latent embedding.
    n_dec_1 : int optional
        number of nodes of decoder layer 1.
    """

    def __init__(self, hidden_dim, out_dim, n_dec_1=128, softmax=True, disp='gene'):
        super().__init__()
        self.dec_1 = nn.Linear(hidden_dim, n_dec_1)
        self.dec_mean = nn.Sequential(nn.Linear(n_dec_1, out_dim), MeanAct(softmax))
        self.dec_pi = nn.Sequential(nn.Linear(n_dec_1, out_dim), nn.Sigmoid())
        self.disp = disp
        if disp == 'gene':
            self.dec_disp = nn.Parameter(torch.ones(out_dim))
        else:
            self.dec_disp = nn.Sequential(nn.Linear(n_dec_1, out_dim), DispAct())

    def forward(self, z):
        """Forward propagation.
        Parameters
        ----------
        z :
            embedding.
        Returns
        -------
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB4
        """

        h = F.relu(self.dec_1(z))
        _mean = self.dec_mean(h)
        if self.disp == 'gene':
            _disp = self.dec_disp.repeat(z.shape[0], 1)
        else:
            _disp = self.dec_disp(h)
        _pi = self.dec_pi(h)
        return _mean, _disp, _pi

class NB(nn.Module):
    """NB Decoder.
    Parameters
    ----------
    input_dim : int
        dimension of input feature.
    n_z : int
        dimension of latent embedding.
    n_dec_1 : int optional
        number of nodes of decoder layer 1.
    """

    def __init__(self, hidden_dim, out_dim, n_dec_1=128, softmax=False, disp='gene'):
        super().__init__()
        self.dec_1 = nn.Linear(hidden_dim, n_dec_1)
        self.dec_mean = nn.Sequential(nn.Linear(n_dec_1, out_dim), MeanAct(softmax))
        self.disp = disp
        if disp == 'gene':
            self.dec_disp = nn.Parameter(torch.randn(out_dim))
            self.dec_disp_act = DispAct()
        else:
            self.dec_disp = nn.Sequential(nn.Linear(n_dec_1, out_dim), DispAct())

    def forward(self, z):
        """Forward propagation.
        Parameters
        ----------
        z :
            embedding.
        Returns
        -------
        _mean :
            data mean from NB.
        _disp :
            data dispersion from NB.
        """

        h = F.relu(self.dec_1(z))
        _mean = self.dec_mean(h)
        if self.disp == 'gene':
            _disp = self.dec_disp_act(self.dec_disp.repeat(z.shape[0], 1))
        else:
            _disp = self.dec_disp(h)
        return _mean, _disp


class NBMLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num=0, dataset_num=0, platform_num=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.layers = nn.ModuleList()
        self.covariate_layers = nn.ModuleList()
        self.covariate_num = {
            'batch': batch_num,
            'dataset': dataset_num,
            'platform': platform_num,
        }
        for i in range(num_layers-1):
            dim = hidden_dim if i > 0 else in_dim
            self.layers.append(nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                create_norm(norm, hidden_dim),
            ))
            if sum(self.covariate_num.values()): # Covariates exist
                self.covariate_layers.append(nn.ModuleDict())
                for cov in self.covariate_num.keys():
                    if self.covariate_num[cov] > 0:
                        self.covariate_layers[-1][cov] = nn.Sequential(
                            nn.Embedding(self.covariate_num[cov], hidden_dim),
                            nn.PReLU(),
                            create_norm(norm, hidden_dim),
                        )
        self.out_layer = NB(
            hidden_dim, out_dim,
        )


    def forward(self, x_dict):
        x = x_dict['h']
        for i, layer in enumerate(self.layers):
            if sum(self.covariate_num.values()): # Covarites (batch/dataset/platform) exist
                x = layer(x)
                for cov in self.covariate_num.keys(): # Iterate over each type of covariate (batch/dataset/platform)
                    if self.covariate_num[cov] > 0: # if a certain type of covariate exist
                        if cov in x_dict: # Whether the covaraite label is input
                            x += self.covariate_layers[i][cov](x_dict[cov])
                        else: # If not input, take average over all of them
                            convariate_layer = self.covariate_layers[i][cov]
                            x += convariate_layer[2](convariate_layer[1](convariate_layer[0].weight.detach().sum(0).unsqueeze(0)))
            else:
                x = layer(x)
        mean, disp = self.out_layer(x)
        return {'mean': mean, 'disp': disp, 'recon': mean, 'latent': x_dict['h']}
