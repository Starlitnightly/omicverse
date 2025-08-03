from dataclasses import dataclass
import torch
import torch.nn as nn
# import torch.nn.functional as F
from dataclasses import dataclass, field
#from dgl.nn import GCN2Conv, GraphConv

@dataclass
class Args:
    dataname : str
    gpu : int = 0
    epochs : int = 1000
    lr1 : float = 1e-3
    wd1 : float = 0.0
    lambd : float = 1e-3
    n_layers : int = 9
    der : float = 0.2
    dfr : float = 0.2
    device : str = field(init=False)
    encoder_dim : int = 256
    use_encoder : bool = False

    def __post_init__(self):
        if self.gpu != -1 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(self.gpu)
        else:
            self.device = 'cpu'



# fix the div zero standard deviation bug, Shuchen Luo (20220217)
def standardize(x, eps = 1e-12):
    return (x - x.mean(0)) / x.std(0).clamp(eps)

class Encoder(nn.Module):
    def __init__(self, in_dim : int, encoder_dim : int):
        super().__init__()
        self.layer = nn.Linear(in_dim, encoder_dim, bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.layer(x))


# GCN2Conv(in_feats, layer, alpha=0.1, lambda_=1, project_initial_features=True, allow_zero_in_degree=False, bias=True, activation=None)
class GCNII(nn.Module):
    def __init__(self, in_dim : int, encoder_dim: int, n_layers : int, alpha=None, lambda_=None, use_encoder=False):
        super().__init__()
        from dgl.nn import GCN2Conv, GraphConv
        self.n_layers = n_layers
        self.use_encoder = use_encoder
        if alpha is None:
            self.alpha = [0.1] * self.n_layers
        else:
            self.alpha = alpha
        if lambda_ is None:
            self.lambda_ = [1.] * self.n_layers
        else:
            self.lambda_ = lambda_
        if self.use_encoder:
            self.encoder = Encoder(in_dim, encoder_dim)
            self.hid_dim = encoder_dim
        else: self.hid_dim = in_dim
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(GCN2Conv(self.hid_dim, i + 1, alpha=self.alpha[i], lambda_=self.lambda_[i], activation=None))
    
    def forward(self, graph, x):
        if self.use_encoder:
            x = self.encoder(x)
        # print('GCNII forward: after encoder', torch.any(torch.isnan(x)))
        feat0 = x
        for i in range(self.n_layers):
            x = self.relu(self.convs[i](graph, x, feat0))
            # print('GCNII layer', i + 1, 'is_nan', torch.any(torch.isnan(x)))
        return x


    
class GCN(nn.Module):
    def __init__(self, in_dim : int, encoder_dim: int, n_layers : int, use_encoder=False):
        from dgl.nn import GCN2Conv, GraphConv
        super().__init__()

        self.n_layers = n_layers
        self.use_encoder = use_encoder

        if self.use_encoder:
            self.encoder = Encoder(in_dim, encoder_dim)
            self.hid_dim = encoder_dim
        else: self.hid_dim = in_dim
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(GraphConv(self.hid_dim, self.hid_dim, activation=None))

    def forward(self, graph, x):
        if self.use_encoder:
            x = self.encoder(x)
        # print('GCN forward: after encoder', torch.any(torch.isnan(x)))
        for i in range(self.n_layers):
            x = self.relu(self.convs[i](graph, x))
            # print('GCN layer', i + 1, 'is_nan', torch.any(torch.isnan(x)))
        return x        



class CCA_SSG(nn.Module):
    def __init__(self, in_dim, encoder_dim, n_layers, backbone='GCNII', alpha=None, lambda_=None, use_encoder=False):
        super().__init__()
        if backbone == 'GCNII':
            self.backbone = GCNII(in_dim, encoder_dim, n_layers, alpha, lambda_, use_encoder)
        elif backbone == 'GCN':
            self.backbone = GCN(in_dim, encoder_dim, n_layers, use_encoder)

    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone(graph1, feat1)
        h2 = self.backbone(graph2, feat2)
        # print('CCASSG forward: h1 is', torch.any(torch.isnan(h1)))
        # print('CCASSG forward: h2 is', torch.any(torch.isnan(h2)))
        z1 = standardize(h1)
        z2 = standardize(h2)
        # print('h1.std', h1.std(0))
        # print('h1-h1.mean(0)', h1 - h1.mean(0))
        # print('CCASSG forward: z1 is', torch.any(torch.isnan(z1)))
        # print('CCASSG forward: z2 is', torch.any(torch.isnan(z2)))

        return z1, z2

