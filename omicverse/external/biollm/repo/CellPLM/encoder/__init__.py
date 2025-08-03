from .transformer import TransformerEncoder
from .mlp import ResMLPEncoder, MLPEncoder
from torch import nn

def setup_encoder(model_type, num_hidden, num_layers, dropout, activation, norm, nhead, covariates_dim=0) -> nn.Module:
    if model_type in ["performer", "cosformer", "transformer", "flowformer"]:
        mod = TransformerEncoder(
            num_hidden = num_hidden,
            nhead = nhead,
            num_layers = num_layers,
            dropout = dropout,
            activation = activation,
            # norm = norm,
            model_type = model_type,
            covariates_dim = covariates_dim,
        )
    elif model_type == 'mlp':
        mod = MLPEncoder(
            num_hidden=num_hidden,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            covariates_dim=covariates_dim,
        )
    elif model_type == "resmlp":
        mod = ResMLPEncoder(
            num_hidden=num_hidden,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            covariates_dim=covariates_dim,
        )
    elif model_type == 'none':
        mod = NullEncoder()
    else:
        raise NotImplementedError(f'Unsupported model type: {model_type}')
    return mod

class NullEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x_dict):
        x = x_dict['h']
        return {'hidden': x}