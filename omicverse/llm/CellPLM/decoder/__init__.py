from .mlp import MLPDecoder, ResMLPDecoder
from .zinb import NBMLPDecoder
from torch import nn

def setup_decoder(model_type, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num=0, dataset_num=0, platform_num=0) -> nn.Module:
    if model_type == 'nbmlp':
        mod = NBMLPDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
            dataset_num=dataset_num,
            platform_num=platform_num,
        )
    elif model_type == 'mlp':
        mod = MLPDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
            dataset_num=dataset_num,
            platform_num=platform_num,
        )
    elif model_type == "resmlp":
        mod = ResMLPDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    else:
        raise NotImplementedError(f'Unsupported model type: {model_type}')
    return mod