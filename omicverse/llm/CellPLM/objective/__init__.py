from .zinb import ZINBReconstructionLoss, NBReconstructionLoss, NBDenoisingLoss, NBImputationLoss
from .autoencoder import ReconstructionLoss
from torch import nn

def create_objective(**config) -> nn.Module:
    if config['type'] == 'recon':
        return ReconstructionLoss(**config)
    elif config['type'] == 'zinb':
        return ZINBReconstructionLoss(**config)
    elif config['type'] == 'nb':
        return NBReconstructionLoss(**config)
    elif config['type'] == 'denoise':
        return NBDenoisingLoss(**config)
    elif config['type'] == 'imputation':
        return NBImputationLoss(**config)
    else:
        raise ValueError(f"Unrecognized latent model name: {config['type']}")

class Objectives(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.layers = nn.ModuleList()
        if configs is not None:
            for c in configs:
                self.layers.append(create_objective(**c))

    def forward(self, out_dict, x_dict):
        if len(self.layers) == 0:
            raise RuntimeError("No objectives added to model.")
        total_loss = 0
        for layer in self.layers:
            loss = layer(out_dict, x_dict)
            total_loss += loss
        return total_loss

    def add_layer(self, **config):
        self.layers.append(create_objective(**config))