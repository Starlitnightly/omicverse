import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np

def positional_encoding(coords, enc_dim, sigma):
    freqs = (
        2 * np.pi * sigma ** (torch.arange(enc_dim//2, dtype=torch.float, device=coords.device) / enc_dim)
    )
    freqs = torch.reshape(freqs, (1,1, torch.numel(freqs)))
    coords = coords.unsqueeze(-1)
    
    freqs = coords * freqs #N x 2 x enc_dim/2
    s = torch.sin(freqs)
    c = torch.cos(freqs)

    x = torch.cat((s,c), axis=-1) #N x 2 x enc_dim
    x = torch.reshape(x, (x.shape[0], -1))

    return x