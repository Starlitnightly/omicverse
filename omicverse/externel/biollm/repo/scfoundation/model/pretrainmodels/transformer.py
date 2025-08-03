# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited


import torch
import torch.nn as nn
import torch.nn.functional as F

class pytorchTransformerModule(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 ff_mult=4,
                 norm_first=False,
                 ):
        super(pytorchTransformerModule, self).__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                     dim_feedforward=dim * ff_mult,
                                                     batch_first=True,
                                                     norm_first=norm_first,
                                                     #activation="gelu",
                                                     ))

        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask):
        b, n, _, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # x get encodings [B, N, D] , batch_first is True
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask) # , src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # x = self.transformer_encoder(x)
        x = self.norm(x)

        return x
