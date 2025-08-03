# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited


import torch
import torch.nn as nn
from .performer import Gene2VecPositionalEmbedding, RandomPositionalEmbedding

from torch.nn import functional as F


class pytorchTransformerLM_mse(nn.Module):
    def __init__(self,
                 num_tokens,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 ff_mult=4,
                 emb_dropout=0.,
                 g2v_position_emb=False
                 ):
        super(pytorchTransformerLM_mse, self).__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len)
            print("pos_emb Gene2Vec", self.pos_emb)
        else:
            self.pos_emb = RandomPositionalEmbedding(dim, max_seq_len)
            print("pos_emb no ", self.pos_emb)

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                        nhead=heads,
                                                        dim_feedforward=dim*ff_mult)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens)
        self.to_final = nn.Linear(num_tokens, 1)

    def forward(self, x):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        x += self.pos_emb(x)

        x = self.dropout(x)  # embedding dropout

        # x = self.transformer_encoder(x)  # get encodings [B, N, D]
        x=torch.transpose(x, 0, 1)
        x = self.transformer_encoder(x)
        x = torch.transpose(x, 0, 1)

        # layernorm and to logits
        x = self.norm(x)
        x = self.to_out(x)  # [B, N, C]

        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2)  # torch.Size([8, 13418])
        else:
            return x

        return x

def exists(val):
    return val is not None


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
