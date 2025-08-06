"""
Model class

"""

import warnings
warnings.filterwarnings("ignore")
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import sys
sys.path.append('../')
from typing import Any
import torch


def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp \
            (torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, token_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, output_dim:int, dropout: float = 0.05):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        self.encoder = nn.Sequential(nn.Linear(token_dim, d_model),
                                     nn.GELU(),
                                     nn.LayerNorm(d_model))



        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)


        self.d_model = d_model
        self.dropout = dropout


        self.decoder = nn.Sequential(full_block(d_model, 1024, self.dropout),
                                     full_block(1024, output_dim, self.dropout),
                                     full_block(output_dim, output_dim, self.dropout),
                                     nn.Linear(output_dim, output_dim)
                                     )

        self.binary_decoder = nn.Sequential(
            full_block(output_dim + 1280, 2048, self.dropout),
            full_block(2048, 512, self.dropout),
            full_block(512, 128, self.dropout),
            nn.Linear(128, 1)
        )

        self.gene_embedding_layer = nn.Sequential(nn.Linear(token_dim, d_model),
                                                  nn.GELU(),
                                                  nn.LayerNorm(d_model))

        self.pe_embedding = None

    def forward(self, src: Tensor, mask: Tensor):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=( 1 -mask))
        gene_output = self.decoder(output) # batch x seq_len x 128
        # embedding = torch.mul(gene_output, mask.t().unsqueeze(2)).sum(0) # average over non zero genes
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[0, :, :] # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize.
        return gene_output, embedding


    def predict(self, cell_embedding, gene_embeddings):
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        dec = self.binary_decoder \
            (torch.hstack((cell_embedding, gene_embeddings)))
        return dec

