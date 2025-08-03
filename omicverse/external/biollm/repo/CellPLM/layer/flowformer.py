import torch
import torch.nn as nn
from .transformer import AbstractTrasnformerLayer
from ..utils import create_norm

## Core code for Flow-Attention, Please refer to each folder for corresponding experiments

class Flow_Attention(nn.Module):
    # flow attention in normal version
    def __init__(self, d_model, n_heads, drop_out=0.01, d_input=None, d_output=None, eps=1e-6):
        super(Flow_Attention, self).__init__()
        self.n_heads = n_heads
        if d_input is None:
            d_input = d_model
        if d_output is None:
            d_output = d_model
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, x):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        ## 1. Linear projection
        queries = keys =  values = x
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


class Flow_Attention_Causal(nn.Module):
    # flow attention in causal version
    def __init__(self, d_model, n_heads, drop_out=0.05, d_input=None, d_output=None, eps=1e-6):
        super(Flow_Attention_Causal, self).__init__()
        self.n_heads = n_heads
        if d_input is None:
            d_input = d_model
        if d_output is None:
            d_output = d_model
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def causal_dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhldm", k, v)
        kv = torch.cumsum(kv, dim=2)
        qkv = torch.einsum("nhld,nhldm->nhlm", q, kv)
        return qkv

    def forward(self, x):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        queries = keys=  values = x
        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Causal Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhld->nhl", queries + self.eps, keys.cumsum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhld->nhl", keys + self.eps, queries.cumsum(dim=2) + self.eps))
        # approximate normal conservation col and row by multiplying corresponding element number
        normal = (((torch.arange(queries.shape[2])).float() + 1.0)).to(queries.device)[None, None, :]
        sink_incoming = sink_incoming * normal
        source_outgoing = source_outgoing * normal
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhld->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).cumsum(dim=2) + self.eps) / normal
        conserved_source = torch.einsum("nhld,nhld->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).cumsum(
                                            dim=2) + self.eps) / normal
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink)
        conserved_source = torch.exp(conserved_source)
        source_competition = (conserved_source / conserved_source.cumsum(dim=-1)) * normal
        # (4) Causal dot product
        x = (self.causal_dot_product(queries * (sink_incoming[:, :, :, None] / normal[:, :, :, None]),  # for value normalization
                                     keys,
                                     values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x

class FlowformerLayer(nn.Module, AbstractTrasnformerLayer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout = 0.0,
            norm = 'layernorm',
            norm_first=True,
            causal=False,
    ):
        super(FlowformerLayer, self).__init__()
        if not causal:
            self.self_attn = Flow_Attention(embed_dim, num_heads)
        else:
            self.self_attn = Flow_Attention_Causal(embed_dim, num_heads)
        self._ff_block = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(embed_dim * 2, embed_dim),
                                       nn.Dropout(dropout),
                                       )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = create_norm(norm, embed_dim)
        self.norm2 = create_norm(norm, embed_dim)
        self.norm_first = norm_first
        self.support_output_attentions = False

    def _sa_block(self, x):
        x = x.unsqueeze(0)
        x = self.self_attn(x)
        return self.dropout1(x)[0, :, :]

    def forward(self, x, attn_mask=None, output_attentions=False):
        assert output_attentions == False, 'output_attentions not implemented for Cosformer'
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x