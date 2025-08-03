import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

from .performer_module import default, Chunk, FeedForward, PreLayerNorm, Gene2VecPositionalEmbedding, RandomPositionalEmbedding


class FullAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads = 8,
                 dim_head = 64,
                 dropout = 0.,
                 qkv_bias = False
                ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # [B, N, H*D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) # [B, H, N, D]
        
        scaled_attention_logits = torch.einsum("bhxd,bhyd -> bhxy", q, k) / torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))
        attention_weights = F.softmax(scaled_attention_logits, dim=-1) # [B, H, N, N]
        
        attn_output = torch.einsum("bhnx,bhxd -> bhnd", attention_weights, v) # [B, H, N, D]
        out = rearrange(attn_output, 'b h n d -> b n (h d)', h = h) # [B, N, H*D]
        out = self.to_out(out)
        
        return self.dropout(out)

class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head = 64,
                 ff_chunks = 1,
                 ff_mult = 4,
                 ff_glu = False,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 qkv_bias = True,
                ):
        super().__init__()
        self.depth = depth
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        
        wrapper_fn = partial(PreLayerNorm, dim)
        for _ in range(depth):
            self.attns.append(wrapper_fn(FullAttention(dim, heads, dim_head, attn_dropout, qkv_bias)))
            self.ffns.append(wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)))
    
    def forward(self, x):
        for i in range(self.depth):
            x = x + self.attns[i](x) # residual link
            x = x + self.ffns[i](x) # residual link
        return x

class TransformerLM(nn.Module):
    def __init__(self,
                 num_tokens,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 dim_head = 64,
                 ff_chunks = 1,
                 ff_mult = 4,
                 ff_glu = False,
                 ff_dropout = 0.,
                 emb_dropout = 0.,
                 attn_dropout = 0.,
                 g2v_position_emb = True,
                 qkv_bias = True,
                ):
        super(TransformerLM, self).__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        
        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len)
            print("pos_emb Gene2Vec", self.pos_emb)
        else:
            self.pos_emb = RandomPositionalEmbedding(dim, max_seq_len)
            print("pos_emb no ", self.pos_emb)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_chunks, ff_mult, ff_glu, ff_dropout, attn_dropout, qkv_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens)
    
    def forward(self, x):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        
        # token and positional embedding
        x = self.token_emb(x)
        x += self.pos_emb(x)
        
        x = self.dropout(x) # embedding dropout
        
        x = self.transformer(x) # get encodings [B, N, D]

        # layernorm and to logits
        x = self.norm(x)
        x = self.to_out(x) # [B, N, C]
        
        return x


class TransformerLM_mse(nn.Module):
    def __init__(self,
                 num_tokens,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 dim_head=64,
                 ff_chunks=1,
                 ff_mult=4,
                 ff_glu=False,
                 ff_dropout=0.,
                 emb_dropout=0.,
                 attn_dropout=0.,
                 g2v_position_emb=False,
                 qkv_bias=True,
                 ):
        super(TransformerLM_mse, self).__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len)
            print("pos_emb Gene2Vec", self.pos_emb)
        else:
            self.pos_emb = RandomPositionalEmbedding(dim, max_seq_len)
            print("pos_emb no ", self.pos_emb)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_chunks, ff_mult, ff_glu, ff_dropout,
                                       attn_dropout, qkv_bias)
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

        x = self.transformer(x)  # get encodings [B, N, D]

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

import torch.nn as nn


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
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
        #                                                 nhead=heads, dim_feedforward=dim * ff_mult, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
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
