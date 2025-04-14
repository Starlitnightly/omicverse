from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.generation.utils import SampleDecoderOnlyOutput


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config

    def forward(self, x, layer_past=None):
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        B, T, C = k.size()
        _, Tq, _ = q.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
   
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=2)
            v = torch.cat((past_value, v), dim=2)
        B, nh, T, hs = k.size()
        
        if Tq < T:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C) # re-assemble all head outputs side by side
    
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None):
        a,kv_cache = self.attn(self.ln_1(x),layer_past)
        x = x+a
        x = x + self.mlp(self.ln_2(x))
        return x,kv_cache

@dataclass
class MulanConfig:
    block_size: int = 1000
    vocab_size: int = 1011 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True 
    train_mode: str = 'pretrian'
    expression_level: int = 10
    ele: int = 0


class scMulanModel_kv(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.expression_level is not None
        assert config.ele == 1
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wee = nn.Embedding(config.expression_level + 1, config.n_embd), # +1 for non gene tokens
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # expr level

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        
    def get_num_params(self): 
        n_params = sum(p.numel() for p in self.parameters())

        return n_params
    
    def forward(self, 
                idx = None,
                inputs_embeds = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                targets = None, 
                xlen = None,
                x_prefix_len = None,
                x_expr = None,
                y_expr = None,
                return_hidden = False,
                ):
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        # kvcache now
        presents_kv = () 
        if idx is not None:
            b, t = idx.size()
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if inputs_embeds is not None:
            tok_emb = inputs_embeds
        expr_emb = self.transformer.wee(x_expr) # expression embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb + expr_emb)
        for i,(block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            x,layer_present = block(x,layer_past)
            presents_kv += (layer_present,)
        x = self.transformer.ln_f(x)

        logits_cls = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        logits_exp = self.epx_head(x[:, [-1], :])
        loss = None
        loss_cls = None
        loss_exp = None
        if return_hidden:
            return logits_cls,logits_exp, x, presents_kv
        return logits_cls, logits_exp, loss, loss_cls, loss_exp, presents_kv

    @torch.no_grad()
    def generate_cellGenesis(self, 
                      input_ids,
                      expression_level,
                      max_new_tokens, 
                      ignore_Idx = None, 
                      top_k = None, 
                      return_dict_in_generate = False,
                      return_hidden = False,
                      gamma = 1):

        scores = ()
        hidden_states = ()
        past_key_values = None
        idx_cond = input_ids
        output_idx = idx_cond
        
        idx_expr = expression_level # 输入的token长度，一开始是3，后面变成nexttoken_expr 1。
        output_expr = expression_level # 输出的 token长度，一开始是3，后面变成nexttoken_expr 1。

        while True:
            if return_hidden:
                logits_cls,logits_exp,hidden,past_key_values = self(idx = idx_cond, x_expr = idx_expr, return_hidden = True, past_key_values = past_key_values)
                hidden_states += (hidden,)
            else:
                logits_cls, logits_exp, loss, loss_cls, loss_exp, past_key_values = self(idx = idx_cond, x_expr = idx_expr,past_key_values = past_key_values)
            logits_cls = logits_cls[:,-1,:] # (B,C)
            logits_exp = logits_exp[:,-1,:] # (B,C)

            if ignore_Idx is not None:
                # return logits, ignore_Idx
                logits_cls[:,ignore_Idx] = float('-inf')
            logits_cls[:,output_idx[0,:]] = float('-inf')
            
            if top_k is not None:
                v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)))
                logits_cls[logits_cls < v[:, [-1]]] = float('-inf')
            
            next_token_scores = logits_cls

            if return_dict_in_generate:
                scores += (next_token_scores,)
            
            probs = F.softmax(logits_cls, dim=-1) #(B,C)
            probs[:,0] = gamma*probs[:,0]
           
            next_tokens = torch.multinomial(probs,num_samples=1) #(B,1)
            next_token_ele = logits_exp[torch.arange(logits_exp.size(0)),next_tokens.squeeze(1)].unsqueeze(1) # (B,1)
            bin_ele_next_token = torch.clamp(torch.round(next_token_ele), 0, 10).int()
            output_idx = torch.cat((output_idx,next_tokens),dim=1)
            output_expr = torch.cat((output_expr,bin_ele_next_token),dim=1)
            idx_cond = next_tokens
            idx_expr = bin_ele_next_token
            
            if next_tokens == 0 or len(output_idx[0]) >= max_new_tokens:
                break
            
        if return_dict_in_generate:
            return SampleDecoderOutput(
                sequences=output_idx,
                scores=scores,
                hidden_states=hidden_states,
                expression=output_expr,
                )
        elif return_hidden:
            return output_idx, output_expr, hidden_states
        else:return output_idx, output_expr

@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None