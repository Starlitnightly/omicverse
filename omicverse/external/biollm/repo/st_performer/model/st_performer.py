#!/usr/bin/env python3
# coding: utf-8
"""
@file: st_performer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2023/10/12  create file.
"""
import torch
from torch import nn
from .performer_pytorch import Performer, cast_tuple, Always
from .embedding import GeneEmbedding, Gene2VecEmbedding, ExpressBinEmbedding, OrganEmbedding, ExpressValueEmbedding, \
    SequenceEmbedding


class StPerformer(nn.Module):
    def __init__(
            self,
            *,
            gene_num,
            gene_pad_idx,
            max_seq_len,  # max length of sequence
            dim,  # dim of tokens
            depth,  # layers
            heads,  # num of heads
            dim_head=64,  # dim of heads
            is_exp_emb=False,
            exp_bins=10,  # num of express tokens, including mask id and pad id
            is_sequence_emb=False,  # sequence embedding
            sequence_num=5,  # sequence number
            is_organ_emb=False,  # organ embedding
            organ_num=0,  # organ number
            g2v_position_emb=True,  # priority: gene2vec, no embedding
            g2v_file=None,
            local_attn_heads=0,
            local_window_size=256,
            causal=False,
            ff_mult=4,
            nb_features=None,
            feature_redraw_interval=1000,
            reversible=False,
            ff_chunks=1,
            ff_glu=False,
            emb_dropout=0.,
            ff_dropout=0.,
            attn_dropout=0.,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            use_scalenorm=False,
            use_rezero=False,
            cross_attend=False,
            no_projection=False,
            auto_check_redraw=True,
            qkv_bias=False
    ):
        super().__init__()
        self.model_type = 'performer'
        self.emb_concat_style = 'sum_pool'  # concact
        self.exp_emb_use_way = 'add'
        self.gene_num = gene_num
        self.organ_num = organ_num
        self.dim = dim
        self.is_exp_emb = is_exp_emb
        self.exp_emb_type = 'bin'  # bin/value
        self.exp_bin_num = exp_bins
        self.is_organ_emb = is_organ_emb
        self.is_sequence_emb = is_sequence_emb
        self.sequence_num = sequence_num
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(emb_dropout)
        # embedding init
        # gene embedding, TODO: add knowledge embedding
        if g2v_position_emb:
            self.gene_emb = Gene2VecEmbedding(gene2vec_file=g2v_file)
        else:
            self.gene_emb = GeneEmbedding(gene_num=gene_num, embedding_dim=dim, pad_index=gene_pad_idx)
        # express embedding, bin or value
        if self.is_exp_emb:
            if self.exp_emb_type == 'bin':
                self.exp_emb = ExpressBinEmbedding(exp_bin_num=exp_bins, embedding_dim=dim)
            else:
                self.exp_emb = ExpressValueEmbedding(embedding_dim=dim)
        # organ embedding
        if self.is_organ_emb:
            self.organ_emb = OrganEmbedding(organ_num=organ_num, embedding_dim=dim)
        # sequence embedding
        if self.is_sequence_emb:
            self.sequence_emb = SequenceEmbedding(self.sequence_num, embedding_dim=dim)
        # performer init
        self.layer_pos_emb = Always(None)
        local_attn_heads = cast_tuple(local_attn_heads)
        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult,
                                   nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention,
                                   kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend,
                                   no_projection, auto_check_redraw, qkv_bias)
        self.norm = nn.LayerNorm(dim)
        # self.cell_encoder = CellEncoder(dim, emb_dropout, dim*2, max_seq_len)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def embedding_layer(self, gene_x, exp_x, is_st=False):
        if is_st:
            gene_x = torch.cat((gene_x, gene_x), 0)
        x = self.gene_emb(gene_x)  # gene_x: gene index
        if self.is_exp_emb:
            if self.exp_emb_use_way == 'add':
                exp_x = self.exp_emb(exp_x)
                x = x + exp_x
            else:
                x = x * self.exp_emb(exp_x)
        else:
            dim = x.shape[-1]
            exp_x = exp_x.unsqueeze(2).expand(-1, -1, dim)
            x = x * exp_x
        x = self.dropout(x)
        if is_st:
            pass
        return x

    def performer_layer(self, x, output_attentions=False, **kwargs):
        layer_pos_emb = self.layer_pos_emb(x)
        if output_attentions:
            x, attn_weights = self.performer(x, pos_emb=layer_pos_emb, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            return x, attn_weights
        x = x.to(torch.float32)
        x = self.performer(x, pos_emb=layer_pos_emb, output_attentions=output_attentions, **kwargs)
        # norm and to logits
        x = self.norm(x)
        # x: [batch_size, max_seq_len, dim]
        # attn_weights: [batch_size, heads, max_seq_len, max_seq_len]
        return x

    def organ_embedding_layer(self, cell_x, organ_x):
        if self.emb_concat_style == 'sum_pool':
            organ_x = self.organ_emb(organ_x)
            cell_x += organ_x
        else:
            cell_x = torch.cat([cell_x, self.organ_emb(organ_x)], dim=1)
        return cell_x

    def cell_embeding_layer(self, x):
        cell_embedding, noise_embedding = self.cell_encoder(x)
        return cell_embedding, noise_embedding

    def forward(self, gene_index, exp_x, organ_x=None, sequence_x=None, is_st=False, output_attentions=False,
                **kwargs):
        x = self.embedding_layer(gene_index, exp_x, is_st=is_st)
        if output_attentions:
            x, attn_weights = self.performer_layer(x, output_attentions, **kwargs)
            cell_emb, noise_emb = self.cell_embeding_layer(x)
            return cell_emb, noise_emb, attn_weights
        x = self.performer_layer(x, output_attentions, **kwargs)
        # cell_emb, noise_emb = self.cell_embeding_layer(x)
        return x


class StPerformerLM(nn.Module):
    def __init__(self, st_performer, gene_tokens_num, exp_bins_num, disease_tokens_num, sequence_tokens_num,
                 batch_tokens_num, is_st=False, predict_gene=True, predict_disease=False):
        super(StPerformerLM, self).__init__()
        self.model = st_performer
        self.vocab_size = self.model.gene_num
        self.dim = self.model.dim
        self.max_seq_len = self.model.max_seq_len
        self.mask_model = MaskedLanguageModel(self.dim, gene_tokens_num, exp_bins_num, disease_tokens_num,
                                              sequence_tokens_num, batch_tokens_num, self.max_seq_len)
        self.predict_gene = predict_gene
        self.predict_disease = predict_disease
        if is_st:
            self.neighbor_model = NeighborPrediction(self.dim)

    def forward(self, gene_x, exp_x, organ_x=None, sequence_x=None, is_st=False, output_attentions=False,
                **kwargs):
        attn_weights = None
        if output_attentions:
            x, attn_weights = self.model(gene_x, exp_x, organ_x, sequence_x, is_st, output_attentions, **kwargs)
        else:
            x = self.model(gene_x, exp_x, organ_x, sequence_x, is_st, output_attentions, **kwargs)
        mask_logits = self.mask_model(x, gene=self.predict_gene, disease=self.predict_disease)
        # mask_logits: [batch_size, max_seq_len, vocab_size]
        neighbor_logit = None
        if is_st:
            cell_x = x[:, 0, :]
            batch_size = int(cell_x.shape[0] / 2)
            cell_x = torch.cat([cell_x[0: batch_size], cell_x[batch_size:]], dim=-1)  # cell_x: [batch_size, dim]
            neighbor_logit = self.neighbor_model(cell_x)

        # attn_weights: [batch_size, heads, max_seq_len, max_seq_len]
        return mask_logits, neighbor_logit, attn_weights


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, dim, gene_tokens, exp_bin_tokens, desease_tokens, sequence_tokens, batch_tokens, max_seq_len):
        """
        :param dim: output size of Performer model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.gene_decoder = Decoder(dim, gene_tokens)
        self.exp_bin_decoder = Decoder(dim, exp_bin_tokens)
        self.disease_decoder = Decoder(dim, desease_tokens)
        self.sequence_decoder = Decoder(dim, sequence_tokens)
        self.batch_decoder = Decoder(dim, batch_tokens)
        self.cell_encoder = CellEncoder(dim, droup_prob=0.1, output_dim=dim, max_seq_len=max_seq_len)

    def forward(self, x, gene=True, exp_bin=True, disease=False):
        result = {}

        if gene:
            gene_logit = self.gene_decoder(x)
            result['gene_logit'] = gene_logit
        if exp_bin:
            exp_bin_logit = self.exp_bin_decoder(x)
            result['exp_bin_logit'] = exp_bin_logit
        if disease:
            disease_logit = self.disease_decoder(x)
            result['disease_logit'] = disease_logit
        emb = self.cell_encoder(x)
        result['sequence_logit'] = self.sequence_decoder(emb)
        result['batch_logit'] = self.batch_decoder(emb)
        return result


class NeighborPrediction(nn.Module):
    """
    2-class classification model : 邻居关系预测
    """

    def __init__(self, dim):
        """
        :param hidden: Performer model output size
        """
        super().__init__()
        self.linear = nn.Linear(dim * 2, 2)

    def forward(self, x):
        # x: [batch_size, dim]
        return self.linear(x)


class CellEncoder(nn.Module):
    def __init__(self, dim, droup_prob, output_dim, max_seq_len):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, dim))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=max_seq_len, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(droup_prob)
        self.fc2 = nn.Linear(in_features=512, out_features=output_dim, bias=True)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        # |x| : (batch_size, dim)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, tokens_num):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, tokens_num)
        )

    def forward(self, x):
        x = self.fc(x)  # |batch_size, seq_len, tokens_num|
        return x


class StPerformerCLS(nn.Module):
    def __init__(self, st_performer, n_class, cls_token_id, cls_pdrop=0.1, max_seq_len=16907):
        super(StPerformerCLS, self).__init__()
        self.model = st_performer
        self.vocab_size = self.model.gene_num
        self.dim = self.model.dim
        self.cls_token_id = cls_token_id
        self.max_seq_len = max_seq_len
        # Classification
        self.conv1 = nn.Conv2d(1, 1, (1, self.dim))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=self.max_seq_len, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(cls_pdrop)
        self.fc2 = nn.Linear(in_features=512, out_features=200, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(cls_pdrop)
        self.fc3 = nn.Linear(in_features=200, out_features=n_class, bias=True)
        self.dropout3 = nn.Dropout(cls_pdrop)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc1.bias, 0)

    def forward(self, gene_index, exp_x, organ_x=None, is_st=False, output_attentions=False,
                **kwargs):
        # gene_index: [batch_size, max_seq_len]
        # exp_x: [batch_size, max_len_seq]
        # organ_x: [batch_size, max_len_seq]
        attn_weights = None
        if output_attentions:
            x, attn_weights = self.model(gene_index, exp_x, organ_x, is_st, output_attentions, **kwargs)
        else:
            x = self.model(gene_index, exp_x, organ_x, is_st, output_attentions, **kwargs)
        # attn_weights: [batch_size, heads, max_seq_len, max_seq_len]
        # x = x[gene_index.eq(self.cls_token_id)]
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        # |x| : (batch_size, dim)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        # |cls_logits| : (batch_size, n_class)
        return x, attn_weights
