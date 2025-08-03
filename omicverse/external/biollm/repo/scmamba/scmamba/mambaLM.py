# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:mambaLM.py
# @Software:PyCharm
# @Created Time:2023/12/20 10:33 AM
# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
from flash_attn.flash_attention import FlashMHA
from collections import namedtuple
import sys

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Bernoulli

from typing import Dict, Mapping, Optional, Tuple, Any, Union
import os

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir + "/../")
from scgpt.model.dsbn import DomainSpecificBatchNorm1d
import torch.distributed as dist
from scgpt.model.grad_reverse import grad_reverse
from scgpt.model.model import ExprDecoder, ClsDecoder, AdversarialDiscriminator, Similarity, CategoryValueEncoder, \
    ContinuousValueEncoder, BatchLabelEncoder, MVCDecoder
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from .BiMamba import BiMamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if if_bimamba:
        mixer_cls = partial(BiMamba, bimamba_type=bimamba_type, layer_idx=layer_idx,
                            if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx,**factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None                                                                                                                                                                                                                                                                                                                                           else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, mask=None, inference_params=None):
        # hidden_states = self.embedding(input_ids)
        # TODO: check if the residual is added
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, mask=mask, inference_params=inference_params
            ) if mask is not None else layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            initializer_cfg=None,
            pad_vocab_size_multiple: int = 1,
            device=None,
            dtype=None,
            **backbone_kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            initializer_cfg=initializer_cfg,
            **backbone_kwargs,
            **factory_kwargs,
        )
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states, mask=None, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, mask=mask, inference_params=inference_params
            ) if mask is not None else layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # #lm_logits = self.lm_head(hidden_states)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # # return CausalLMOutput(logits=lm_logits)
        return hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config = load_config_hf(pretrained_model_name)
        model = cls(**config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model


class MambaEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nlayers: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            fused_add_norm=False,
            residual_in_fp32=False,
            initializer_cfg=None,
            device=None,
            dtype=None,
            norm_scheme="post",  # "pre" or "post"
            if_bimamba=False,
            bimamba_type="none",
            if_devide_out=False,
            init_layer_scale=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # self.mamba=MambaLMHeadModel(d_model=d_model, n_layer=nlayers,fused_add_norm=fused_add_norm,
        #                  vocab_size=vocab.__len__(), norm_epsilon=norm_epsilon, rms_norm=rms_norm,
        #                  ssm_cfg=ssm_cfg, initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,**factory_kwargs)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        self.bidirectional = if_bimamba
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(nlayers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=nlayers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            num_last_tokens=0,
            inference_params=None,
            **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

                Args:
                    src: the sequence to the encoder layer (required). [batch,seq_len,feat_dim]
                    src_mask: the mask for the src sequence (optional).
                    src_key_padding_mask: the mask for the src keys per batch (optional).
                """
        # TODO:这个部分的代码集成到mamba-model里面去，作为mask检测的一部分
        if src_mask is not None:
            raise ValueError("MambaEncoderLayer does not support src_mask")
        # if not src_key_padding_mask.any().item():# return False if there isn't any True element, else return True
        #     # no padding tokens in src
        #     src_key_padding_mask_ = None
        # else:
        if src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()
        src_key_padding_mask_ = ~src_key_padding_mask
        # max_len=src_key_padding_mask_.sum(dim=1).max()
        # src=src[:,:max_len,:]
        # src=self.mamba(src,mask=src_key_padding_mask_)
        # src = self.mamba(src)
        # return src.logits
        hidden_states = src
        residual = None
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.bidirectional:
            for i in range(len(self.layers) // 2):
                # forward
                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, mask=src_mask, inference_params=inference_params
                ) if src_mask is not None else self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )

                # backward
                ### fliping
                hidden_states_flip, residual_flip = self.flip_with_padding(hidden_states, residual,
                                                                           src_key_padding_mask_)
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states_flip, None if residual_flip == None else residual_flip, mask=src_mask,
                    inference_params=inference_params
                ) if src_mask is not None else self.layers[i * 2 + 1](
                    hidden_states_flip, None if residual_flip == None else residual_flip,
                    inference_params=inference_params
                )  # TODO: check the flip in our case
                hidden_states_b, residual_b = self.flip_with_padding(hidden_states_b, residual_b, src_key_padding_mask_)
                ### fliping
                hidden_states = hidden_states_f + hidden_states_b
                residual = residual_f + residual_b
        else:
            for layer in self.layers:
                hidden_states, residual = layer(
                    hidden_states, residual, mask=src_mask, inference_params=inference_params
                ) if src_mask is not None else layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # #lm_logits = self.lm_head(hidden_states)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # # return CausalLMOutput(logits=lm_logits)
        return hidden_states

    def flip_with_padding(self, hidden_states, residual, src_key_padding_mask_):
        seq_len_batch = src_key_padding_mask_.sum(dim=1)
        hidden_states_flip = []
        if residual is not None:
            residual_flip = []
            for seq_len, hs, res in zip(seq_len_batch, hidden_states, residual):
                hs_f = hs[:seq_len, :].flip([0])
                res_f = res[:seq_len, :].flip([0])
                hidden_states_flip.append(torch.cat([hs_f, hs[seq_len:]], dim=0).unsqueeze(0))  # [1,seq_len,emb]
                residual_flip.append(torch.cat([res_f, res[seq_len:]], dim=0).unsqueeze(0))  # [1,seq_len,emb]
            hidden_states_flip = torch.cat(hidden_states_flip, dim=0)
            residual_flip = torch.cat(residual_flip, dim=0)
        else:
            for seq_len, hs in zip(seq_len_batch, hidden_states):
                hs_f = hs[:seq_len, :].flip([0])
                hidden_states_flip.append(torch.cat([hs_f, hs[seq_len:]], dim=0).unsqueeze(0))  # [1,seq_len,emb]
            hidden_states_flip = torch.cat(hidden_states_flip, dim=0)
            residual_flip = None
        return hidden_states_flip, residual_flip


class MambaModel(nn.Module):
    def __init__(
            self,
            ntoken: int,
            d_model: int,
            nlayers: int,
            nlayers_cls: int = 3,
            device=None,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            fused_add_norm=False,
            residual_in_fp32=False,
            initializer_cfg=None,
            n_cls: int = 1,
            vocab: Any = None,
            dropout: float = 0.5,
            pad_token: str = "<pad>",
            pad_value: int = 0,
            do_mvc: bool = False,
            do_dab: bool = False,
            do_cce: bool = False,
            use_batch_labels: bool = False,
            num_batch_labels: Optional[int] = None,
            domain_spec_batchnorm: Union[bool, str] = False,
            input_emb_style: str = "continuous",
            n_input_bins: Optional[int] = None,
            cell_emb_style: str = "cls",
            mvc_decoder_style: str = "inner product",
            ecs_threshold: float = 0.3,
            explicit_zero_prob: bool = False,
            pre_norm: bool = False,
            do_pretrain=False,
            topo_graph: bool = False,
            if_devide_out=False,
            init_layer_scale=None,
            if_bimamba=False,
            bimamba_type="none",
            do_pert=False,
            pert_pad_id: int = 2,
            token_emb_freeze=False
    ):
        super().__init__()
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.pert_pad_id = pert_pad_id
        self.do_pert = do_pert
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.do_pretrain = do_pretrain
        self.token_emb_freeze = token_emb_freeze
        self.norm_scheme = "pre" if pre_norm else "post"
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool", "final", 'attn']:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        # TODO: add dropout in the GeneEncoder
        if if_bimamba:
            self.mamba_encoder = MambaEncoderLayer(d_model=d_model, nlayers=nlayers, device=device,
                                                   fused_add_norm=fused_add_norm, norm_epsilon=norm_epsilon,
                                                   rms_norm=rms_norm,
                                                   ssm_cfg=ssm_cfg, initializer_cfg=initializer_cfg,
                                                   residual_in_fp32=residual_in_fp32, bimamba_type=bimamba_type,
                                                   if_bimamba=if_bimamba, if_devide_out=if_devide_out,
                                                   init_layer_scale=init_layer_scale)
        else:
            self.mamba_encoder = MambaEncoderLayer(d_model=d_model, nlayers=nlayers, device=device,
                                                   fused_add_norm=fused_add_norm,
                                                   norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                                                   ssm_cfg=ssm_cfg, initializer_cfg=initializer_cfg,
                                                   residual_in_fp32=residual_in_fp32)
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token],
                                   dropout=dropout)  # embedding module is integrated in mamba encoder
        if do_pert:
            self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value, dropout=dropout
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Batch Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        if domain_spec_batchnorm:
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        else:
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )
        if n_cls > 1:
            self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
        if do_cce:
            self.sim = Similarity(temp=0.5)  # TODO: auto set temp
            self.creterion_cce = nn.CrossEntropyLoss()

        if self.cell_emb_style == 'attn':
            self.dropout = nn.Dropout(dropout)
            self.cell_norm = nn.LayerNorm(d_model, eps=1e-5)
            self.cell_attn = nn.Linear(d_model, 1)
        self.init_weights()

        self.topo_graph = topo_graph
        if self.do_pretrain and topo_graph:
            vocab_size = len(vocab)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.encoder.embedding.weight  # tie weight

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: initialize the embedding using pretrain scGPT
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
            self,
            src: Tensor,
            values: Tensor,
            src_key_padding_mask: Tensor,
            batch_labels: Optional[Tensor] = None,  # (batch,)
            sorted_layer_idx=None,
            input_pert_flags=None,
    ) -> Tensor:
        self._check_batch_labels(batch_labels)
        if self.topo_graph and self.token_emb_freeze:
            with torch.no_grad():
                src = self.encoder(src, sorted_layer_idx)  # (batch, seq_len, embsize)
        else:
            src = self.encoder(src, sorted_layer_idx)
            if not self.topo_graph:
                self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values
        if self.do_pert:
            perts = self.pert_encoder(input_pert_flags)
            total_embs = total_embs + perts
        if self.domain_spec_batchnorm:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        else:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.mamba_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
            self, layer_output: Tensor, weights: Tensor = None, src_key_padding_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        # elif self.cell_emb_style == "avg-pool":
        #     cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)
        elif self.cell_emb_style in ["final", "avg-pool"]:
            if not src_key_padding_mask.any().item():  # return False if there isn't any True element, else return True
                # no padding tokens in src
                if self.cell_emb_style == 'final':
                    cell_emb = layer_output[:, -1, :]  # (batch, embsize)
                elif self.cell_emb_style == 'avg-pool':
                    cell_emb = torch.mean(layer_output, dim=1)
            else:
                # NOTE: uses mask 0 for padding tokens, which is the opposite
                if src_key_padding_mask.dtype != torch.bool:
                    src_key_padding_mask = src_key_padding_mask.bool()
                src_key_padding_mask_ = ~src_key_padding_mask
                final_token = src_key_padding_mask_.sum(dim=1)
                if self.cell_emb_style == 'final':
                    cell_emb = torch.cat(
                        [seq[gene_idx, :].unsqueeze(0) for seq, gene_idx in zip(layer_output, final_token - 1)], dim=0)
                elif self.cell_emb_style == 'avg-pool':
                    cell_emb = torch.cat([seq[:gene_idx, :].mean(dim=0).unsqueeze(0) for seq, gene_idx in
                                          zip(layer_output, final_token - 1)], dim=0)
        elif self.cell_emb_style == 'attn':
            # layer_output = self.cell_norm(layer_output)
            # cell_emb = self.cell_emb_attn(layer_output, key_padding_mask=src_key_padding_mask)
            # cell_emb = layer_output + self.dropout(cell_emb)
            attn_scores = self.cell_attn(layer_output)
            attn_scores[src_key_padding_mask] = float('-inf')
            attn_weights = F.softmax(attn_scores, dim=1)
            cell_emb = torch.sum(attn_weights * layer_output, dim=1)
            cell_emb = self.cell_norm(cell_emb)
            cell_emb = self.dropout(cell_emb)
        return cell_emb

    def generate(
            self,
            cell_emb: Tensor,
            src: Tensor,
            values: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            gen_iters: int = 1,
            batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (batch, embsize)
            src(:obj:`Tensor`): shape (batch, seq_len)
            values(:obj:`Tensor`): shape (batch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (batch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            batch_labels(:obj:`Tensor`): shape (batch,), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration
        try:
            self._check_batch_labels(batch_labels)
        except:
            import warnings

            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.encoder(src)  # (batch, seq_len, embsize)

        if values is not None:
            values = self.value_encoder(values)  # (batch, seq_len, embsize)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        if self.domain_spec_batchnorm:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        else:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        mamba_output = self.mamba_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        mlm_output = self.decoder(
            mamba_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    mamba_output,
                    batch_emb.unsqueeze(1).repeat(1, mamba_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        output = mlm_output["pred"]  # (batch, seq_len)

        return output  # (batch, seq_len)

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )

    def forward(
            self,
            src: Tensor,
            values: Tensor,
            src_key_padding_mask: Tensor,
            batch_labels: Optional[Tensor] = None,
            CLS: bool = False,
            CCE: bool = False,
            MVC: bool = False,
            ECS: bool = False,
            do_sample: bool = False,
            input_sorted_gene=None,
            topo_padding_mask=None,
            sorted_layer_idx=None,
            input_pert_flags=None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        mamba_output = self._encode(
            src, values, src_key_padding_mask, batch_labels, sorted_layer_idx, input_pert_flags=input_pert_flags
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        output = {}
        mlm_output = self.decoder(
            mamba_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    mamba_output,
                    batch_emb.unsqueeze(1).repeat(1, mamba_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(mamba_output, values, src_key_padding_mask=src_key_padding_mask)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if CCE:
            cell1 = cell_emb
            mamba_output2 = self._encode(
                src, values, src_key_padding_mask, batch_labels, sorted_layer_idx
            )
            cell2 = self._get_cell_emb_from_layer(mamba_output2, src_key_padding_mask=src_key_padding_mask)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if self.do_pretrain and self.topo_graph:
            output["lm_logit"] = self.topo_forward(input_sorted_gene=input_sorted_gene,
                                                   topo_padding_mask=topo_padding_mask,
                                                   sorted_layer_idx=sorted_layer_idx)
        if MVC:
            if self.topo_graph:
                self.cur_gene_token_embs = self.encoder(src, sorted_layer_idx)
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        return output

    def topo_forward(self, input_sorted_gene, topo_padding_mask, sorted_layer_idx=None):
        '''
        Used for topological prediction.
        Args:
            input_sorted_gene: the extended and sorted gene ids
            masked_input_sorted_gene: masked 'input_sorted_gene' by layer
            sorted_layer_idx: topo layer idx of input_sorted_gene
        Returns:
        '''
        emb = self.encoder(input_sorted_gene, sorted_layer_idx)  # [bsz,seq_len,emb_size], with mask
        h = self.mamba_encoder(emb, src_key_padding_mask=topo_padding_mask)
        lm_logit = self.lm_head(h)
        return lm_logit

    def encode_batch(
            self,
            src: Tensor,
            values: Tensor,
            src_key_padding_mask: Tensor,
            batch_size: int,
            batch_labels: Optional[Tensor] = None,
            output_to_cpu: bool = True,
            return_np: bool = False,
            time_step: Optional[int] = None,
            sorted_layer_idx=None
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = ((N, self.d_model))
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i: i + batch_size].to(device),
                values[i: i + batch_size].to(device),
                src_key_padding_mask[i: i + batch_size].to(device),
                batch_labels[i: i + batch_size].to(device)
                if batch_labels is not None
                else None,
                sorted_layer_idx=sorted_layer_idx[i: i + batch_size].to(
                    device) if sorted_layer_idx is not None else None
            )
            cell_emb = self._get_cell_emb_from_layer(raw_output,
                                                     src_key_padding_mask=src_key_padding_mask[i: i + batch_size].to(
                                                         device))

            cell_emb = cell_emb.detach()
            if output_to_cpu:
                cell_emb = cell_emb.cpu()
            if return_np:
                cell_emb = cell_emb.numpy()
            outputs[i: i + batch_size] = cell_emb
        return outputs

    def pred_perturb(
            self,
            batch_data,
            include_zero_gene="batch-wise",
            gene_ids=None,
            amp=True,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = torch.Tensor(np.stack(batch_data.pert_flag)).long().to(device)
        sorted_layer_idx = batch_data.sorted_layer_idx

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self(
                    mapped_input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    sorted_layer_idx=sorted_layer_idx,
                    input_pert_flags=input_pert_flags,
                )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values


class TopoLayerEncoding(torch.nn.Module):
    def __init__(self, d_model, max_layer=100):
        super(TopoLayerEncoding, self).__init__()
        pe = torch.zeros(max_layer, d_model)
        position = torch.arange(0, max_layer, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, layer_index):
        x = x + self.pe[layer_index].view(x.size())
        return x


class GeneEncoder(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            dropout: Optional[float] = 0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_emb = TopoLayerEncoding(embedding_dim)

    def forward(self, x: Tensor, sorted_layer_idx=None) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        if sorted_layer_idx is not None:
            x = self.layer_emb(x, sorted_layer_idx)
        x = self.enc_norm(x)
        return self.dropout(x)


def map_raw_id_to_vocab_id(
        raw_ids: Union[np.ndarray, torch.Tensor],
        gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)
