import torch
import torch.nn as nn
from ..utils import create_activation
from ..layer import CosformerLayer, PerformerLayer, VanillaTransformerLayer, FlowformerLayer
from ..utils.pe import select_pe_encoder

class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_hidden,
                 nhead,
                 num_layers,
                 dropout,
                 activation,
                 norm=None,
                 model_type='performer',
                 covariates_dim=0,
                 ):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        if model_type == 'cosformer':
            TransformerLayer = CosformerLayer
        elif model_type == 'performer':
            TransformerLayer = PerformerLayer
        elif model_type == 'transformer':
            TransformerLayer = VanillaTransformerLayer
        elif model_type == 'flowformer':
            TransformerLayer = FlowformerLayer
        else:
            raise NotImplementedError(f'Not implemented transformer type: {model_type}')

        for i in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    embed_dim=num_hidden, num_heads=nhead,
                    dropout=dropout)
            )

    def forward(self, x_dict, output_attentions=False):
        h = x_dict['h']
        att_list = []
        for l in range(self.num_layers):

            if l == 0:
                x_dict['base0'] = h.detach()
            if output_attentions:
                h, att = self.layers[l](h, output_attentions=True)
                att_list.append(att)
            else:
                h = self.layers[l](h)
            if l == 0:
                x_dict['base1'] = h.detach()

        if output_attentions:
            return {'hidden': h, 'attn': att_list}
        else:
            return {'hidden': h}
