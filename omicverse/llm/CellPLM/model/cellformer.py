import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..embedder import OmicsEmbeddingLayer
from ..utils.mask import MaskBuilder, NullMaskBuilder, HiddenMaskBuilder
from ..encoder import setup_encoder
from ..decoder import setup_decoder
from ..latent import LatentModel, PreLatentNorm
from ..latent.adversarial import AdversarialLatentLayer
from ..objective import Objectives
from ..head import setup_head

class OmicsFormer(nn.Module):
    def __init__(self, gene_list, enc_mod, enc_hid, enc_layers, post_latent_dim, dec_mod, dec_hid, dec_layers,
                 out_dim, batch_num=0, dataset_num=0, platform_num=0, mask_type='input', model_dropout=0.1,
                 activation='gelu', norm='layernorm', enc_head=8, mask_node_rate=0.5,
                 mask_feature_rate=0.8, drop_node_rate=0., max_batch_size=2000, cat_dim=None, conti_dim=None,
                 pe_type='sin', cat_pe=True,
                 gene_emb=None, latent_mod='vae', w_li=1., w_en=1., w_ce=1.,
                 head_type=None, dsbn=False, ecs=False, dar=False, input_covariate=False,
                 num_clusters=16, dae=True, lamda=0.5, mask_beta=False, **kwargs):
        super(OmicsFormer, self).__init__()

        self.embedder = OmicsEmbeddingLayer(gene_list, enc_hid, norm, activation, model_dropout,
                                            pe_type, cat_pe, gene_emb, inject_covariate=input_covariate, batch_num=batch_num)
        self.gene_set = set(gene_list)
        self.mask_type = mask_type
        if mask_node_rate > 0 and mask_feature_rate > 0:
            if mask_type == 'input':
                self.mask_model = MaskBuilder(mask_node_rate, mask_feature_rate, drop_node_rate, max_batch_size, mask_beta)
            elif mask_type == 'hidden':
                self.mask_model = HiddenMaskBuilder(mask_node_rate, mask_feature_rate, drop_node_rate, max_batch_size)
            else:
                raise NotImplementedError(f"Only support mask_type in ['input', 'hidden'], but got {mask_type}")
        else:
            self.mask_model = NullMaskBuilder(drop_node_rate, max_batch_size)
        self.encoder = setup_encoder(enc_mod, enc_hid, enc_layers, model_dropout, activation, norm, enc_head)

        self.latent = LatentModel()
        self.latent_mod = latent_mod
        if latent_mod=='vae':
            self.latent.add_layer(type='vae', enc_hid=enc_hid, latent_dim=post_latent_dim)
        elif latent_mod=='ae':
            self.latent.add_layer(type='merge', conti_dim=enc_hid, cat_dim=0, post_latent_dim=post_latent_dim)
        elif latent_mod=='gmvae':
            self.latent.add_layer(type='gmvae', enc_hid=enc_hid, latent_dim=post_latent_dim, batch_num=batch_num,
                                  w_li=w_li, w_en=w_en, w_ce=w_ce, dropout=model_dropout, num_layers=dec_layers,
                                  num_clusters=num_clusters, lamda=lamda)
        elif latent_mod=='split':
            self.latent.add_layer(type='split', enc_hid=enc_hid, latent_dim=None, conti_dim=conti_dim, cat_dim=cat_dim)
            self.latent.add_layer(type='merge', conti_dim=conti_dim, cat_dim=cat_dim, post_latent_dim=post_latent_dim)
        elif latent_mod == 'none':
            post_latent_dim = enc_hid
        else:
            raise NotImplementedError(f'Latent mod "{latent_mod}" is not implemented.')
        if latent_mod is not None:
            if dar:
                self.latent.add_layer(type='adversarial', input_dims=np.arange(post_latent_dim), label_key='batch',
                                      discriminator_hidden=64, disc_lr=1e-3,
                                      target_classes=batch_num)
            if ecs:
                self.latent.add_layer(type='ecs')

        self.head_type = head_type
        if head_type is not None:
            self.head = setup_head(head_type, post_latent_dim, dec_hid, out_dim, dec_layers,
                                   model_dropout, norm, batch_num=batch_num)
        else:
            self.decoder = setup_decoder(dec_mod, post_latent_dim, dec_hid, out_dim, dec_layers,
                                         model_dropout, norm, batch_num=batch_num, dataset_num=dataset_num, platform_num=platform_num)
            if 'objective' in kwargs:
                self.objective = Objectives([{'type': kwargs['objective']}])
            else:
                if 'nb' in dec_mod:
                    self.objective = Objectives([{'type': 'nb', 'dae': dae}])
                else:
                    self.objective = Objectives([{'type': 'recon'}])

        if dsbn:
            self.pre_latent_norm = PreLatentNorm('dsbn', enc_hid, dataset_num)
        else:
            self.pre_latent_norm = PreLatentNorm('ln', enc_hid)
        # self.post_latent_norm = nn.LayerNorm(post_latent_dim, dataset_num)

    def forward(self, x_dict, input_gene_list=None, d_iter=False):
        if self.mask_type == 'input':
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict['h'] = self.embedder(x_dict, input_gene_list)
        if self.mask_type == 'hidden':
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict['h'] = self.encoder(x_dict)['hidden']
        x_dict['h'] = self.pre_latent_norm(x_dict)
        x_dict['h'], latent_loss = self.latent(x_dict)

        # x_dict['h'] = self.post_latent_norm(x_dict['h'])
        # if 'ecs' in x_dict:
        #     x_dict['h'] = self.latent_norm(x_dict['h'])

        if d_iter:
            return self.latent.d_train(x_dict)
        else:
            if self.head_type is not None:
                out_dict, loss = self.head(x_dict)
                out_dict['latent_loss'] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                out_dict['target_loss'] = loss.item()
            else:
                out_dict = self.decoder(x_dict)
                loss = latent_loss + self.objective(out_dict, x_dict) #/ 1e4
                out_dict['latent_loss'] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                out_dict['target_loss'] = loss.item() - out_dict['latent_loss']
            return out_dict, loss

    def nondisc_parameters(self):
        other_params = []
        for pname, p in self.named_parameters():
            if 'discriminator' not in pname:
                other_params += [p]
            else:
                print(pname)
        return other_params
