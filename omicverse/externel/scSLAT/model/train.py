r"""
Training functions with different strategy
"""
from math import ceil
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from .loss import feature_reconstruct_loss


def train_GAN(wdiscriminator:torch.nn.Module,
                optimizer_d:torch.optim.Optimizer,
                embds:List[torch.Tensor],
                batch_d_per_iter:Optional[int]=5,
                anchor_scale:Optional[float]=0.8
    )->torch.Tensor:
    r"""
    GAN training strategy
    
    Parameters
    ----------
    wdiscriminator
        WGAN
    optimizer_d
        WGAN parameters
    embds
        list of LGCN embd
    batch_d_per_iter
        WGAN train iter numbers
    anchor_scale
        ratio of anchor cells
    """
    embd0, embd1 = embds
    
    wdiscriminator.train()
    anchor_size = ceil(embd1.size(0)*anchor_scale)

    for j in range(batch_d_per_iter):
        w0 = wdiscriminator(embd0)
        w1 = wdiscriminator(embd1)
        anchor1 = w1.view(-1).argsort(descending=True)[: anchor_size]
        anchor0 = w0.view(-1).argsort(descending=False)[: anchor_size]
        embd0_anchor = embd0[anchor0, :].clone().detach()
        embd1_anchor = embd1[anchor1, :].clone().detach()
        optimizer_d.zero_grad()
        loss = -torch.mean(wdiscriminator(embd0_anchor)) + torch.mean(wdiscriminator(embd1_anchor))
        loss.backward()
        optimizer_d.step()
        for p in wdiscriminator.parameters():
            p.data.clamp_(-0.1, 0.1)
    w0 = wdiscriminator(embd0)
    w1 = wdiscriminator(embd1)
    anchor1 = w1.view(-1).argsort(descending=True)[: anchor_size]
    anchor0 = w0.view(-1).argsort(descending=False)[: anchor_size]
    embd0_anchor = embd0[anchor0, :]
    embd1_anchor = embd1[anchor1, :]
    loss = -torch.mean(wdiscriminator(embd1_anchor))
    return loss


def train_reconstruct(recon_models:torch.nn.Module,
                        optimizer_recons,
                        embds:List[torch.Tensor],
                        features:List[torch.Tensor],
                        batch_r_per_iter:Optional[int]=10
    )->torch.Tensor:
    r"""
    Data reconstruction network training strategy
    
    Parameters
    ----------
    recon_models
        list of reconstruction model
    optimizer_recons
        list of reconstruction optimizer
    embds
        list of LGCN embd
    features
        list of rae node features
    batch_d_per_iter
        WGAN train iter numbers
    """
    recon_model0, recon_model1 = recon_models
    optimizer_recon0, optimizer_recon1 = optimizer_recons
    embd0, embd1 = embds
    
    recon_model0.train()
    recon_model1.train()
    embd0_copy = embd0.clone().detach()
    embd1_copy = embd1.clone().detach()    
    for t in range(batch_r_per_iter):
        optimizer_recon0.zero_grad()
        loss = feature_reconstruct_loss(embd0_copy, features[0], recon_model0)
        loss.backward()
        optimizer_recon0.step()
    for t in range(batch_r_per_iter):
        optimizer_recon1.zero_grad()
        loss = feature_reconstruct_loss(embd1_copy, features[1], recon_model1)
        loss.backward()
        optimizer_recon1.step()
    loss = 0.5 * feature_reconstruct_loss(embd0, features[0], recon_model0) + 0.5 * feature_reconstruct_loss(embd1, features[1], recon_model1)

    return loss
    
        
def check_align(embds:List[torch.Tensor],
                ground_truth:torch.Tensor,
                k:Optional[int]=[5,10],
                mode:Optional[str]='cosine'
    )->List[float]:
    r"""
    Check embedding correspondence in given distance (default cosine similarity) under ground truth
    
    Parameters
    -----------
    embds
        List of graph features, each element is (node_num, feature_dim)
    ground_truth
        mapping ground_truth (2, node_num)
    k
        list of top k (only support 2 elements yet)
    mode
        distance quota
    """
    embd0, embd1 = embds
    assert k[1] > k[0]
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())
    
    cossim = torch.zeros(embd1.size(0), embd0.size(0))
    for i in range(embd1.size(0)):
        cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
    
    ind = cossim.argsort(dim=1, descending=True)[:, :k[1]]
    a1 = 0
    ak0 = 0
    ak1 = 0
    for i, node in enumerate(g_list):
        if ind[node, 0].item() == g_map[node]:
            a1 += 1
            ak0 += 1
            ak1 += 1
        else:
            for j in range(1, k[0]):
                if ind[node, j].item() == g_map[node]:
                    ak0 += 1
                    ak1 += 1
                    break
                else:
                    for l in range(k[0], k[1]):
                        if ind[node, l].item() == g_map[node]:
                            ak1 += 1
                        break

    a1 /= len(g_list)
    ak0 /= len(g_list)
    ak1 /= len(g_list)
    print(f'H@1:{a1*100}; H@{k[0]}:{ak0*100}; H@{k[1]}:{ak1*100}')
    return a1, ak0, ak1