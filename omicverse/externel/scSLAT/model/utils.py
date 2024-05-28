r"""
Useful functions
"""
import time
import math
from typing import List, Mapping, Optional, Union
from pathlib import Path



import scanpy as sc
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData

from ..utils import get_free_gpu
from .train import train_GAN, train_reconstruct
from .graphmodel import LGCN, LGCN_mlp, WDiscriminator, ReconDNN
from .loaddata import load_anndatas
from .preprocess import Cal_Spatial_Net


def run_LGCN(features:List,
            edges:List,
            LGCN_layer:Optional[int]=2
    ):
    """
    Run LGCN model
    
    Parameters
    ----------
    features
        list of graph node features
    edges
        list of graph edges
    LGCN_layer
        LGCN layer number, we suggest set 2 for barcode based and 4 for fluorescence based
    """
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)
    
    LGCN_model =LGCN(input_size=features[0].size(1), K=LGCN_layer).to(device=device)
    
    time1 = time.time()
    embd0 = LGCN_model(features[0], edges[0])
    embd1 = LGCN_model(features[1], edges[1])
    
    run_time = time.time() - time1
    print(f'LGCN time: {run_time}')
    return embd0, embd1, run_time   


def run_SLAT(features:List,
            edges:List,
            epochs:Optional[int]=6,
            LGCN_layer:Optional[int]=1,
            mlp_hidden:Optional[int]=256,
            hidden_size:Optional[int]=2048,
            alpha:Optional[float]=0.01,
            anchor_scale:Optional[float]=0.8,
            lr_mlp:Optional[float]=0.0001,
            lr_wd:Optional[float]=0.0001,
            lr_recon:Optional[float]=0.01,
            batch_d_per_iter:Optional[int]=5,
            batch_r_per_iter:Optional[int]=10
    ) -> List:
    r"""
    Run SLAT model
    
    Parameters
    ----------
    features
        list of graph node features
    edges
        list of graph edges
    epochs
        epoch number of SLAT (not exceed 10)
    LGCN_layer
        LGCN layer number, we suggest set 1 for barcode based and 4 for fluorescence based
    mlp_hidden
        MLP hidden layer size
    hidden_size
        size of LGCN output
    transform
        if use transform
    alpha
        scale of loss
    anchor_scale
        ratio of cells selected as pairs
    lr_mlp
        learning rate of MLP
    lr_wd
        learning rate of WGAN discriminator
    lr_recon
        learning rate of reconstruction
    batch_d_per_iter
        batch number for WGAN train per iter
    batch_r_per_iter
        batch number for reconstruct train per iter
    
    Return
    ----------
    embd0
        cell embedding of dataset1
    embd1
        cell embedding of dataset2
    time
        run time of SLAT model
    """
    
    feature_size = features[0].size(1)
    feature_output_size = hidden_size
    
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)

    feature_size = features[0].size(1)
    feature_output_size = hidden_size

    LGCN_model = LGCN_mlp(feature_size, hidden_size, K=LGCN_layer, hidden_size=mlp_hidden).to(device)
    optimizer_LGCN = torch.optim.Adam(LGCN_model.parameters(), lr=lr_mlp, weight_decay=5e-4)

    wdiscriminator = WDiscriminator(feature_output_size).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=lr_wd, weight_decay=5e-4)

    recon_model0 = ReconDNN(feature_output_size, feature_size).to(device)
    recon_model1 = ReconDNN(feature_output_size, feature_size).to(device)
    optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=lr_recon, weight_decay=5e-4)


    print('Running')
    time1 = time.time()
    for i in range(1, epochs + 1):
        print(f'---------- epochs: {i} ----------')
        
        LGCN_model.train()
        optimizer_LGCN.zero_grad()
        
        embd0 = LGCN_model(features[0], edges[0])
        embd1 = LGCN_model(features[1], edges[1])

        loss = train_GAN(wdiscriminator, optimizer_wd, [embd0,embd1], batch_d_per_iter=batch_d_per_iter, anchor_scale=anchor_scale)
        loss_feature = train_reconstruct([recon_model0, recon_model1], [optimizer_recon0, optimizer_recon1], [embd0,embd1], features,batch_r_per_iter=batch_r_per_iter)
        loss = (1-alpha) * loss + alpha * loss_feature
        
        loss.backward()
        optimizer_LGCN.step()
        
    LGCN_model.eval()
    embd0 = LGCN_model(features[0], edges[0])
    embd1 = LGCN_model(features[1], edges[1])

    time2 = time.time()
    print('Training model time: %.2f' % (time2-time1))
    # torch.cuda.empty_cache()
    return embd0, embd1, time2-time1


def spatial_match(embds:List[torch.Tensor],
                  reorder:Optional[bool]=True,
                  smooth:Optional[bool]=True,
                  smooth_range:Optional[int]=20,
                  scale_coord:Optional[bool]=True,
                  adatas:Optional[List[AnnData]]=None,
                  return_euclid:Optional[bool]=False,
                  verbose:Optional[bool]=False,
                  get_null_distri:Optional[bool]=False
    )-> List[Union[np.ndarray,torch.Tensor]]:
    r"""
    Use embedding to match cells from different datasets based on cosine similarity
    
    Parameters
    ----------
    embds
        list of embeddings
    reorder
        if reorder embedding by cell numbers
    smooth
        if smooth the mapping by Euclid distance
    smooth_range
        use how many candidates to do smooth
    scale_coord
        if scale the coordinate to [0,1]
    adatas
        list of adata object
    verbose
        if print log
    get_null_distri
        if get null distribution of cosine similarity
    
    Note
    ----------
    Automatically use larger dataset as source
    
    Return
    ----------
    Best matching, Top n matching and cosine similarity matrix of top n  
    
    Note
    ----------
    Use faiss to accelerate, refer https://github.com/facebookresearch/faiss/issues/95
    """
    import faiss
    if reorder and embds[0].shape[0] < embds[1].shape[0]:
        embd0 = embds[1]
        embd1 = embds[0]
        adatas = adatas[::-1] if adatas is not None else None
    else:
        embd0 = embds[0]
        embd1 = embds[1]
        
    if get_null_distri:
        embd0 = torch.tensor(embd0)
        embd1 = torch.tensor(embd1)
        sample1_index = torch.randint(0, embd0.shape[0], (1000,))
        sample2_index = torch.randint(0, embd1.shape[0], (1000,))
        cos = torch.nn.CosineSimilarity(dim=1)
        null_distri = cos(embd0[sample1_index], embd1[sample2_index])

    index = faiss.index_factory(embd1.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    embd0_np = embd0.detach().cpu().numpy() if torch.is_tensor(embd0) else embd0
    embd1_np = embd1.detach().cpu().numpy() if torch.is_tensor(embd1) else embd1
    embd0_np = embd0_np.copy().astype('float32')
    embd1_np = embd1_np.copy().astype('float32')
    faiss.normalize_L2(embd0_np)
    faiss.normalize_L2(embd1_np)
    index.add(embd0_np)
    similarity, order = index.search(embd1_np, smooth_range)
    best = []
    if smooth and adatas != None:
        if verbose:
            print('Smoothing mapping, make sure object is in same direction')
        if scale_coord:
            # scale spatial coordinate of every adata to [0,1]
            adata1_coord = adatas[0].obsm['spatial'].copy()
            adata2_coord = adatas[1].obsm['spatial'].copy()
            for i in range(2):
                    adata1_coord[:,i] = (adata1_coord[:,i]-np.min(adata1_coord[:,i]))/(np.max(adata1_coord[:,i])-np.min(adata1_coord[:,i]))
                    adata2_coord[:,i] = (adata2_coord[:,i]-np.min(adata2_coord[:,i]))/(np.max(adata2_coord[:,i])-np.min(adata2_coord[:,i]))
        dis_list = []
        for query in range(embd1_np.shape[0]):
            ref_list = order[query, :smooth_range]
            dis = euclidean_distances(adata2_coord[query,:].reshape(1, -1),
                                      adata1_coord[ref_list,:])
            dis_list.append(dis)
            best.append(ref_list[np.argmin(dis)])
    else:
        best = order[:,0]

    if return_euclid and smooth and adatas != None:
        dis_array = np.squeeze(np.array(dis_list))
        if get_null_distri:
            return np.array(best), order, similarity, dis_array, null_distri
        else:
            return np.array(best), order, similarity, dis_array
    else:
        return np.array(best), order, similarity
    
    
def probabilistic_match(cos_cutoff:float=0.6, euc_cutoff:int=5, **kargs)-> List[List[int]]:
    
    best, index, similarity, eucli_array, null_distri = \
        spatial_match(**kargs, return_euclid=True, get_null_distri=True)
    # filter the cosine similarity via p_value
    # mask1 = similarity > cos_cutoff
    null_distri = np.sort(null_distri)
    p_val = 1 - np.searchsorted(null_distri, similarity) / null_distri.shape[0]
    mask1 = p_val < 0.05
    
    # filter the euclidean distance
    sorted_indices = np.argpartition(eucli_array, euc_cutoff, axis=1)[:, :euc_cutoff]
    mask2 = np.full(eucli_array.shape, False, dtype=bool)
    mask2[np.arange(eucli_array.shape[0])[:, np.newaxis], sorted_indices] = True
    
    mask_mat = np.logical_and(mask1, mask2)
    filter_list = [row[mask].tolist() for row, mask in zip(index, mask_mat)]
    matching = [ [i,j] for i,j in zip(np.arange(index.shape[0]), filter_list) ]

    return matching


def run_SLAT_multi(adatas:List[AnnData],
                     order:Optional[list]=None,
                     k_cutoff:Optional[int]=10,
                     feature:Optional[str]='DPCA',
                     cos_cutoff:Optional[float]=0.85,
                     n_jobs:Optional[int]=-1,
                     top_K:Optional[int]=50
    )->List[np.ndarray]:
    r"""
    Run SLAT on multi-dataset for 3D re-construct
    
    Parameters
    -----------
    adatas
        list of adatas
    order
        biological order of the slides
    k_cutoff
        k nearest neighbor
    feature
        feature to use, one of ['DPCA', 'PCA', 'harmony']
    cos_cutoff
        cosine similarity cutoff of mapping results
    n_jobs
        cpu cores to use
    top_K
        top K smooth mapping results
        
    Return
    ----------
    matching_list
        list of precise mapping results
    index_list
        list of top mapping index
    """
    from joblib import Parallel, delayed
    order = range(len(adatas)) if order == None else order
    n_jobs = len(adatas) + 1 if n_jobs < 0 else n_jobs
    # for adata in adatas:
    #     Cal_Spatial_Net(adata, k_cutoff=k_cutoff, model='KNN')
    adatas = Parallel(n_jobs=n_jobs)(delayed(Cal_Spatial_Net)(adata, k_cutoff=k_cutoff, model='KNN',return_data=True) for adata in adatas)
    matching_list = []

    def parall_SLAT(a1, a2, i):
        print(f'Parallel mapping dataset:{i} --- dataset:{i+1}')
        edges, features = load_anndatas([a1, a2], feature=feature, check_order=False)
        embd0, embd1, _ = run_SLAT(features, edges)
        best, index, distance = spatial_match([embd0,embd1], reorder=False, smooth_range=top_K, adatas=[a1,a2])
        return best, index, distance
    
    unfiltered_zip_res = Parallel(n_jobs=n_jobs)(delayed(parall_SLAT)(a1,a2,i)\
        for i, (a1, a2) in enumerate(zip(adatas, adatas[1:])))
    matching_list = []
    for (best, index, distance) in unfiltered_zip_res:
        matching = np.array([range(index.shape[0]), best])
        matching_filter = matching[:, distance[:,0]>cos_cutoff]
        matching_list.append(matching_filter)
    # assert [x.shape[1] for x in matching_list] == [y.shape[0] for y in adatas[1:]] # check order
    return matching_list, unfiltered_zip_res


def calc_k_neighbor(features:List[torch.Tensor],
                    k_list:List[int]
    ) -> Mapping:
    r"""
    cal k nearest neighbor
    
    Parameters:
    ----------
    features
        feature list to find KNN
    k
        list of k to find (must have 2 elements)
    """
    assert len(k_list) == 2
    k_list = sorted(k_list)
    nbr_dict = {}
    for k in k_list:
        nbr_dict[k] = [None, None]

    for i, feature in enumerate(features): # feature loop first
        for k in k_list:     # then k list loop
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(feature)
            distances, indices = nbrs.kneighbors(feature) # indices include it self
            nbr_dict[k][i] = nbrs

    return nbr_dict


def add_noise(adata,
              noise:Optional[str]='nb',
              inverse_noise:Optional[float]=5
    ) -> AnnData:
    r"""
    Add poisson or negative binomial noise on raw counts
    also run scanpy pipeline to PCA step
    
    Parameters
    ----------
    adata
        anndata object
    noise
        type of noise, one of 'poisson' or 'nb'
    inverse_noise
        if noise is 'nb', control the noise level 
        (smaller means larger variance) 
    """
    if 'counts' not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()
    mu = torch.tensor(adata.X.todense())
    if noise.lower() == 'poisson':
        adata.X = torch.distributions.poisson.Poisson(mu).sample().numpy()
    elif noise.lower() == 'nb':
        adata.X = torch.distributions.negative_binomial.NegativeBinomial(inverse_noise,logits=(mu.log()-math.log(inverse_noise))).sample().numpy()
    else:
        raise NotImplementedError('Can not add this type noise')
    return adata.copy()