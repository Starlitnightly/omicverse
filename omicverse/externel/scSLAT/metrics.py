r"""
Align metrics
"""
from typing import List, Optional, Union
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy import sparse
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from anndata import AnnData

from .model.prematch import rotate_via_numpy


def hit_k(features:List[torch.Tensor],
          ground_truth:torch.Tensor,
          k:Optional[int]=10
    ) -> pd.DataFrame:
    r"""
    Calculate first and top k-th hit ratio of matching
    
    Parameters
    ----------
    features
        features of embed
    ground_truth
        in shape of (2, n_node)
    k
        top k
    """
    embd0 = features[0]
    embd1 = features[1]
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())

    cossim = torch.zeros(embd1.size(0), embd0.size(0))
    for i in range(embd1.size(0)):
        cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)

    ind = cossim.argsort(dim=1, descending=True)[:, :k]
    # adata2 = adata.copy()

    a1 = 0
    ak = 0 
    df = pd.DataFrame(columns=['node','target','trans','h1',f'h{k}'])
    for i, node in enumerate(g_list):
        # print(i, node)
        node = int(node)
        df.loc[i] = [-1,-1,-1,None,None]
        df.iloc[i,0] = node
        df.iloc[i,1] = ind[node, 0].item()
        df.iloc[i,2] = g_map[node]
        df.iloc[i,3] = False
        df.iloc[i,4] = False
        if ind[node, 0].item() == g_map[node]:
            a1 += 1
            ak += 1
            df.iloc[i,3] = True
            df.iloc[i,4] = True
        else:
            for j in range(1, ind.shape[1]):
                if ind[node, j].item() == g_map[node]:
                    ak += 1
                    df.iloc[i,4] = True
                    break
    a1 /= len(g_list)
    ak /= len(g_list)
    print('H@1 %.2f%% H@%d %.2f%%' % (a1*100, k, ak*100))
    
    return df


# TODO: modify to Calculate from adata directly
def hit_celltype(source_df:pd.DataFrame, 
                 target_df:pd.DataFrame,
                 matching:np.ndarray, 
                 meta:Optional[str]='celltype'
    ) -> np.ndarray:
    r"""
    Statistics of cell type correct mapping ratio
        
    Parameters
    ----------
    source_df
        source dataset meta dataframe
    target_df
        target dataset meta dataframe
    matching
        cell correspondence array
    meta
        column name in source and target specify the celltype info  
    """
    
    assert all(item in ['index','x','y',meta] for item in source_df.columns.values)
    assert all(item in ['index','x','y',meta] for item in matching.columns.values)
    
    graph_map = {}
    for i in range(matching.shape[1]):
        graph_map[str(matching[0,i])] = str(matching[1,i])
    source_df['target'] = 'unknown'
    source_df['target_celltype'] = 'unknown'
    
    for index in source_df['index']:
        index = str(index)
    target = graph_map[index]
    target_celltype = target_df[target_df['index']==int(target)]['celltype'].astype(str).values
    # print(dataset_B[dataset_B['index']==int(index)]['target'] )
    source_df.loc[source_df['index']==int(index), 'target'] = int(target)
    source_df.loc[source_df['index']==int(index),'target_celltype'] = target_celltype 
    
    result = (source_df['celltype'] == source_df['target_celltype']).value_counts()
    print(result)
    return result


def global_score(adatas: List[AnnData],
                 matching: Union[np.ndarray, List],
                 biology_meta: Optional[str]='',
                 topology_meta: Optional[str]=''
    ) -> float:
    r"""
    Calculate global score, which consider celltype and histology, in two aligned graph (higher means better)
    
    Parameters
    ----------
    adatas
        list of anndata object
    matching
        matching result
    biology_meta
        colname of adata.obs which keeps the biology info such as celltype of cells
    topology_meta
        colname of adata.obs which keeps the topology info such as histology region of cells
        
    Return
    ----------
    global score of mapping, higher means better
    """
    assert len(adatas) == 2
    for adata in adatas:
        assert biology_meta in adata.obs.columns or topology_meta in adata.obs.columns
        if biology_meta not in adata.obs.columns:
            adata.obs[biology_meta] = 'Unknown'
            print(f"Warning! column {biology_meta} not in adata.obs ")
        if topology_meta not in adata.obs.columns:
            adata.obs[topology_meta] = 'Unknown'
            print(f"Warning! column {topology_meta} not in adata.obs ")
        adata.obs['global_meta'] = adata.obs[biology_meta].astype(str) + '-' + adata.obs[topology_meta].astype(str)
    count = 0
    if isinstance(matching, list):
        print('Using probabilistic matching')
        for i in range(len(matching)): # query dataset
            query_meta = adatas[1].obs.iloc[i].loc['global_meta']
            ref_meta = adatas[0].obs.iloc[matching[i][1]].loc[:,'global_meta'].to_list()
            if len(ref_meta) == 0:
                continue
            # find the most frequent meta in ref_meta
            vote = max(ref_meta, key = ref_meta.count)
            count = count + 1 if query_meta == vote else count
    elif isinstance(matching, np.ndarray):
        for i in range(matching.shape[0]): # query dataset
            query_meta = adatas[1].obs.iloc[i].loc['global_meta']
            ref_meta = adatas[0].obs.iloc[matching[i,1]].loc['global_meta']
            count = count + 1 if query_meta == ref_meta else count
    else:
        raise ValueError('matching should be list or np.ndarray')
    score = count / adatas[1].shape[0]
    del adatas[0].obs['global_meta']
    
    return score


def edge_score(edges:List[torch.Tensor],
               matching:torch.Tensor,
               score:Optional[List[float]]=[1,-1],
               punish_distance:Optional[bool]=False,
               punish_scale:Optional[float]=1,
    ) -> float:
    r"""
    Calculate edge score in two aligned graph (higher means better)
    
    Parameters
    ----------
    edges
        list of edge of every dataset 
    matching
        matching result
    score
        score of [match_edge, mismatch_edge]
    punish_distance
        if punish on the distance of mismatch cell
    punish_scale
        punish scale if `punish_distance` is `True`
    Reference
    ----------
    Joel Douglas et al. "Metrics for Evaluating Network Alignment"
    """
    a0 = pyg.utils.to_scipy_sparse_matrix(edges[0]).tocsc()
    a1 = pyg.utils.to_scipy_sparse_matrix(edges[1]).tocsc()

    a0_reindex = a0[matching[1,:],:][:,matching[1,:]]
    res = a0_reindex + a1 - 2*sparse.eye(a0_reindex.shape[0])
    res.data[np.where(res.data==2)] = score[0]   # matched edge
    res.data[np.where(res.data==1)] = score[1]   # mismatch edge
    score = res.sum()/res.shape[0]
    return float(score)


def euclidean_dis(adata1:AnnData,
                  adata2:AnnData,
                  matching:np.ndarray,
                  spatial_key:Optional[str]='spatial'
    ) -> float:
    r"""
    Calculate euclidean distance between two datasets with ground truth (lower means better)
    
    Parameters
    ----------
    adata1
        adata1 with spatial
    adata2
        adata2 with spatial
    matching
        matching result
    spatial_key
        key of spatial data in adata.obsm
    """
    # reindex adata1 and adata2 by matching then calculate the pairwise euclidean distance
    for adata in [adata1, adata2]:
        coord = adata.obsm[spatial_key]
        if abs(coord.ptp()) > 1 or abs(coord.max()) > 1:
            adata.obsm['scale_spatial'] = (coord - coord.min(0))/coord.ptp(0)
        else:
            adata.obsm['scale_spatial'] = coord
    coord1 = adata1.obsm['scale_spatial'][matching[1,:]]
    coord2 = adata2.obsm['scale_spatial']
    distance = np.sqrt((coord1[:,0] - coord2[:,0])**2+(coord1[:,1] - coord2[:,1])**2)
    return float(distance.sum()/distance.shape[0])


# TODO: optimize the speed
def calc_NMI():
    r"""
    (Abandon) NMI: neighbor matching index, measure how many K nearest spatial neighbors has matching
    from 0 to 1
    
    Parameters
    ----------
    adata
        adata with spatial
    matching
        matching result, must one to one now
    k
        k nearest neighbor
    spatial_key
        key of spatial data in adata.obsm 
    
    Note
    ----------
    This function runs very slow in current version, need optimize it 
    """
    pass
    

def __interval_statistics(data, intervals:int)->None:
    r"""
    Print the distribution of the value in intervals
    
    Parameters:
    -----------
    data
        array or list
    intervals
        number of intervals
    """
    if len(data) == 0:
        return
    for num in data:
        for interval in intervals:
            lr = tuple(interval.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals[interval] += 1
    for key, value in intervals.items():
        print("%10s" % key, end='')   
        print("%10s" % value, end='')  
        print('%16s' % '{:.3%}'.format(value * 1.0 / len(data)))


def region_statistics(input,
                      step:Optional[float]=0.05,
                      start:Optional[float]=0.5,
                      number_of_interval:Optional[int]=10
    ) -> None:
    r"""
    Print the region statistic results
    
    Parameters
    -----------
    input
        list, 1D np.array or 1D torch.Tensor
    step 
        stride size
    start 
        start point
    number_of_interval
        number of interval

    """
    intervals = {'{:.3f}~{:.3f}'.format(step *x+start, step *(x+1)+start): 0 for x in range(number_of_interval)} 
    __interval_statistics(input, intervals)


def rotation_angle(X, Y, pi, ground_truth:float = 0,
                    output_angle:bool = True, output_matrix:bool = False
    ) -> Union[float, Optional[np.ndarray]]:
    r"""
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).

    Parameters:
    ----------
    X
        np array of spatial coordinates (ex: sliceA.obs['spatial'])
    Y
        np array of spatial coordinates (ex: sliceB.obs['spatial'])
    pi
        mapping between the two layers output in PASTE format (N x M matching matrix)
    ground_truth
        If known, the ground truth rotation angle to use for calculating error.
    output_angle
        Boolean of whether to return rotation angle.
    output_matrix
        Boolean of whether to return the rotation as a matrix or an angle.

    Returns
    ----------
    Aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y.
        
    Reference
    ----------
    Modify from https://github.com/raphael-group/paste/blob/a9b10b24ba33e94a89dd89e8ee5e4900e18b1886/src/paste/visualization.py#L157
    """
    assert X.shape[1] == 2 and Y.shape[1] == 2
    
    rad = np.deg2rad(ground_truth)
    X = rotate_via_numpy(X, rad)

    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    M = np.array([[0,-1],[1,0]])
    theta = np.arctan2(np.trace(M.dot(H)), np.trace(H))
    theta = -np.degrees(theta)
    delta = np.absolute(theta - ground_truth)
    if output_angle and not output_matrix:
        return delta
    elif output_angle and output_angle:
        return delta, X, Y, R, tX, tY
    else:
        return X, Y