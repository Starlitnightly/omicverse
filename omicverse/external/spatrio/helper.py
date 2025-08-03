import numpy as np
import scanpy as sc
import pandas as pd
import scipy

import math
from typing import List
from anndata import AnnData
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Read in 
def load_data(path):
    """
    Read from the specified path, such as data in AnnData format.
    
    Args:
        path: Path to input data.
    
    Returns:
        -Input dataset in AnnData format.
    """
    X = pd.read_csv(path, index_col=0).T
    adata = sc.AnnData(X = X)
    return adata

# Process input
def process_input(adata1,adata2,marker_use=True,top_marker_num=100,marker1_by="type",marker2_by="type",min_cells = 3,hvg_use=False):
    """
    Process input datasets.
    
    Args:
        adata1: Spatial transcriptomic data.
        adata2: Single cell multi-omics data.
        marker_use:  Determines whether to select differential genes of each cell/spot type for subsequent analysis.
        top_marker_num: Number of differential genes in each cell/spot type.
        marker1_by: Which obseravation in adata1 is used as the basis for differential gene calculation.
        marker2_by: Which obseravation in adata2 is used as the basis for differential gene calculation.
        min_cells: Parameters for filtering genes. If the expressed cells are lower than this value, the gene will be filtered.
        hvg_use: Determines whether to only reserve highly variable genes.

    Returns:
        -Processed datasets in AnnData format.
    """
    
    ## Select common genes
    if min_cells > 0:
        sc.pp.filter_genes(adata1,min_cells=min_cells)
        sc.pp.filter_genes(adata2,min_cells=min_cells)
    common_genes = intersect(adata1.var.index, adata2.var.index)
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]

    ## Select deg
    if marker_use:
        print("Using marker genes, top_marker_num = "+str(top_marker_num))
        marker1 = find_marker(adata1,top_marker_num,maker_by=marker1_by)
        marker2 = find_marker(adata2,top_marker_num,maker_by=marker2_by)
        marker1.extend(marker2)
        marker = np.unique(marker1).tolist()
        adata1 = adata1[:, marker]
        adata2 = adata2[:, marker]
        print("Spot data has "+str(adata1.shape[0])+" spots and "+str(adata1.shape[1])+" features.")
        print("Single-cell data has "+str(adata2.shape[0])+" cells and "+str(adata2.shape[1])+" features.")
    if hvg_use:
        print("Using hvg genes")
        marker1 = find_hvg(adata1)
        marker2 = find_hvg(adata2)
        marker = list(set(marker1).intersection(set(marker1)))
        adata1 = adata1[:, marker]
        adata2 = adata2[:, marker]
        print("Spot data has "+str(adata1.shape[0])+" spots and "+str(adata1.shape[1])+" features.")
        print("Single-cell data has "+str(adata2.shape[0])+" cells and "+str(adata2.shape[1])+" features.")
        
    return adata1, adata2

# Process anndata
def process_anndata(adata,ndims=50,scale=True,pca=True):
    """
    Process anndata.
    
    Args:
        adata: Data in AnnData format.
        ndims: Number of pca dimensions selected when processing input data.
        scale: Determines whether to scale the data.
        pca: Determines whether to perform principal components analysis.
        
    Returns:
        -Processed dataset in AnnData format. 
    """

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata) 
    if pca:
        sc.tl.pca(adata,n_comps=ndims)
    return adata

# DEG calculation
def find_marker(adata,top_marker_num,maker_by='type'):
    """
    Calculate differential genes.
    
    Args:
        adata: Data in AnnData format.
        top_marker_num: Number of differential genes in each cell/spot type
        maker_by: Which obseravation in adata is used as the basis for differential gene calculation.

    Returns:
        -The list of differential genes. 
    """
    adata_copy = adata.copy()
    adata_copy = process_anndata(adata_copy,ndims=50,scale=False,pca=False)
    sc.tl.rank_genes_groups(adata_copy, maker_by, method='wilcoxon')
    marker_df = pd.DataFrame(adata_copy.uns['rank_genes_groups']['names']).head(top_marker_num)
    marker_array = np.array(marker_df)
    marker_array = np.ravel(marker_array)
    marker_array = np.unique(marker_array)
    marker = list(marker_array)
    return marker

# Highly variable genes calculation
def find_hvg(adata):
    """
    Calculate highly variable genes.
    
    Args:
        adata: Data in AnnData format.
        
    Returns:
        -The list of highly variable genes. 
    """
    
    adata_copy = adata.copy()
    adata_copy = process_anndata(adata_copy,ndims=50,scale=False,pca=False)
    sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
    marker = list(adata_copy.var.index[adata_copy.var['highly_variable']])
    return marker

# Graph construction with KNN
def construct_graph(X, k, mode= "connectivity", metric="minkowski",p=2):
    """
    Construct graph with KNN.
    
    Args:
        X: 
        k: Number of neighbors to be used when constructing kNN graphs.
        mode: "connectivity" or "distance". Determines whether to use a connectivity graph or a distance graph.Default="connectivity".
        metric: Sets the metric to use while constructing nearest neighbor graphs. The default distance is 'euclidean' ('minkowski' metric with the pparam equal to 2.)
        p: Power parameter for the Minkowski metric.
        
    Returns:
        -The knn graph of input data. 
    """
    
    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    c_graph=kneighbors_graph(X, k, mode=mode, metric=metric, include_self=include_self,p=p)
    return c_graph

# Distance calculation
def distances_cal(graph, type_aware=None, aware_power=2):
    """
    Calculate distance between cells/spots based on graph.
    
    Args:
        graph: KNN graph.
        type_aware: A dataframe contains cells/spots id and type information.
        aware_power: Type aware parameter. The greater the parameter, the greater the distance between different areas/types of spots/cells in the graph.
        
    Returns:
        -The distance matrix of cells/spots. 
    """
    from tqdm.auto import tqdm  # 自动选择适合环境的进度条
    
    # 计算最短路径时添加进度条
    with tqdm(total=1, desc='Calculating shortest paths') as pbar:
        shortestPath = dijkstra(csgraph=csr_matrix(graph), directed=False, return_predecessors=False)
        pbar.update(1)

    if type_aware is not None:
        # 类型感知处理步骤添加进度条
        with tqdm(total=4, desc='Applying type adjustments') as pbar:
            shortestPath = to_dense_array(shortestPath)
            pbar.update(1)
            
            # 获取类型信息并创建掩码矩阵
            type_ids = type_aware.iloc[:, 1].values
            type_mask = (type_ids[:, None] != type_ids[None, :])
            pbar.update(1)
            
            # 应用类型感知调整
            shortestPath = shortestPath * np.where(type_mask, aware_power, 1)
            pbar.update(1)
            
            # 保持矩阵对称性
            shortestPath = np.maximum(shortestPath, shortestPath.T)
            pbar.update(1)

    the_max = np.nanmax(shortestPath[shortestPath != np.inf])
    shortestPath[shortestPath > the_max] = the_max
    C_dis = shortestPath / shortestPath.max()
    C_dis -= np.mean(C_dis)
    return C_dis

# KL divergence
def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence of two matrices X and Y.
    
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    
    Returns:
        Pairwise KL divergence matrix.
    """
    
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

# KL divergence with POT backend
def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence of two matrices X and Y with POT backend to speed up.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    
    Returns:
        Pairwise KL divergence matrix.
    """
    import ot
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X,Y)
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)

# Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

# Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep] 



def top_n(df, n=3, column='APM'):
    """
    Get a subset of the DataFrame according to the values of a column.
    
    """
    return df.sort_values(by=column, ascending=False)[:n]

def dist_cal(x1,y1,x2,y2):
    dist_x = x2 - x1
    dist_y = y2 - y1
    square_all = math.sqrt(dist_x*dist_x + dist_y*dist_y)
    return square_all

def extract_exp(data):
    """
    Extract gene expression dataframe from AnnData.

    Args:
        data: AnnData
    
    Returns:
        exp_data: DataFrame of gene expression.
    """
    
    ##exp_data = pd.DataFrame(data.X)
    #exp_data.columns = data.var.index.tolist()
    #exp_data.index = data.obs.index.tolist()
    return data.to_df()

def scale_num(list):
    """
    Scale the input list.

    Args:
        list: List
    
    Returns:
        scale_list: List of scaled elements.
    """
    
    a = max(list)
    b = min(list)
    scale_list = []
    for i in list:
        scale_num = (i-b)/(a-b)
        scale_list.append(scale_num)
    return scale_list

def intersect(lst1, lst2): 
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List
    
    Returns:
        lst3: List of common elements.
    """
    
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3