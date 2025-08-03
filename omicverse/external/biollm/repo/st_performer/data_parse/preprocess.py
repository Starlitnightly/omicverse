import scanpy as sc
import numpy as np
import os
import pickle
import lmdb
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy
import json
from anndata import AnnData
from typing import Optional, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from scipy.sparse import issparse


#############
# data2lmdb
#############
def data2lmdb(dpath,type:Literal['sc', 'st'],write_frequency=10000,n_neighbors=10,):
    print("Generate LMDB to %s" % dpath)
    train_db = lmdb.open(dpath + 'train.db', map_size=int(5000e9), readonly=False, meminit=False, map_async=True)
    txn = train_db.begin(write=True)
    length = 1
    data_id = 1
    for f in os.listdir(dpath):
        if f.endswith('.h5ad'):
            adata = sc.read_h5ad(os.path.join(dpath, f))
            data = adata.X
            meta = {'m'+str(data_id): {'organ': 'brain', 'gene_list': adata.var['gene_id'].tolist()}}
            if type == 'sc':
                for i in range(data.shape[0]):
                    x = data[i].A.tolist()
                    res = {'x': x, 'meta': 'm'+str(data_id) }
                    value = res
                    txn.put(str(length).encode(), dict_to_bytes(value))
                    length += 1
                    if (length + 1) % write_frequency == 0:
                        print('write: ', length)
                        txn.commit()
                        txn = train_db.begin(write=True)
                txn.put(('m'+str(data_id)).encode(), dict_to_bytes(meta['m'+str(data_id)]))
                data_id += 1
                print(data_id)
            if type == 'st':
                for obs_id in range(data.shape[0]):
                    ## neigh
                    neighbor_indices = adata.uns['spatial_neighbors']['indices'][obs_id].tolist()[1:]
                    exp_pairs = [[data[obs_id].A[0].tolist(), data[k].A[0].tolist()] for k in neighbor_indices]
                    for i in range(len(neighbor_indices)):
                        res = {'x': exp_pairs[i], 'meta': 'm'+str(data_id),'labels':1}
                        value = res
                        txn.put(str(length).encode(), dict_to_bytes(value))
                        length += 1
                    ## non_neigh
                    non_neighbor_indices = adata.uns['spatial_non_neighbors']['indices'][obs_id].tolist()
                    non_exp_pairs = [[data[obs_id].A[0].tolist(), data[k].A[0].tolist()] for k in non_neighbor_indices]
                    for j in range(len(non_neighbor_indices)):
                        res = {'x': exp_pairs[i], 'meta': 'm'+str(data_id),'labels':0}
                        value = res
                        txn.put(str(length).encode(), dict_to_bytes(value))
                        length += 1
                    if (length + 1) % write_frequency == 0:
                        print('write: ', length)
                        txn.commit()
                        txn = train_db.begin(write=True)
                txn.put(('m'+str(data_id)).encode(), dict_to_bytes(meta['m'+str(data_id)]))
                data_id += 1
                print(data_id)
            print(length)
            del adata
    # finish iterating through dataset
    txn.commit()
    with train_db.begin(write=True) as txn:
        txn.put(b'__len__', str(length).encode())
    print("Flushing database ...")
    train_db.sync()
    train_db.close()

def dict_to_bytes(d):
    return json.dumps(d).encode()

##################
#data preprocess
##################
def data_process(file_path,save_path,gene_vocab_path,suffix: Literal['.h5ad', '.gef', '.gem', '.tif'],type:Literal['sc', 'st']):
    files = os.listdir(file_path)
    h5ad_files = [file for file in files if file.endswith(suffix)]
    adata_list = []
    for file_name in h5ad_files:
        h5ad_path = os.path.join(file_path, file_name)  
        ## load data
        adata = sc.read_h5ad(h5ad_path)
        ## gene_vocab,'gene_vocsb scdata as init' 
        gene_dict = load_gene_dict(gene_vocab_path)
        gene_list = adata.var['Symbol'].tolist()
        gene_indices = search_gene_indices(gene_dict, gene_list)
        save_gene_dict(gene_vocab_path, gene_dict)
        adata.var['gene_id'] = gene_indices
        ## sc_data
        if type == 'sc':
            deal_exp_matrix(adata,layer=None,normal_method='log1p',inplace=True)
        ## st_data    
        elif type == 'st':
            ### 1. pp obsm[spatial_llm]
            spatial = 'spatial_llm'
            if 'spatial_stereoseq' in adata.obsm.keys():
                adata.obsm[spatial] = np.zeros((adata.shape[0], 2))
                adata.obsm[spatial][:, 0] = adata.obsm['spatial_stereoseq']['X']
                adata.obsm[spatial][:, 1] = adata.obsm['spatial_stereoseq']['Y']
            elif 'spatial_visium' in adata.obsm.keys():
                adata.obsm[spatial] = np.zeros((adata.shape[0], 2))
                adata.obsm[spatial][:, 0] = adata.obsm['spatial_visium']['array_row']
                adata.obsm[spatial][:, 1] = adata.obsm['spatial_visium']['array_col']
            elif 'spatial' in adata.obsm.keys():
                adata.obsm[spatial] = np.zeros((adata.shape[0], 2))
                adata.obsm[spatial][:, 0] = adata.obsm['spatial']['X']
                adata.obsm[spatial][:, 1] = adata.obsm['spatial']['Y']
            ### 2.calculate neighbors
            adata = neighbors(adata,basis='spatial',spatial_key=spatial,n_neighbors=10)
            deal_exp_matrix(adata,layer=None,normal_method='log1p',inplace=True)
        print("process adata %s" % file_name)
        adata.write(os.path.join(save_path, file_name))
        adata_list.append(adata)
        del adata
    return adata_list

def deal_exp_matrix(adata, 
                    layer='X',
                    normal_method=Literal['pearson_residuals','log1p'],
                    inplace=True):
    """
    Normalize expression matrix.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    X=X.A if issparse(X) else X
    if np.max(X[0:10, :] - np.int32(X[0:10, :])) == np.int32(0): 
        if normal_method=='pearson_residuals':
            _pearson_residuals(adata,layer=layer,theta=100,clip=None,inplace=inplace)
        elif normal_method=='log1p':
            sc.pp.normalize_total(adata,layers=[layer],target_sum=1e4, inplace=inplace)
            sc.pp.log1p(adata,base=2)
    return adata

def _pearson_residuals(adata,
                      layer=None, 
                      theta: float = 100, 
                      clip: Optional[float] = None,
                      inplace=True):
    """
    Perform pearson residuals on expression matrix.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    X=X.A if issparse(X) else X
    computed_on = layer if layer else 'adata.X'
    
    log = f'computing analytic Pearson residuals on {computed_on}'
    print(log)
    
    # check theta
    if theta <= 0:
        # TODO: would "underdispersion" with negative theta make sense?
        # then only theta=0 were undefined..
        raise ValueError('Pearson residuals require theta > 0')
    
    # prepare clipping
    if clip is None:
        n = X.shape[0]
        clip = np.sqrt(n)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    
    sums_genes = np.sum(X, axis=0, keepdims=True)
    sums_cells = np.sum(X, axis=1, keepdims=True)
    sum_total = np.sum(sums_genes)

    mu = np.array(sums_cells @ sums_genes / sum_total).astype(np.float32)
    del sum_total, sums_genes, sums_cells
    diff = np.array(X - mu)
    del X
    residuals = diff / np.sqrt(mu + mu ** 2 / theta)
    del diff
    residuals = np.clip(residuals, a_min=-clip, a_max=clip)
      
    # save result adata
    settings_dict = dict(theta=theta, clip=clip, computed_on=computed_on)
    
    if inplace:
        adata.uns['pearson_residuals_normalization'] = settings_dict
        adata.X = residuals
        print("finished!")
    else:
        return residuals

### gene_vocab
def load_gene_dict(file_path):
    with open(file_path, "r") as f:
        gene_dict = json.load(f)
    return gene_dict

def save_gene_dict(file_path, gene_dict):
    with open(file_path, "w") as f:
        json.dump(gene_dict, f)

def search_gene_indices(gene_dict, gene_list):
    indices = []
    for gene_name in gene_list:
        found = False
        for index, name in gene_dict.items():
            if name == gene_name:
                indices.append(int(index))
                found = True
                break
        if not found:
            max_index = max(map(int, gene_dict.keys())) if gene_dict else 0
            max_index += 1
            gene_dict[str(max_index)] = gene_name
            indices.append(int(max_index))
    return indices

## calculate neighbors
def neighbors(
    adata: AnnData,
    nbr_object: NearestNeighbors = None,
    basis: str = "pca",
    spatial_key: str = "spatial",
    n_neighbors_method: str = "ball_tree",
    n_pca_components: int = 30,
    n_neighbors: int = 10,
) -> AnnData:
    """Given an AnnData object, compute pairwise connectivity matrix in transcriptomic space

    Args:
        adata : an anndata object.
        nbr_object: An optional sklearn.neighbors.NearestNeighbors object. Can optionally create a nearest neighbor
            object with custom functionality.
        basis: str, default 'pca'
            The space that will be used for nearest neighbor search. Valid names includes, for example, `pca`, `umap`,
            or `X` for gene expression neighbors, 'spatial' for neighbors in the physical space.
        spatial_key: Optional, can be used to specify .obsm entry in adata that contains spatial coordinates. Only
            used if basis is 'spatial'.
        n_neighbors_method: str, default 'ball_tree'
            Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree".
        n_pca_components: Only used if 'basis' is 'pca'. Sets number of principal components to compute (if PCA has
            not already been computed for this dataset).
        n_neighbors: Number of neighbors for kneighbors queries.

    Returns:
        adata : Modified AnnData object, .uns[spatial_neighbors],.obsp['spatial_connectivities'],.obsp['spatial_distances']
    """

    if basis == "pca" and "X_pca" not in adata.obsm_keys():
        print(
            "PCA to be used as basis for :func `transcriptomic_connectivity`, X_pca not found, " "computing PCA...",
            indent_level=2,
        )
        pca = PCA(
            n_components=min(n_pca_components, adata.X.shape[1] - 1),
            svd_solver="arpack",
            random_state=0,
        )
        fit = pca.fit(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else pca.fit(adata.X)
        X_pca = fit.transform(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else fit.transform(adata.X)
        adata.obsm["X_pca"] = X_pca

    if basis == "X":
        X_data = adata.X
    elif basis == "spatial":
        X_data = adata.obsm[spatial_key]
    elif "X_" + basis in adata.obsm_keys():
        # Assume basis can be found in .obs under "X_{basis}":
        X_data = adata.obsm["X_" + basis]
    else:
        raise ValueError("Invalid option given to 'basis'. Options: 'pca', 'umap', 'spatial' or 'X'.")

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm=n_neighbors_method, n_neighbors=n_neighbors, metric="euclidean").fit(X_data)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    # Update AnnData to add spatial distances, spatial connectivities and spatial neighbors from the sklearn
    # NearestNeighbors run:
    distances, knn = nbrs.kneighbors(X_data)
    distances, connectivities = compute_distances_and_connectivities(knn, distances)

    ## random sample pair non_neighbors in adata
    non_neighbor = np.zeros((knn.shape[0], n_neighbors-1))
    for i in range(knn.shape[0]):
        row = knn[i]
        random_indices = np.random.choice(np.setdiff1d(range(adata.n_obs), row), size=n_neighbors-1, replace=False)
        non_neighbor[i] = random_indices
    
    ## insert neighbors and non-neighbors in adata
    if basis != "spatial":
        adata.obsp["exp_distances"] = distances
        adata.obsp["exp_connectivities"] = connectivities
        adata.uns["exp_neighbors"] = {}
        adata.uns["exp_non_neighbors"] = {}
        adata.uns["exp_neighbors"]["indices"] = knn
        adata.uns["exp_neighbors"]["params"] = {"n_neighbors": n_neighbors, "method": n_neighbors_method, "metric": "euclidean"}
        adata.uns["exp_non_neighbors"]['indices'] = non_neighbor.astype(int)
    else:
        adata.obsp["spatial_distances"] = distances
        adata.obsp["spatial_connectivities"] = connectivities
        adata.uns["spatial_neighbors"] = {}
        adata.uns["spatial_non_neighbors"] = {}
        adata.uns["spatial_neighbors"]["indices"] = knn
        adata.uns["spatial_neighbors"]["params"] = {"n_neighbors": n_neighbors, "method": n_neighbors_method, "metric": "euclidean"}
        adata.uns["spatial_non_neighbors"]['indices'] = non_neighbor.astype(int)
    return adata

def compute_distances_and_connectivities(
    knn_indices: np.ndarray, distances: np.ndarray
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """Computes connectivity and sparse distance matrices

    Args:
        knn_indices: Array of shape (n_samples, n_samples) containing the indices of the nearest neighbors for each
            sample.
        distances: The distances to the n_neighbors the closest points in knn graph

    Returns:
        distances: Sparse distance matrix
        connectivities: Sparse connectivity matrix
    """
    n_obs, n_neighbors = knn_indices.shape
    distances = scipy.sparse.csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn_indices.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    return distances, connectivities
