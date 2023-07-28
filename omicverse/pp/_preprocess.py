"""
Copy from pegasus and cellual

"""


import anndata
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Sequence, List, Dict
import skmisc.loess as sl
import scanpy as sc

from scipy.sparse import issparse, csr_matrix
from ..utils import load_signatures_from_file,predefined_signatures

def identify_robust_genes(data: anndata.AnnData, percent_cells: float = 0.05) -> None:
    """ 
    Identify robust genes as candidates for HVG selection and remove genes that are not expressed in any cells.

    Arguments:
        data: Use current selected modality in data, which should contain one RNA expression matrix.
        percent_cells: Only assign genes to be ``robust`` that are expressed in at least ``percent_cells`` % of cells.


    Update ``data.var``:

        * ``n_cells``: Total number of cells in which each gene is measured.
        * ``percent_cells``: Percent of cells in which each gene is measured.
        * ``robust``: Boolean type indicating if a gene is robust based on the QC metrics.
        * ``highly_variable_features``: Boolean type indicating if a gene is a highly variable feature. By default, set all robust genes as highly variable features.

    """

    prior_n = data.shape[1]

    if issparse(data.X):
        data.var["n_cells"] = data.X.getnnz(axis=0)
        data._inplace_subset_var(data.var["n_cells"] > 0)
        data.var["percent_cells"] = (data.var["n_cells"] / data.shape[0]) * 100
        data.var["robust"] = data.var["percent_cells"] >= percent_cells
    else:
        data.var["robust"] = True

    data.var["highly_variable_features"] = data.var["robust"]  # default all robust genes are "highly" variable
    print(f"After filtration, {data.shape[1]}/{prior_n} genes are kept. Among {data.shape[1]} genes, {data.var['robust'].sum()} genes are robust.")

def calc_mean_and_var(X: Union[csr_matrix, np.ndarray], axis: int) -> Tuple[np.ndarray, np.ndarray]:
    if issparse(X):
        from ..cylib.fast_utils import calc_mean_and_var_sparse
        return calc_mean_and_var_sparse(X.shape[0], X.shape[1], X.data, X.indices, X.indptr, axis)
    else:
        from ..cylib.fast_utils import calc_mean_and_var_dense
        return calc_mean_and_var_dense(X.shape[0], X.shape[1], X, axis)
    
def calc_stat_per_batch(X: Union[csr_matrix, np.ndarray], batch: Union[pd.Categorical, np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pandas.api.types import is_categorical_dtype
    if is_categorical_dtype(batch):
        nbatch = batch.categories.size
        codes = batch.codes.astype(np.int32)
    else:
        codes = np.array(batch, dtype = np.int32)
        nbatch = codes.max() + 1 # assume cluster label starts from 0

    if issparse(X):
        from ..cylib.fast_utils import calc_stat_per_batch_sparse
        return calc_stat_per_batch_sparse(X.shape[0], X.shape[1], X.data, X.indices, X.indptr, nbatch, codes)
    else:
        from ..cylib.fast_utils import calc_stat_per_batch_dense
        return calc_stat_per_batch_dense(X.shape[0], X.shape[1], X, nbatch, codes)

def estimate_feature_statistics(data: anndata.AnnData, batch: str) -> None:
    """ Estimate feature (gene) statistics per channel, such as mean, var etc.
    """
    if batch is None:
        data.var["mean"], data.var["var"] = calc_mean_and_var(data.X, axis=0)
    else:
        ncells, means, partial_sum = calc_stat_per_batch(data.X, data.obs[batch].values)
        partial_sum[partial_sum < 1e-6] = 0.0

        data.uns["ncells"] = ncells
        data.varm["means"] = means
        data.varm["partial_sum"] = partial_sum

        data.var["mean"] = np.dot(means, ncells) / data.shape[0]
        data.var["var"] = partial_sum.sum(axis=1) / (data.shape[0] - 1.0)



def select_hvf_pegasus(
    data: anndata.AnnData, batch: str, n_top: int = 2000, span: float = 0.02
) -> None:
    """ Select highly variable features using the pegasus method
    """
    if "robust" not in data.var:
        raise ValueError("Please run `identify_robust_genes` to identify robust genes")

    estimate_feature_statistics(data, batch)

    robust_idx = data.var["robust"].values
    hvf_index = np.zeros(robust_idx.sum(), dtype=bool)

    mean = data.var.loc[robust_idx, "mean"]
    var = data.var.loc[robust_idx, "var"]

    span_value = span
    while True:
        lobj = fit_loess(mean, var, span = span_value, degree = 2)
        if lobj is not None:
            break
        span_value += 0.01
    if span_value > span:
        print("Leoss span is adjusted from {:.2f} to {:.2f} to avoid fitting errors.".format(span, span_value))

    rank1 = np.zeros(hvf_index.size, dtype=int)
    rank2 = np.zeros(hvf_index.size, dtype=int)

    delta = var - lobj.outputs.fitted_values
    fc = var / lobj.outputs.fitted_values

    rank1[np.argsort(delta)[::-1]] = range(hvf_index.size)
    rank2[np.argsort(fc)[::-1]] = range(hvf_index.size)
    hvf_rank = rank1 + rank2

    hvf_index[np.argsort(hvf_rank)[:n_top]] = True

    data.var["hvf_loess"] = 0.0
    data.var.loc[robust_idx, "hvf_loess"] = lobj.outputs.fitted_values

    data.var["hvf_rank"] = -1
    data.var.loc[robust_idx, "hvf_rank"] = hvf_rank
    data.var["highly_variable_features"] = False
    data.var.loc[robust_idx, "highly_variable_features"] = hvf_index

def calc_expm1(X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
    if not issparse(X):
        return np.expm1(X)
    res = X.copy()
    np.expm1(res.data, out = res.data)
    return res

def select_hvf_seurat_single(
    X: Union[csr_matrix, np.ndarray],
    n_top: int,
    min_disp: float,
    max_disp: float,
    min_mean: float,
    max_mean: float,
) -> List[int]:
    """ HVF selection for one channel using Seurat method
    """
    X = calc_expm1(X)

    mean, var = calc_mean_and_var(X, axis=0)

    dispersion = np.full(X.shape[1], np.nan)
    idx_valid = (mean > 0.0) & (var > 0.0)
    dispersion[idx_valid] = var[idx_valid] / mean[idx_valid]

    mean = np.log1p(mean)
    dispersion = np.log(dispersion)

    df = pd.DataFrame({"log_dispersion": dispersion, "bin": pd.cut(mean, bins=20)})
    log_disp_groups = df.groupby("bin")["log_dispersion"]
    log_disp_mean = log_disp_groups.mean()
    log_disp_std = log_disp_groups.std(ddof=1)
    log_disp_zscore = (
        df["log_dispersion"].values - log_disp_mean.loc[df["bin"]].values
    ) / log_disp_std.loc[df["bin"]].values
    log_disp_zscore[np.isnan(log_disp_zscore)] = 0.0

    hvf_rank = np.full(X.shape[1], -1, dtype=int)
    ords = np.argsort(log_disp_zscore)[::-1]

    if n_top is None:
        hvf_rank[ords] = range(X.shape[1])
        idx = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                log_disp_zscore > min_disp,
                log_disp_zscore < max_disp,
            )
        )
        hvf_rank[~idx] = -1
    else:
        hvf_rank[ords[:n_top]] = range(n_top)

    return hvf_rank



def select_hvf_seurat(
    data: anndata.AnnData,
    batch: str,
    n_top: int,
    min_disp: float,
    max_disp: float,
    min_mean: float,
    max_mean: float,
    n_jobs: int,
) -> None:
    """ Select highly variable features using Seurat method.
    """

    robust_idx = data.var["robust"].values
    X = data.X[:, robust_idx]

    hvf_rank = (
        select_hvf_seurat_single(
            X,
            n_top=n_top,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
        )
        
    )

    hvf_index = hvf_rank >= 0

    data.var["hvf_rank"] = -1
    data.var.loc[robust_idx, "hvf_rank"] = hvf_rank
    data.var["highly_variable_features"] = False
    data.var.loc[robust_idx, "highly_variable_features"] = hvf_index

def highly_variable_features(
    data: anndata.AnnData,
    batch: str = None,
    flavor: str = "pegasus",
    n_top: int = 2000,
    span: float = 0.02,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    min_mean: float = 0.0125,
    max_mean: float = 7,
    n_jobs: int = -1,
) -> None:
    """ Highly variable features (HVF) selection. The input data should be logarithmized.

    Arguments:
        data: Annotated data matrix with rows for cells and columns for genes.
        batch: A key in data.obs specifying batch information. If `batch` is not set, do not consider batch effects in selecting highly variable features. Otherwise, if `data.obs[batch]` is not categorical, `data.obs[batch]` will be automatically converted into categorical before highly variable feature selection.
        flavor: The HVF selection method to use. Available choices are ``"pegasus"`` or ``"Seurat"``.
        n_top: Number of genes to be selected as HVF. if ``None``, no gene will be selected.
        span: Only applicable when ``flavor`` is ``"pegasus"``. The smoothing factor used by *scikit-learn loess* model in pegasus HVF selection method.
        min_disp: Minimum normalized dispersion.
        max_disp: Maximum normalized dispersion. Set it to ``np.inf`` for infinity bound.
        min_mean: Minimum mean.
        max_mean: Maximum mean.
        n_jobs: Number of threads to be used during calculation. If ``-1``, all physical CPU cores will be used.


    Update ``adata.var``:
        * ``highly_variable_features``: replace with Boolean type array indicating the selected highly variable features.

    Examples
    --------
    >>> ov.pp.highly_variable_features(data)
    >>> ov.pp.highly_variable_features(data, batch="Channel")
    """

    if flavor == "pegasus":
        select_hvf_pegasus(data, batch, n_top=n_top, span=span)
    else:
        assert flavor == "Seurat"
        select_hvf_seurat(
            data,
            batch,
            n_top=n_top,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
            n_jobs=n_jobs,
        )

    data.uns.pop("_tmp_fmat_highly_variable_features", None) # Pop up cached feature matrix

    print(f"{data.var['highly_variable_features'].sum()} highly variable features have been selected.")

def fit_loess(x: List[float], y: List[float], span: float, degree: int) -> object:
    try:
        lobj = sl.loess(x, y, span=span, degree=degree)
        lobj.fit()
        return lobj
    except ValueError:
        return None
    
def corr2_coeff(A, B):
    """
    Calculate Pearson correlation between matrix A and B
    A and B are allowed to have different shapes. Taken from Cospar, Wang et al., 2023.
    """
    resol = 10 ** (-15)

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    corr = np.dot(A_mA, B_mB.T) / (np.sqrt(np.dot(ssA[:, None], ssB[None])) + resol)

    return corr

def remove_cc_genes(adata:anndata.AnnData, organism:str='human', corr_threshold:float=0.1):
    """
    Update adata.var['highly_variable_features'] discarding cc correlated genes. 
    Taken from Cospar, Wang et al., 2023.

    Arguments:
        adata: Annotated data matrix with rows for cells and columns for genes.
        organism: Organism of the dataset. Available choices are ``"human"`` or ``"mouse"``.
        corr_threshold: Threshold for correlation with cc genes. Genes having a correlation with cc genes > corr_threshold will be discarded.
    """
    # Get cc genes
    cycling_genes = load_signatures_from_file(predefined_signatures[f'cell_cycle_{organism}'])
    cc_genes = list(set(cycling_genes['G1/S']) | set(cycling_genes['G2/M']))
    cc_genes = [ x for x in cc_genes if x in adata.var_names ]
   
    # Compute corr
    cc_expression = adata[:, cc_genes].X.A.T
    HVGs = adata.var_names[adata.var['highly_variable_features']]
    hvgs_expression = adata[:, HVGs].X.A.T
    cc_corr = corr2_coeff(hvgs_expression, cc_expression)

    # Discard genes having the maximum correlation with one of the cc > corr_threshold
    max_corr = np.max(abs(cc_corr), 1)
    HVGs_no_cc = HVGs[max_corr < corr_threshold]
    print(
        f'Number of selected non-cycling highly variable genes: {HVGs_no_cc.size}\n'
        f'{np.sum(max_corr > corr_threshold)} cell cycle correlated genes will be removed...'
    )
    # Update 
    adata.var['highly_variable_features'] = adata.var_names.isin(HVGs_no_cc)

from sklearn.cluster import KMeans  


def preprocess(adata, mode='shiftlog|pearson', target_sum=50*1e4, n_HVGs=2000,
    organism='human', no_cc=False):
    """
    Preprocesses the AnnData object adata using either a scanpy or a pearson residuals workflow for size normalization
    and highly variable genes (HVGs) selection, and calculates signature scores if necessary. 

    Arguments:
        adata: The data matrix.
        mode: The mode for size normalization and HVGs selection. It can be either 'scanpy' or 'pearson'. If 'scanpy', performs size normalization using scanpy's normalize_total() function and selects HVGs 
            using pegasus' highly_variable_features() function with batch correction. If 'pearson', selects HVGs 
            using scanpy's experimental.pp.highly_variable_genes() function with pearson residuals method and performs 
            size normalization using scanpy's experimental.pp.normalize_pearson_residuals() function. 
        target_sum: The target total count after normalization.
        n_HVGs: the number of HVGs to select.
        organism: The organism of the data. It can be either 'human' or 'mouse'. 
        no_cc: Whether to remove cc-correlated genes from HVGs.

    Returns:
        adata: The preprocessed data matrix. 
    """

    # Log-normalization, HVGs identification
    print('Begin robust gene identification')
    adata.raw = adata.copy()
    identify_robust_genes(adata, percent_cells=0.05)
    adata = adata[:, adata.var['robust']]
    print(f'End of robust gene identification.')
    method_list = mode.split('|')
    print(f'Begin size normalization: {method_list[0]} and HVGs selection {method_list[1]}')
    adata.layers['counts'] = adata.X.copy()

    if method_list[0] == 'shiftlog': # Size normalization + scanpy batch aware HVGs selection
        sc.pp.normalize_total(
            adata, 
            target_sum=target_sum,
            exclude_highly_expressed=True,
            max_fraction=0.2
        )
        sc.pp.log1p(adata)
    elif method_list[0] == 'pearson':
        # Perason residuals workflow
        sc.experimental.pp.normalize_pearson_residuals(adata)

    if method_list[1] == 'pearson': # Size normalization + scanpy batch aware HVGs selection
        sc.experimental.pp.highly_variable_genes(
            adata, 
            flavor="pearson_residuals",
            layer='counts',
            n_top_genes=n_HVGs,
        )
        if no_cc:
            remove_cc_genes(adata, organism=organism, corr_threshold=0.1)
    elif method_list[1] == 'seurat':
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            layer='counts',
            n_top_genes=n_HVGs,
        )
        if no_cc:
            remove_cc_genes(adata, organism=organism, corr_threshold=0.1)

    adata.var = adata.var.drop(columns=['highly_variable_features'])
    adata.var['highly_variable_features'] = adata.var['highly_variable']
    adata.var = adata.var.drop(columns=['highly_variable'])
    adata.var = adata.var.rename(columns={'means':'mean', 'variances':'var'})
    print(f'End of size normalization: {method_list[0]} and HVGs selection {method_list[1]}')
   
    return adata 

def normalize_pearson_residuals(adata,**kwargs):
    sc.experimental.pp.normalize_pearson_residuals(adata,kwargs)

def highly_variable_genes(adata,**kwargs):
    sc.experimental.pp.highly_variable_genes(
        adata, kwargs,
    )

def scale(adata,max_value=10):
    """
    Scale the input AnnData object.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. Adds a new layer called 'scaled' that stores
            the expression matrix that has been scaled to unit variance and zero mean.

    """
    adata_mock = sc.pp.scale(adata, copy=True,max_value=max_value)
    adata.layers['scaled'] = adata_mock.X.copy()
    del adata_mock

def regress(adata):
    """
    Regress out covariates from the input AnnData object.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. Should contain columns 'mito_perc' and 'nUMIs'
            that represent the percentage of mitochondrial genes and the total number of UMI counts, respectively.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. Adds a new layer called 'regressed' that stores
            the expression matrix with covariates regressed out.

    """
    adata_mock = sc.pp.regress_out(adata, ['mito_perc', 'nUMIs'], n_jobs=8, copy=True)
    adata.layers['regressed'] = adata_mock.X
    return adata

def regress_and_scale(adata):
    """
    Regress out covariates from the input AnnData object and scale the resulting expression matrix.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. Should contain a layer called 'regressed'
            that stores the expression matrix with covariates regressed out.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. Adds a new layer called 'regressed_and_scaled'
            that stores the expression matrix with covariates regressed out and then scaled.

    """
    if 'regressed' not in adata.layers:
        raise KeyError('Regress out covariates first!')
    adata_mock= adata.copy()
    adata_mock.X = adata_mock.layers['regressed']
    adata_mock = scale(adata_mock)
    adata.layers['regressed_and_scaled'] = adata_mock.layers['scaled']

    return adata

from sklearn.decomposition import PCA 
class my_PCA:
    """
    A class to store the results of a sklearn PCA (i.e., embeddings, loadings and 
    explained variance ratios).
    """
    def __init__(self):
        self.n_pcs = None
        self.embs = None
        self.loads = None
        self.var_ratios = None

    def calculate_PCA(self, M, n_components=50):
        '''
        Perform PCA decomposition of some input obs x genes matrix.
        '''
        self.n_pcs = n_components
        # Convert to dense np.array if necessary)
        if isinstance(M, np.ndarray) == False:
            M = M.toarray()

        # Perform PCA
        model = PCA(n_components=n_components, random_state=1234)
        # Store results accordingly
        self.embs = np.round(model.fit_transform(M), 2) # Round for reproducibility
        self.loads = model.components_.T
        self.var_ratios = model.explained_variance_ratio_
        self.cum_sum_eigenvalues = np.cumsum(self.var_ratios)

        return self

def pca(adata, n_pcs=50, layer='scaled',inplace=True):
    """
    Performs Principal Component Analysis (PCA) on the data stored in a scanpy AnnData object.

    Arguments:
        adata : Annotated data matrix with rows representing cells and columns representing features.
        n_pcs : Number of principal components to calculate.
        layer : The name of the layer in `adata` where the data to be analyzed is stored. Defaults to the 'scaled' layer,
            and falls back to 'lognorm' if that layer does not exist. Raises a KeyError if the specified layer is not present.

    Returns:
        adata : The original AnnData object with the calculated PCA embeddings and other information stored in its `obsm`, `varm`,
            and `uns` fields.
    """
    if 'lognorm' not in adata.layers:
        adata.layers['lognorm'] = adata.X
    if layer in adata.layers: 
        X = adata.layers[layer]
        key = f'{layer}|original'
    else:
        raise KeyError(f'Selected layer {layer} is not present. Compute it first!')

    model = my_PCA()
    model.calculate_PCA(X, n_components=n_pcs)
    adata.obsm[key + '|X_pca'] = model.embs
    adata.varm[key + '|pca_loadings'] = model.loads
    adata.uns[key + '|pca_var_ratios'] = model.var_ratios
    adata.uns[key + '|cum_sum_eigenvalues'] = np.cumsum(model.var_ratios)
    if inplace:
        return None
    else:
        return adata  

def red(adata):
    """
    Reduce the input AnnData object to highly variable features and store the resulting expression matrices.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. Should contain a variable 'highly_variable_features'
            that indicates which features are considered to be highly variable.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. Adds new layers called 'lognorm' and 'raw' that store
            the logarithmized normalized expression matrix and the unnormalized expression matrix, respectively.
            The matrix is reduced to the highly variable features only.

    """
    adata = adata[:, adata.var['highly_variable_features']].copy()
    adata.layers['lognorm'] = adata.X
    adata.layers['raw'] = adata.raw.to_adata()[:, adata.var_names].X
    return adata

def counts_store(adata,layers):
    adata.uns[layers] = adata.X.to_df().copy()

def counts_retrieve(adata,layers):
    cell_idx=adata.obs.index
    adata.uns['raw_store'] = adata.X.to_df().copy()
    adata.X=adata.uns[layers].loc[cell_idx,:].values

from scipy.stats import median_abs_deviation
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


from types import MappingProxyType
from typing import (
    Union,
    Optional,
    Any,
    Mapping,
    Callable,
    NamedTuple,
    Generator,
    Tuple,
    Literal,
)
N_DCS = 15  # default number of diffusion components

_Method = Literal['umap', 'gauss', 'rapids']
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
# from sklearn.metrics.pairwise_distances.__doc__:
_MetricSparseCapable = Literal[
    'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
]
_MetricScipySpatial = Literal[
    'braycurtis',
    'canberra',
    'chebyshev',
    'correlation',
    'dice',
    'hamming',
    'jaccard',
    'kulsinski',
    'mahalanobis',
    'minkowski',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule',
]
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]


def neighbors(
    adata: anndata.AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    knn: bool = True,
    random_state: int= 0,
    method: Optional[_Method] = 'umap',
    metric: Union[_Metric, _MetricFn] = 'euclidean',
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Arguments:
        adata: Annotated data matrix.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring data
            points) used for manifold approximation. Larger values result in more
            global views of the manifold, while smaller values result in more local
            data being preserved. In general values should be in the range 2 to 100.
            If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
            is `False`, a Gaussian kernel width is set to the distance of the
            `n_neighbors` neighbor.
        knn: If `True`, use a hard threshold to restrict the number of neighbors to
            `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
            Kernel to assign low weights to neighbors more distant than the
            `n_neighbors` nearest neighbor.
        random_state: A numpy random seed.
        method: Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_
            with adaptive width [Haghverdi16]_) for computing connectivities.
            Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU
            only).
        metric: A known metric’s name or a callable that returns a distance.
        metric_kwds: Options for the metric.
        key_added: If not specified, the neighbors data is stored in .uns['neighbors'],
            distances and connectivities are stored in .obsp['distances'] and
            .obsp['connectivities'] respectively.
            If specified, the neighbors data is added to .uns[key_added],
            distances are stored in .obsp[key_added+'_distances'] and
            connectivities in .obsp[key_added+'_connectivities'].
        copy: Return a copy instead of writing to adata.

    Returns: Depending on `copy`, updates or returns `adata` with the following:

    See `key_added` parameter description for the storage path of
    connectivities and distances.

    **connectivities** : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.

    Notes
    -----
    If `method='umap'`, it's highly recommended to install pynndescent ``pip install pynndescent``.
    Installing `pynndescent` can significantly increase performance,
    and in later versions it will become a hard dependency.
    
    """
    return sc.pp.neighbors(adata,n_neighbors,
                           n_pcs,use_rep,knn,random_state,
                           method,metric,metric_kwds,key_added) if copy else None