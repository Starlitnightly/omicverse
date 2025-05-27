"""
Copy from pegasus and cellual

"""

from typing import Union, Tuple, Optional, Sequence, List, Dict
import anndata
import numpy as np
import pandas as pd
import skmisc.loess as sl
import scanpy as sc
import time 

from scipy.sparse import issparse, csr_matrix
from ..utils import load_signatures_from_file,predefined_signatures
from .._settings import settings,print_gpu_usage_color,EMOJI,add_reference
from datetime import datetime



# Emoji map for UMAP status reporting


def identify_robust_genes(data: anndata.AnnData, percent_cells: float = 0.05) -> None:
    """ 
    Identify robust genes as candidates for HVG selection and remove genes 
    that are not expressed in any cells.

    Arguments:
        data: Use current selected modality in data, which should contain one RNA expression matrix.
        percent_cells: Only assign genes to be ``robust`` that are expressed in at least 
        ``percent_cells`` % of cells.


    Update ``data.var``:

        * ``n_cells``: Total number of cells in which each gene is measured.
        * ``percent_cells``: Percent of cells in which each gene is measured.
        * ``robust``: Boolean type indicating if a gene is robust based on the QC metrics.
        * ``highly_variable_features``: Boolean type indicating if a gene 
        is a highly variable feature. 
        By default, set all robust genes as highly variable features.

    """

    prior_n = data.shape[1]

    if issparse(data.X):
        data.var["n_cells"] = data.X.getnnz(axis=0)
        data._inplace_subset_var(data.var["n_cells"] > 0)
        data.var["percent_cells"] = (data.var["n_cells"] / data.shape[0]) * 100
        data.var["robust"] = data.var["percent_cells"] >= percent_cells
    else:
        data.var["robust"] = True

    data.var["highly_variable_features"] = data.var["robust"]  
    # default all robust genes are "highly" variable
    print(f"After filtration, {data.shape[1]}/{prior_n} genes are kept. \
    Among {data.shape[1]} genes, {data.var['robust'].sum()} genes are robust.")

def calc_mean_and_var(X: Union[csr_matrix, np.ndarray], axis: int) -> Tuple[np.ndarray, np.ndarray]:
    if issparse(X):
        from ..cylib.fast_utils import calc_mean_and_var_sparse
        return calc_mean_and_var_sparse(X.shape[0], X.shape[1], X.data, X.indices, X.indptr, axis)
    else:
        from ..cylib.fast_utils import calc_mean_and_var_dense
        return calc_mean_and_var_dense(X.shape[0], X.shape[1], X, axis)
def calc_stat_per_batch(X: Union[csr_matrix, np.ndarray], batch: \
    Union[pd.Categorical, np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    '''
    exponential minus one
    
    '''
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
        batch: A key in data.obs specifying batch information. 
        If `batch` is not set, do not consider batch effects in selecting highly variable features. 
        Otherwise, if `data.obs[batch]` is not categorical, 
        `data.obs[batch]` will be automatically converted into categorical 
        before highly variable feature selection.
        flavor: The HVF selection method to use. 
        Available choices are ``"pegasus"`` or ``"Seurat"``.
        n_top: Number of genes to be selected as HVF. if ``None``, no gene will be selected.
        span: Only applicable when ``flavor`` is ``"pegasus"``. 
        The smoothing factor used by *scikit-learn loess* model in pegasus HVF selection method.
        min_disp: Minimum normalized dispersion.
        max_disp: Maximum normalized dispersion. Set it to ``np.inf`` for infinity bound.
        min_mean: Minimum mean.
        max_mean: Maximum mean.
        n_jobs: Number of threads to be used during calculation. 
        If ``-1``, all physical CPU cores will be used.


    Update ``adata.var``:
        * ``highly_variable_features``: replace with Boolean type array 
        indicating the selected highly variable features.

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

    print(f"{data.var['highly_variable_features'].sum()} \
        highly variable features have been selected.")

def fit_loess(x: List[float], y: List[float], span: float, degree: int) -> object:
    '''
    A LOESS (Locally Weighted Regression) model is used to fit a given data set
    '''
    try:
        lobj = sl.loess(x, y, span=span, degree=degree)
        lobj.fit()
        return lobj
    except ValueError:
        return None
def corr2_coeff(a, b):
    """
    Calculate Pearson correlation between matrix a and b
    a and b are allowed to have different shapes. Taken from Cospar, Wang et al., 2023.
    """
    resol = 10 ** (-15)

    a_ma = a - a.mean(1)[:, None]
    b_mb = b - b.mean(1)[:, None]
    ssa = (a_ma ** 2).sum(1)
    ssb = (b_mb ** 2).sum(1)

    corr = np.dot(a_ma, b_mb.T) / (np.sqrt(np.dot(ssa[:, None], ssb[None])) + resol)

    return corr

def remove_cc_genes(adata:anndata.AnnData, organism:str='human', corr_threshold:float=0.1):
    """
    Update adata.var['highly_variable_features'] discarding cc correlated genes. 
    Taken from Cospar, Wang et al., 2023.

    Arguments:
        adata: Annotated data matrix with rows for cells and columns for genes.
        organism: Organism of the dataset. Available choices are ``"human"`` or ``"mouse"``.
        corr_threshold: Threshold for correlation with cc genes. 
        Genes having a correlation with cc genes > corr_threshold will be discarded.
    """
    # Get cc genes
    cycling_genes = load_signatures_from_file(predefined_signatures[f'cell_cycle_{organism}'])
    cc_genes = list(set(cycling_genes['G1/S']) | set(cycling_genes['G2/M']))
    cc_genes = [ x for x in cc_genes if x in adata.var_names ]
    # Compute corr
    cc_expression = adata[:, cc_genes].X.A.T
    hvgs = adata.var_names[adata.var['highly_variable_features']]
    hvgs_expression = adata[:, hvgs].X.A.T
    cc_corr = corr2_coeff(hvgs_expression, cc_expression)

    # Discard genes having the maximum correlation with one of the cc > corr_threshold
    max_corr = np.max(abs(cc_corr), 1)
    hvgs_no_cc = hvgs[max_corr < corr_threshold]
    print(
        f'Number of selected non-cycling highly variable genes: {hvgs_no_cc.size}\n'
        f'{np.sum(max_corr > corr_threshold)} cell cycle correlated genes will be removed...'
    )
    # Update
    adata.var['highly_variable_features'] = adata.var_names.isin(hvgs_no_cc)

from sklearn.cluster import KMeans  

def anndata_to_GPU(adata,**kwargs):
    '''
    Migrate the data of AnnData objects to the GPU for processing
    '''
    import rapids_singlecell as rsc
    rsc.get.anndata_to_GPU(adata,**kwargs)
    print('Data has been moved to GPU')
    print('Don`t forget to move it back to CPU after analysis is done')
    print('Use `ov.pp.anndata_to_CPU(adata)`')

def anndata_to_CPU(adata,layer=None, convert_all=True, copy=False):
    '''
    Migrate the data of AnnData objects to the CPU for processing
    '''
    import rapids_singlecell as rsc
    rsc.get.anndata_to_CPU(adata,layer=layer, convert_all=convert_all, copy=copy)


def preprocess(adata, mode='shiftlog|pearson', target_sum=50*1e4, n_HVGs=2000,
    organism='human', no_cc=False,batch_key=None,):
    """
    Preprocesses the AnnData object adata using either a scanpy or 
    a pearson residuals workflow for size normalization
    and highly variable genes (HVGs) selection, and calculates signature scores if necessary. 

    Arguments:
        adata: The data matrix.
        mode: The mode for size normalization and HVGs selection. 
        It can be either 'scanpy' or 'pearson'. If 'scanpy', 
        performs size normalization using scanpy's normalize_total() function and selects HVGs 
        using pegasus' highly_variable_features() function with batch correction. 
        If 'pearson', selects HVGs sing scanpy's experimental.pp.highly_variable_genes() function
        with pearson residuals method and performs 
        size normalization using scanpy's experimental.pp.normalize_pearson_residuals() function. 
        target_sum: The target total count after normalization.
        n_HVGs: the number of HVGs to select.
        organism: The organism of the data. It can be either 'human' or 'mouse'. 
        no_cc: Whether to remove cc-correlated genes from HVGs.

    Returns:
        adata: The preprocessed data matrix. 
    """

    # Log-normalization, HVGs identification
    adata.layers['counts'] = adata.X.copy()
    print('Begin robust gene identification')
    identify_robust_genes(adata, percent_cells=0.05)
    adata = adata[:, adata.var['robust']]
    print('End of robust gene identification.')
    method_list = mode.split('|')
    print(f'Begin size normalization: {method_list[0]} and HVGs selection {method_list[1]}')
    if settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed':
        data_load_start = time.time()
        if method_list[0] == 'shiftlog': # Size normalization + scanpy batch aware HVGs selection
            sc.pp.normalize_total(
                adata,
                target_sum=target_sum,
                exclude_highly_expressed=True,
                max_fraction=0.2,
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
                batch_key=batch_key,
            )
            if no_cc:
                remove_cc_genes(adata, organism=organism, corr_threshold=0.1)
        elif method_list[1] == 'seurat':
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
            if no_cc:
                remove_cc_genes(adata, organism=organism, corr_threshold=0.1)
        data_load_end = time.time()
        print(f'Time to analyze data in cpu: {data_load_end - data_load_start} seconds.')
    else:
        import rapids_singlecell as rsc
        data_load_start = time.time()
        if method_list[0] == 'shiftlog': # Size normalization + scanpy batch aware HVGs selection
            rsc.pp.normalize_total(adata, target_sum=target_sum)
            rsc.pp.log1p(adata)
        elif method_list[0] == 'pearson':
            # Perason residuals workflow
            rsc.pp.normalize_pearson_residuals(adata)
        if method_list[1] == 'pearson': # Size normalization + scanpy batch aware HVGs selection
            rsc.pp.highly_variable_genes(
                adata, 
                flavor="pearson_residuals",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
        elif method_list[1] == 'seurat':
            rsc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
        data_load_end = time.time()
        print(f'Time to analyze data in gpu: {data_load_end - data_load_start} seconds.')


    adata.var = adata.var.drop(columns=['highly_variable_features'])
    adata.var['highly_variable_features'] = adata.var['highly_variable']
    adata.var = adata.var.drop(columns=['highly_variable'])
    #adata.var = adata.var.rename(columns={'means':'mean', 'variances':'var'})
    print(f'End of size normalization: {method_list[0]} and HVGs selection {method_list[1]}')

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['preprocess']=True
    adata.uns['status_args']['preprocess']={
        'mode':mode, 'target_sum':target_sum, 'n_HVGs':n_HVGs,
        'organism':organism,
    }
    add_reference(adata,'scanpy','size normalization with scanpy')
    return adata
def normalize_pearson_residuals(adata,**kwargs):
    '''
    normalize
    '''

    sc.experimental.pp.normalize_pearson_residuals(adata,kwargs)

def highly_variable_genes(adata,**kwargs):
    '''
    highly_variable_genes calculation
    '''
    sc.experimental.pp.highly_variable_genes(
        adata, kwargs,
    )

def scale(adata,max_value=10,layers_add='scaled'):
    """
    Scale the input AnnData object.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Adds a new layer called 'scaled' that stores
            the expression matrix that has been scaled to unit variance and zero mean.

    """
    if settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed':
        adata_mock = sc.pp.scale(adata, copy=True,max_value=max_value)
        adata.layers[layers_add] = adata_mock.X.copy()
        del adata_mock
    else:
        import rapids_singlecell as rsc
        adata.layers['scaled']=rsc.pp.scale(adata, max_value=max_value,inplace=False)

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}
    add_reference(adata,'scanpy','scaling with scanpy')
    adata.uns['status']['scaled'] = True

def regress(adata,**kwargs):
    """
    Regress out covariates from the input AnnData object.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Should contain columns 'mito_perc' and 'nUMIs'that represent the percentage of 
        mitochondrial genes and the total number of UMI counts, respectively.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Adds a new layer called 'regressed' that stores
            the expression matrix with covariates regressed out.

    """
    if settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed':
        adata_mock = sc.pp.regress_out(adata, ['mito_perc', 'nUMIs'], n_jobs=8, copy=True,**kwargs)
        adata.layers['regressed'] = adata_mock.X.copy()
        del adata_mock
    else:
        import rapids_singlecell as rsc
        adata.layers['regressed']=rsc.pp.regress_out(adata, ['mito_perc', 'nUMIs'], inplace=False)
    add_reference(adata,'scanpy','regressing out covariates with scanpy')

def regress_and_scale(adata):
    """
    Regress out covariates from the input AnnData object and scale the resulting expression matrix.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Should contain a layer called 'regressed'
            that stores the expression matrix with covariates regressed out.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Adds a new layer called 'regressed_and_scaled'
            that stores the expression matrix with covariates regressed out and then scaled.

    """
    if 'regressed' not in adata.layers:
        raise KeyError('Regress out covariates first!')
    adata_mock= adata.copy()
    adata_mock.X = adata_mock.layers['regressed']
    scale(adata_mock)
    adata.layers['regressed_and_scaled'] = adata_mock.layers['scaled']
    add_reference(adata,'scanpy','regressing out covariates with scanpy')
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
        if isinstance(M, np.ndarray) is False:
            M = M.toarray()

        # Perform PCA
        model = PCA(n_components=n_components, random_state=1234)
        # Store results accordingly
        self.embs = np.round(model.fit_transform(M), 2) # Round for reproducibility
        self.loads = model.components_.T
        self.var_ratios = model.explained_variance_ratio_
        self.cum_sum_eigenvalues = np.cumsum(self.var_ratios)

        return self

def pca(adata, n_pcs=50, layer='scaled',inplace=True,**kwargs):
    """
    Performs Principal Component Analysis (PCA) on the data stored in a scanpy AnnData object.

    Arguments:
        adata : Annotated data matrix with rows representing cells 
        and columns representing features.
        n_pcs : Number of principal components to calculate.
        layer : The name of the layer in `adata` where the data to be analyzed is stored. 
        Defaults to the 'scaled' layer,
            and falls back to 'lognorm' if that layer does not exist. 
        Raises a KeyError if the specified layer is not present.

    Returns:
        adata : The original AnnData object with the calculated PCA embeddings 
        and other information stored in its `obsm`, `varm`,
            and `uns` fields.
    """
    #if 'lognorm' not in adata.layers:
    #    adata.layers['lognorm'] = adata.X
    if layer in adata.layers:
        X = adata.layers[layer]
        key = f'{layer}|original'
    else:
        raise KeyError(f'Selected layer {layer} is not present. Compute it first!')
    if settings.mode == 'cpu':
        if sc.__version__ <'1.10':
            adata_mock=sc.AnnData(adata.layers[layer],obs=adata.obs,var=adata.var)
            sc.pp.pca(adata_mock, n_comps=n_pcs)
            adata.obsm[key + '|X_pca'] = adata_mock.obsm['X_pca']
            adata.varm[key + '|pca_loadings'] = adata_mock.varm['PCs']
            adata.uns[key + '|pca_var_ratios'] = adata_mock.uns['pca']['variance_ratio']
            adata.uns[key + '|cum_sum_eigenvalues'] = adata_mock.uns['pca']['variance']
        else:
            sc.pp.pca(adata, layer=layer,n_comps=n_pcs,**kwargs)
            adata.obsm[key + '|X_pca'] = adata.obsm['X_pca']
            adata.varm[key + '|pca_loadings'] = adata.varm['PCs']
            adata.uns[key + '|pca_var_ratios'] = adata.uns['pca']['variance_ratio']
            adata.uns[key + '|cum_sum_eigenvalues'] = adata.uns['pca']['variance']
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['gpu']} Using GPU to calculate PCA...")
        print_gpu_usage_color()
        from ._pca import pca as _pca
        _pca(adata, layer=layer,n_comps=n_pcs,use_gpu=True,**kwargs)
        adata.obsm[key + '|X_pca'] = adata.obsm['X_pca']
        adata.varm[key + '|pca_loadings'] = adata.varm['PCs']
        adata.uns[key + '|pca_var_ratios'] = adata.uns['pca']['variance_ratio']
        adata.uns[key + '|cum_sum_eigenvalues'] = adata.uns['pca']['variance']
    else:
        import rapids_singlecell as rsc
        rsc.pp.pca(adata, layer=layer,n_comps=n_pcs)
        adata.obsm[key + '|X_pca'] = adata.obsm['X_pca']
        adata.varm[key + '|pca_loadings'] = adata.varm['PCs']
        adata.uns[key + '|pca_var_ratios'] = adata.uns['pca']['variance_ratio']
        adata.uns[key + '|cum_sum_eigenvalues'] = adata.uns['pca']['variance']
    
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['pca'] = True
    adata.uns['status_args']['pca']={
        'layer':layer,
        'n_pcs':n_pcs,
    }
    add_reference(adata,'scanpy','PCA with scanpy')
    if inplace:
        return None
    else:
        return adata

def red(adata):
    """
    Reduce the input AnnData object to highly variable features 
    and store the resulting expression matrices.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Should contain a variable 'highly_variable_features'that 
        indicates which features are considered to be highly variable.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Adds new layers called 'lognorm' and 'raw' that store
            the logarithmized normalized expression matrix and 
            the unnormalized expression matrix, respectively.
            The matrix is reduced to the highly variable features only.

    """
    adata = adata[:, adata.var['highly_variable_features']].copy()
    adata.layers['lognorm'] = adata.X
    adata.layers['raw'] = adata.raw.to_adata()[:, adata.var_names].X
    return adata

def counts_store(adata,layers):
    '''
    counts store
    '''
    adata.uns[layers] = adata.X.to_df().copy()

def counts_retrieve(adata,layers):
    '''
    counts retrieve
    '''
    cell_idx=adata.obs.index
    adata.uns['raw_store'] = adata.X.to_df().copy()
    adata.X=adata.uns[layers].loc[cell_idx,:].values

from scipy.stats import median_abs_deviation
def is_outlier(adata, metric: str, nmads: int):
    '''
    Identify outliers based on specific metrics in a given AnnData object
    '''
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
'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
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
from types import MappingProxyType
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
    if settings.mode =='cpu' or settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['cpu']} Using Scanpy CPU to calculate neighbors...")
        sc.pp.neighbors(adata,use_rep=use_rep,n_neighbors=n_neighbors, n_pcs=n_pcs,
                         random_state=random_state,method=method,metric=metric,
                         metric_kwds=metric_kwds,
                         key_added=key_added,copy=copy)
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate neighbors...")
        import rapids_singlecell as rsc
        rsc.pp.neighbors(adata,use_rep=use_rep,n_neighbors=n_neighbors, n_pcs=n_pcs,
                         random_state=random_state,algorithm=method,metric=metric,
                         metric_kwds=metric_kwds,
                         key_added=key_added,copy=copy)
    add_reference(adata,'scanpy','neighbors with scanpy')


def umap(adata, **kwargs):
    """
    Run UMAP on AnnData, choosing implementation based on settings.mode,
    The argument could be found in `scanpy.pp.umap`
    """
    print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running UMAP in '{settings.mode}' mode...")
    try:
        if settings.mode == 'cpu':
            print(f"{EMOJI['cpu']} Using Scanpy CPU UMAP...")
            sc.tl.umap(adata, **kwargs)
            add_reference(adata,'umap','UMAP with scanpy')

        elif settings.mode == 'cpu-gpu-mixed':
            print(f"{EMOJI['gpu']} Using torch GPU to calculate UMAP...")
            print_gpu_usage_color()
            from ._umap import umap as _umap
            _umap(adata,method='mde', **kwargs)
            add_reference(adata,'pymde','UMAP with pymde')
            add_reference(adata,'umap','UMAP with pymde')
            

        else:
            print(f"{EMOJI['gpu']} Using RAPIDS GPU UMAP...")
            import rapids_singlecell as rsc
            rsc.tl.umap(adata, **kwargs)
            add_reference(adata,'umap','UMAP with RAPIDS')

        print(f"{EMOJI['done']} UMAP completed successfully.")
    except Exception as e:
        print(f"{EMOJI['error']} UMAP failed: {e}")
        raise


def louvain(adata, **kwargs):
    '''
    Louvain clustering
    '''

    if settings.mode =='cpu' or settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['cpu']} Using Scanpy CPU Louvain...")
        sc.tl.louvain(adata, **kwargs)
        add_reference(adata,'louvain','Louvain clustering with scanpy')
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate Louvain...")
        import rapids_singlecell as rsc
        rsc.tl.louvain(adata, **kwargs)
        add_reference(adata,'louvain','Louvain clustering with RAPIDS')

def leiden(adata, **kwargs):
    '''
    leiden clustering
    '''

    if settings.mode =='cpu' or settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['cpu']} Using Scanpy CPU Leiden...")
        sc.tl.leiden(adata, **kwargs)
        add_reference(adata,'leiden','Leiden clustering with scanpy')
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate Leiden...")
        import rapids_singlecell as rsc
        rsc.tl.leiden(adata, **kwargs)
        add_reference(adata,'leiden','Leiden clustering with RAPIDS')


def score_genes_cell_cycle(adata,species='human',s_genes=None, g2m_genes=None):
    """
    Score cell cycle .

    Arguments:
        adata: Annotated data matrix with rows for cells and columns for genes.
        species: The species of the data. It can be either 'human' or 'mouse'.
        s_genes: The list of genes that are specific to the S phase of the cell cycle.
        g2m_genes: The list of genes that are specific to the G2/M phase of the cell cycle.
    
    """
    if s_genes is None:
        if species=='human':
            s_genes=['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 
            'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 
            'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 
            'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
            'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
             'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
        elif species=='mouse':
            s_genes=['Cdca7', 'Mcm4', 'Mcm7', 'Rfc2', 'Ung', 'Mcm6', 
            'Rrm1', 'Slbp', 'Pcna', 'Atad2', 'Tipin', 'Mcm5', 'Uhrf1', 
            'Polr1b', 'Dtl', 'Prim1', 'Fen1', 'Hells', 'Gmnn', 'Pold3', 
            'Nasp', 'Chaf1b', 'Gins2', 'Pola1', 'Msh2', 'Casp8ap2', 'Cdc6',
             'Ubr7', 'Ccne2', 'Wdr76', 'Tyms', 'Cdc45', 'Clspn', 'Rrm2', 
             'Dscc1', 'Rad51', 'Usp1', 'Exo1', 'Blm', 'Rad51ap1', 'Cenpu', 'E2f8', 'Mrpl36']       
    if g2m_genes is None:
        if species=='human':
            g2m_genes=['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 
            'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 
            'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 
            'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1',
             'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C',
              'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR',
               'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 
               'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
        elif species=='mouse':
            g2m_genes=['Cbx5', 'Aurkb', 'Cks1b', 'Cks2', 'Jpt1', 'Hmgb2', 
            'Anp32e', 'Lbr', 'Tmpo', 'Top2a', 'Tacc3', 'Tubb4b', 'Ncapd2', 
            'Rangap1', 'Cdk1', 'Smc4', 'Kif20b', 'Cdca8', 'Ckap2', 'Ndc80',
             'Dlgap5', 'Hjurp', 'Ckap5', 'Bub1', 'Ckap2l', 'Ect2', 'Kif11',
              'Birc5', 'Cdca2', 'Nuf2', 'Cdca3', 'Nusap1', 'Ttk', 'Aurka', 
              'Mki67', 'Pimreg', 'Ccnb2', 'Tpx2', 'Hjurp', 'Anln', 'Kif2c',
               'Cenpe', 'Gtse1', 'Kif23', 'Cdc20', 'Ube2c', 'Cenpf', 'Cenpa',
                'Hmmr', 'Ctcf', 'Psrc1', 'Cdc25c', 'Nek2', 'Gas2l3', 'G2e3']
    sc.tl.score_genes_cell_cycle(adata,s_genes=s_genes, g2m_genes=g2m_genes)
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}
        
    adata.uns['status']['cell_cycle'] = True
    adata.uns['status_args']['cell_cycle']={
        's_genes':s_genes,
        'g2m_genes':g2m_genes
    }


def tsne(adata,**kwargs):
    '''
    t-SNE
    '''
    if settings.mode == 'cpu':
        print(f"{EMOJI['cpu']} Using Scanpy CPU t-SNE...")
        sc.tl.tsne(adata, **kwargs)
        add_reference(adata,'tsne','t-SNE with scanpy')
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['mixed']} Using torch CPU/GPU mixed mode to calculate t-SNE...")
        print_gpu_usage_color()
        from ._tsne import tsne as _tsne
        _tsne(adata, **kwargs)
        add_reference(adata,'tsne','t-SNE with omicverse')
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate t-SNE...")
        import rapids_singlecell as rsc
        rsc.tl.tsne(adata, **kwargs)
        add_reference(adata,'tsne','t-SNE with RAPIDS')


def mde(adata,embedding_dim=2,n_neighbors=15, basis='X_mde',n_pcs=None, use_rep=None, knn=True, 
        transformer=None, metric='euclidean',verbose=False,
        key_added=None,random_state=0,repulsive_fraction=0.7,constraint=None):
    '''
    MDE
    '''
    import pymde
    import logging
    import time

    # 配置日志
    #logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    #logger = logging.getLogger()
    # 记录开始时间
    start_time = time.time()

    print("computing neighbors")
    if use_rep is None:
        use_rep='X_pca'
    data=adata.obsm[use_rep]
    if n_pcs is None:
        n_pcs=50
    data=data[:,:n_pcs]

    if constraint is None:
        _kwargs = {
        "embedding_dim": embedding_dim,
        "constraint": pymde.Standardized(),
        "repulsive_fraction": repulsive_fraction,
        "verbose": verbose,
        "device": 'cuda',
        "n_neighbors": n_neighbors,
    }
    else:
        _kwargs = {
        "embedding_dim": embedding_dim,
        "constraint": constraint,
        "repulsive_fraction": repulsive_fraction,
        "verbose": verbose,
        "device": 'cuda',
        "n_neighbors": n_neighbors,
    }
    #_kwargs.update(kwargs)
    
    gr=pymde.preprocess.k_nearest_neighbors(data,k=n_neighbors)

    mde = pymde.preserve_neighbors(data, **_kwargs)
    import torch
    emb=mde.embed(verbose=_kwargs["verbose"])

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    from scipy.spatial import KDTree
    # 使用KNN算法找到最近邻居
    n_neighbors = n_neighbors  # 设置KNN的邻居数
    kdtree = KDTree(emb)
    distances, indices = kdtree.query(emb, k=n_neighbors)

    # 构建稀疏的距离矩阵
    n_items = emb.shape[0]
    row_indices = np.repeat(np.arange(n_items), n_neighbors)
    col_indices = indices.flatten()
    distances = distances.flatten()
    sparse_distance_matrix = csr_matrix((distances, (row_indices, col_indices)), \
    shape=(n_items, n_items))

    # 构建连接矩阵
    # 这里我们简单地将连接矩阵设置为距离矩阵的二值化形式
    connectivities = (sparse_distance_matrix > 0).astype(float)

    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"

    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key

    neighbors_dict["params"] = dict(
        n_neighbors=n_neighbors,
        method='mde',
        random_state=random_state,
        metric=metric,
    )
    if use_rep is not None:
        neighbors_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        neighbors_dict["params"]["n_pcs"] = n_pcs


    # 创建或更新AnnData对象
    #adata = anndata.AnnData(X=data)
    adata.obsp[dists_key] = sparse_distance_matrix
    adata.obsp[conns_key] = gr.adjacency_matrix
    adata.obsm[basis]=emb
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印结果和日志信息
    print("    finished: added to `.uns['neighbors']`")
    print(f"    `.obsm['{basis}']`, MDE coordinates")
    if key_added is None:
        print("    `.obsp['distances']`, distances for each pair of neighbors")
        print("    `.obsp['connectivities']`, weighted adjacency matrix (0:{:02}:{:02})".format(int(elapsed_time // 60), int(elapsed_time % 60)))

    else:
        print(f"    `.obsp['{key_added}_distances']`, distances for each pair of neighbors")
        print("    `.obsp['{}_connectivities']`, weighted adjacency matrix (0:{:02}:{:02})".format(key_added,int(elapsed_time // 60), int(elapsed_time % 60)))
    add_reference(adata,'pymde','MDE with pymde')
    #return emb
