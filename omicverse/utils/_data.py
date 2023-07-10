r"""
Pyomic data (Pyomic.utils._data)
"""


import time
import requests
import os
import pandas as pd
import scanpy as sc
from ._genomics import read_gtf,Gtf
import anndata
import numpy as np
from typing import Callable, List, Mapping, Optional,Dict
from ._enum import ModeEnum
from scipy.sparse import diags, issparse, spmatrix, csr_matrix, isspmatrix_csr
import warnings
from scipy.stats import norm


def read(path,**kwargs):
    if path.split('.')[-1]=='h5ad':
        return sc.read(path,**kwargs)
    elif path.split('.')[-1]=='csv':
        return pd.read_csv(path,**kwargs)
    elif path.split('.')[-1]=='tsv' or path.split('.')[-1]=='txt':
        return pd.read_csv(path,sep='\t',**kwargs)
    elif path.split('.')[-1]=='gz':
        if path.split('.')[-2]=='csv':
            return pd.read_csv(path,**kwargs)
        elif path.split('.')[-2]=='tsv' or path.split('.')[-2]=='txt':
            return pd.read_csv(path,sep='\t',**kwargs)
    else:
        raise ValueError('The type is not supported.')
    
def read_csv(**kwargs):
    return pd.read_csv(**kwargs)

def read_10x_mtx(**kwargs):
    return sc.read_10x_mtx(**kwargs)

def read_h5ad(**kwargs):
    return sc.read_h5ad(**kwargs)

def read_10x_h5(**kwargs):
    return sc.read_10x_h5(**kwargs)


def data_downloader(url,path,title):
    r"""datasets downloader
    
    Arguments
    ---------
    - url: `str`
        the download url of datasets
    - path: `str`
        the save path of datasets
    - title: `str`
        the name of datasets
    
    Returns
    -------
    - path: `str`
        the save path of datasets
    """
    if os.path.isfile(path):
        print("......Loading dataset from {}".format(path))
        return path
    else:
        print("......Downloading dataset save to {}".format(path))
        
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))
    
    start = time.time()
    size = 0
    res = requests.get(url, stream=True)

    chunk_size = 1024000
    content_size = int(res.headers["content-length"]) 
    if res.status_code == 200:
        print('......[%s Size of file]: %0.2f MB' % (title, content_size/chunk_size/10.24))
        with open(path, 'wb') as f:
            for data in res.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data) 
                print('\r'+ '......[Downloader]: %s%.2f%%' % ('>'*int(size*50/content_size), float(size/content_size*100)), end='')
        end = time.time()
        print('\n' + ".......Finish！%s.2f s" % (end - start))
    
    return path

def download_CaDRReS_model():
    r"""load CaDRReS_model
    
    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'cadrres-wo-sample-bias_output_dict_all_genes':'https://figshare.com/ndownloader/files/39753568',
        'cadrres-wo-sample-bias_output_dict_prism':'https://figshare.com/ndownloader/files/39753571',
        'cadrres-wo-sample-bias_param_dict_all_genes':'https://figshare.com/ndownloader/files/39753574',
        'cadrres-wo-sample-bias_param_dict_prism':'https://figshare.com/ndownloader/files/39753577',
    }
    for datasets_name in _datasets.keys():
        print('......CaDRReS model download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='models/{}.pickle'.format(datasets_name),title=datasets_name)
    print('......CaDRReS model download finished!')

def download_GDSC_data():
    r"""load GDSC_data
    
    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'masked_drugs':'https://figshare.com/ndownloader/files/39753580',
        'GDSC_exp':'https://figshare.com/ndownloader/files/39744025',
    }
    for datasets_name in _datasets.keys():
        print('......GDSC data download start:',datasets_name)
        if datasets_name == 'masked_drugs':
            data_downloader(url=_datasets[datasets_name],path='models/{}.csv'.format(datasets_name),title=datasets_name)
        elif datasets_name == 'GDSC_exp':
            data_downloader(url=_datasets[datasets_name],path='models/{}.tsv.gz'.format(datasets_name),title=datasets_name)
    print('......GDSC data download finished!')

def download_pathway_database():
    r"""load pathway_database

    """
    _datasets = {
        'GO_Biological_Process_2021':'https://figshare.com/ndownloader/files/39820720',
        'GO_Cellular_Component_2021':'https://figshare.com/ndownloader/files/39820714',
        'GO_Molecular_Function_2021':'https://figshare.com/ndownloader/files/39820711',
        'WikiPathway_2021_Human':'https://figshare.com/ndownloader/files/39820705',
        'WikiPathways_2019_Mouse':'https://figshare.com/ndownloader/files/39820717',
        'Reactome_2022':'https://figshare.com/ndownloader/files/39820702',
    }
     
    for datasets_name in _datasets.keys():
        print('......Pathway Geneset download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.txt'.format(datasets_name),title=datasets_name)
    print('......Pathway Geneset download finished!')
    print('......Other Genesets can be dowload in `https://maayanlab.cloud/Enrichr/#libraries`')

def download_geneid_annotation_pair():
    r"""load geneid_annotation_pair

    """
    _datasets = {
        'pair_GRCm39':'https://figshare.com/ndownloader/files/39820684',
        'pair_T2TCHM13':'https://figshare.com/ndownloader/files/39820687',
        'pair_GRCh38':'https://figshare.com/ndownloader/files/39820690',
        'pair_GRCh37':'https://figshare.com/ndownloader/files/39820693',
        'pair_danRer11':'https://figshare.com/ndownloader/files/39820696',
        'pair_danRer7':'https://figshare.com/ndownloader/files/39820699',
    }
     
    for datasets_name in _datasets.keys():
        print('......Geneid Annotation Pair download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.tsv'.format(datasets_name),title=datasets_name)
    print('......Geneid Annotation Pair download finished!')

def download_tosica_gmt():
    r"""load TOSICA gmt dataset

    """
    _datasets = {
        'GO_bp':'https://figshare.com/ndownloader/files/41460072',
        'TF':'https://figshare.com/ndownloader/files/41460066',
        'reactome':'https://figshare.com/ndownloader/files/41460051',
        'm_GO_bp':'https://figshare.com/ndownloader/files/41460060',
        'm_TF':'https://figshare.com/ndownloader/files/41460057',
        'm_reactome':'https://figshare.com/ndownloader/files/41460054',
        'immune':'https://figshare.com/ndownloader/files/41460063',
    }
     
    for datasets_name in _datasets.keys():
        print('......TOSICA gmt dataset download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.gmt'.format(datasets_name),title=datasets_name)
    print('......TOSICA gmt dataset download finished!')

def geneset_prepare(geneset_path,organism='Human'):
    r"""load geneset

    Parameters
    ----------
    - geneset_path: `str`
        Path of geneset file.
    - organism: `str`
        Organism of geneset file. Default: 'Human'

    Returns
    -------
    - go_bio_dict: `dict`
        A dictionary of geneset.
    """
    go_bio_geneset=pd.read_csv(geneset_path,sep='\t\t',header=None)
    go_bio_dict={}
    if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.lower().capitalize() for i in go_bio_geneset.loc[i,1].split('\t')]
    elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.upper() for i in go_bio_geneset.loc[i,1].split('\t')]
    else:
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i for i in go_bio_geneset.loc[i,1].split('\t')]
    return go_bio_dict

def get_gene_annotation(
        adata: anndata.AnnData, var_by: str = None,
        gtf: os.PathLike = None, gtf_by: str = None,
        by_func: Optional[Callable] = None
) -> None:
    r"""
    Get genomic annotation of genes by joining with a GTF file.
    It was writed by scglue, and I just copy it.

    Arguments:
        adata: Input dataset.
        var_by: Specify a column in ``adata.var`` used to merge with GTF attributes, 
            otherwise ``adata.var_names`` is used by default.
        gtf: Path to the GTF file.
        gtf_by: Specify a field in the GTF attributes used to merge with ``adata.var``,
            e.g. "gene_id", "gene_name".
        by_func: Specify an element-wise function used to transform merging fields,
            e.g. removing suffix in gene IDs.

    Note:
        The genomic locations are converted to 0-based as specified
        in bed format rather than 1-based as specified in GTF format.

    """
    if gtf is None:
        raise ValueError("Missing required argument `gtf`!")
    if gtf_by is None:
        raise ValueError("Missing required argument `gtf_by`!")
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = read_gtf(gtf).query("feature == 'gene'").split_attribute()
    if by_func:
        by_func = np.vectorize(by_func)
        var_by = by_func(var_by)
        gtf[gtf_by] = by_func(gtf[gtf_by])  # Safe inplace modification
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_by], keep="last"
    )  # Typically, scaffolds come first, chromosomes come last
    merge_df = pd.concat([
        pd.DataFrame(gtf.to_bed(name=gtf_by)),
        pd.DataFrame(gtf).drop(columns=Gtf.COLUMNS)  # Only use the splitted attributes
    ], axis=1).set_index(gtf_by).reindex(var_by).set_index(adata.var.index)
    adata.var = adata.var.assign(**merge_df)
    
import pkg_resources

predefined_signatures = dict(
    cell_cycle_human=pkg_resources.resource_filename("omicverse", "data_files/cell_cycle_human.gmt"),
    cell_cycle_mouse=pkg_resources.resource_filename("omicverse", "data_files/cell_cycle_mouse.gmt"),
    gender_human=pkg_resources.resource_filename("omicverse", "data_files/gender_human.gmt"),
    gender_mouse=pkg_resources.resource_filename("omicverse", "data_files/gender_mouse.gmt"),
    mitochondrial_genes_human=pkg_resources.resource_filename("omicverse", "data_files/mitochondrial_genes_human.gmt"),
    mitochondrial_genes_mouse=pkg_resources.resource_filename("omicverse", "data_files/mitochondrial_genes_mouse.gmt"),
    ribosomal_genes_human=pkg_resources.resource_filename("omicverse", "data_files/ribosomal_genes_human.gmt"),
    ribosomal_genes_mouse=pkg_resources.resource_filename("omicverse", "data_files/ribosomal_genes_mouse.gmt"),
    apoptosis_human=pkg_resources.resource_filename("omicverse", "data_files/apoptosis_human.gmt"),
    apoptosis_mouse=pkg_resources.resource_filename("omicverse", "data_files/apoptosis_mouse.gmt"),
    human_lung=pkg_resources.resource_filename("omicverse", "data_files/human_lung.gmt"),
    mouse_lung=pkg_resources.resource_filename("omicverse", "data_files/mouse_lung.gmt"),
    mouse_brain=pkg_resources.resource_filename("omicverse", "data_files/mouse_brain.gmt"),
    mouse_liver=pkg_resources.resource_filename("omicverse", "data_files/mouse_liver.gmt"),
    emt_human=pkg_resources.resource_filename("omicverse", "data_files/emt_human.gmt"),
)

def load_signatures_from_file(input_file: str) -> Dict[str, List[str]]:
    signatures = {}
    with open(input_file) as fin:
        for line in fin:
            items = line.strip().split('\t')
            signatures[items[0]] = list(set(items[2:]))
    print(f"Loaded signatures from GMT file {input_file}.")
    return signatures

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Literal,
    TypeVar,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence,
)

class TestMethod(ModeEnum):  # noqa
    FISHER = "fisher"
    PERM_TEST = "perm_test"



def _mat_mat_corr_sparse(
    X: csr_matrix,
    Y: np.ndarray,
) -> np.ndarray:
    n = X.shape[1]

    X_bar = np.reshape(np.array(X.mean(axis=1)), (-1, 1))
    X_std = np.reshape(
        np.sqrt(np.array(X.power(2).mean(axis=1)) - (X_bar**2)), (-1, 1)
    )

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)

def correlation_pseudotime(
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    method: TestMethod = TestMethod.FISHER,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the correlation between rows in matrix ``X`` columns of matrix ``Y``.

    Parameters
    ----------
    X
        Array or matrix of `(M, N)` elements.
    Y
        Array of `(N, K)` elements.
    method
        Method for p-value calculation.
    n_perms
        Number of permutations if ``method='perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    kwargs
        Keyword arguments for :func:`cellrank._utils._parallelize.parallelize`.

    Returns
    -------
        Correlations, p-values, corrected p-values, lower and upper bound of 95% confidence interval.
        Each array if of shape ``(n_genes, n_lineages)``.
    """

    def perm_test_extractor(
        res: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pvals, corr_bs = zip(*res)
        pvals = np.sum(pvals, axis=0) / float(n_perms)

        corr_bs = np.concatenate(corr_bs, axis=0)
        corr_ci_low, corr_ci_high = np.quantile(corr_bs, q=ql, axis=0), np.quantile(
            corr_bs, q=qh, axis=0
        )

        return pvals, corr_ci_low, corr_ci_high

    if not (0 <= confidence_level <= 1):
        raise ValueError(
            f"Expected `confidence_level` to be in interval `[0, 1]`, found `{confidence_level}`."
        )

    n = X.shape[1]  # genes x cells
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    if issparse(X) and not isspmatrix_csr(X):
        X = csr_matrix(X)

    corr = _mat_mat_corr_sparse(X, Y) if issparse(X) else _mat_mat_corr_dense(X, Y)

    if method == TestMethod.FISHER:
        # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
        mean, se = np.arctanh(corr), 1.0 / np.sqrt(n - 3)
        z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

        z = norm.ppf(qh)
        corr_ci_low = np.tanh(mean - z * se)
        corr_ci_high = np.tanh(mean + z * se)
        pvals = 2 * norm.cdf(-np.abs(z_score))
    else:
        raise NotImplementedError(method)
    '''
    elif method == TestMethod.PERM_TEST:
        if not isinstance(n_perms, int):
            raise TypeError(
                f"Expected `n_perms` to be an integer, found `{type(n_perms).__name__}`."
            )
        if n_perms <= 0:
            raise ValueError(f"Expcted `n_perms` to be positive, found `{n_perms}`.")


        pvals, corr_ci_low, corr_ci_high = parallelize(
            _perm_test,
            np.arange(n_perms),
            as_array=False,
            unit="permutation",
            extractor=perm_test_extractor,
            **kwargs,
        )(corr, X, Y, seed=seed)
    '''
    

    return corr, pvals, corr_ci_low, corr_ci_high

def _np_apply_along_axis(func1d, axis: int, arr: np.ndarray) -> np.ndarray:
    """
    Apply a reduction function over a given axis.

    Parameters
    ----------
    func1d
        Reduction function that operates only on 1 dimension.
    axis
        Axis over which to apply the reduction.
    arr
        The array to be reduced.

    Returns
    -------
    The reduced array.
    """

    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
        return result

    result = np.empty(arr.shape[0])
    for i in range(len(result)):
        result[i] = func1d(arr[i, :])

    return result

def np_mean(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.mean, axis, array)

def np_std(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.std, axis, array)

def _mat_mat_corr_dense(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    #from cellrank.kernels._utils import np_std, np_mean

    n = X.shape[1]

    X_bar = np.reshape(np_mean(X, axis=1), (-1, 1))
    X_std = np.reshape(np_std(X, axis=1), (-1, 1))

    y_bar = np.reshape(np_mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np_std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _perm_test(
    ixs: np.ndarray,
    corr: np.ndarray,
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    seed: Optional[int] = None,
    queue=None,
) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(None if seed is None else seed + ixs[0])
    cell_ixs = np.arange(X.shape[1])
    pvals = np.zeros_like(corr, dtype=np.float64)
    corr_bs = np.zeros((len(ixs), X.shape[0], Y.shape[1]))  # perms x genes x lineages

    mmc = _mat_mat_corr_sparse if issparse(X) else _mat_mat_corr_dense

    for i, _ in enumerate(ixs):
        rs.shuffle(cell_ixs)
        corr_i = mmc(X, Y[cell_ixs, :])
        pvals += np.abs(corr_i) >= np.abs(corr)

        bootstrap_ixs = rs.choice(cell_ixs, replace=True, size=len(cell_ixs))
        corr_bs[i, :, :] = mmc(X[:, bootstrap_ixs], Y[bootstrap_ixs, :])

        if queue is not None:
            queue.put(1)

    if queue is not None:
        queue.put(None)

    return pvals, corr_bs

def anndata_sparse(adata):
    """
    Set adata.X to csr_matrix

    Arguments:
        adata: AnnData

    Returns:
        adata: AnnData

    """

    from scipy.sparse import csr_matrix
    x = csr_matrix(adata.X.copy())
    adata.X=x
    return adata

def store_layers(adata,layers='counts'):
    """
    Store the X of adata in adata.uns['layers_{}'.format(layers)]

    Arguments:
        adata: AnnData
        layers: the layers name to store, default 'counts'
    """


    if issparse(adata.X) and not isspmatrix_csr(adata.X):
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif issparse(adata.X):
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in {}'.format(layers))
def retrieve_layers(adata,layers='counts'):
    """
    Retrieve the X of adata from adata.uns['layers_{}'.format(layers)]

    Arguments:
        adata: AnnData
        layers: the layers name to retrieve, default 'counts'
    
    """

    adata_test=adata.uns['layers_{}'.format(layers)].copy()
    adata_test=adata_test[adata.obs.index,adata.var.index]
    
    if issparse(adata.X) and not isspmatrix_csr(adata.X):
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif issparse(adata.X):
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in raw')
    adata.X=adata_test.X.copy()
    print('......The layers {} of adata have been retreved'.format(layers))
    del adata_test
