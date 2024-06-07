from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Iterable, Union, Optional



### Refer to: https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/_compat.py
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass

### Refer to Scanpy       
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices


### Import from Scanpy
from scipy.sparse import issparse

### Import from Scanpy
def select_groups(adata, groups_order_subset='all', key='groups'):
    """Get subset of groups in adata.obs[key]."""
    groups_order = adata.obs[key].cat.categories
    if key + '_masks' in adata.uns:
        groups_masks = adata.uns[key + '_masks']
    else:
        groups_masks = np.zeros(
            (len(adata.obs[key].cat.categories), adata.obs[key].values.size), dtype=bool
        )
        for iname, name in enumerate(adata.obs[key].cat.categories):
            # if the name is not found, fallback to index retrieval
            if adata.obs[key].cat.categories[iname] in adata.obs[key].values:
                mask = adata.obs[key].cat.categories[iname] == adata.obs[key].values
            else:
                mask = str(iname) == adata.obs[key].values
            groups_masks[iname] = mask
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != 'all':
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(
                np.where(adata.obs[key].cat.categories.values == name)[0][0]
            )
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = np.where(
                np.in1d(
                    np.arange(len(adata.obs[key].cat.categories)).astype(str),
                    np.array(groups_order_subset),
                )
            )[0]
        if len(groups_ids) == 0:
            logg.debug(
                f'{np.array(groups_order_subset)} invalid! specify valid '
                f'groups_order (or indices) from {adata.obs[key].cat.categories}',
            )
            from sys import exit

            exit(0)
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = adata.obs[key].cat.categories[groups_ids].values
    else:
        groups_order_subset = groups_order.values
    return groups_order_subset, groups_masks


### Import from Scanpy   
import numpy as np
from scipy import sparse
import numba




def _get_mean_var(X, *, axis=0):
    if sparse.issparse(X):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean ** 2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def sparse_mean_variance_axis(mtx: sparse.spmatrix, axis: int):
    """
    This code and internal functions are based on sklearns
    `sparsefuncs.mean_variance_axis`.
    Modifications:
    * allow deciding on the output type, which can increase accuracy when calculating the mean and variance of 32bit floats.
    * This doesn't currently implement support for null values, but could.
    * Uses numba not cython
    """
    assert axis in (0, 1)
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError("This function only works on sparse csr and csc matrices")
    if axis == ax_minor:
        return sparse_mean_var_major_axis(
            mtx.data, mtx.indices, mtx.indptr, *shape, np.float64
        )
    else:
        return sparse_mean_var_minor_axis(mtx.data, mtx.indices, *shape, np.float64)


@numba.njit(cache=True)
def sparse_mean_var_minor_axis(data, indices, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse matrix for the minor axis.
    Given arrays for a csr matrix, returns the means and variances for each
    column back.
    """
    non_zero = indices.shape[0]

    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)

    for i in range(non_zero):
        col_ind = indices[i]
        means[col_ind] += data[i]

    for i in range(minor_len):
        means[i] /= major_len

    for i in range(non_zero):
        col_ind = indices[i]
        diff = data[i] - means[col_ind]
        variances[col_ind] += diff * diff
        counts[col_ind] += 1

    for i in range(minor_len):
        variances[i] += (major_len - counts[i]) * means[i] ** 2
        variances[i] /= major_len

    return means, variances


@numba.njit(cache=True)
def sparse_mean_var_major_axis(data, indices, indptr, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse array for the major axis.
    Given arrays for a csr matrix, returns the means and variances for each
    row back.
    """
    means = np.zeros(major_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    for i in range(major_len):
        startptr = indptr[i]
        endptr = indptr[i + 1]
        counts = endptr - startptr

        for j in range(startptr, endptr):
            means[i] += data[j]
        means[i] /= minor_len

        for j in range(startptr, endptr):
            diff = data[j] - means[i]
            variances[i] += diff * diff

        variances[i] += (minor_len - counts) * means[i] ** 2
        variances[i] /= minor_len

    return means, variances  

### Import from Scanpy
class _RankGenes:
    def __init__(
        self,
        adata,
        groups,
        groupby,
        reference='rest',
        use_raw=True,
        layer=None,
        comp_pts=False,
    ):

        if 'log1p' in adata.uns_keys() and adata.uns['log1p']['base'] is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(adata.uns['log1p']['base']))
        else:
            self.expm1_func = np.expm1

        self.groups_order, self.groups_masks = select_groups(
            adata, groups, groupby
        )

        # Singlet groups cause division by zero errors
        invalid_groups_selected = set(self.groups_order) & set(
            adata.obs[groupby].value_counts().loc[lambda x: x < 2].index
        )

        if len(invalid_groups_selected) > 0:
            raise ValueError(
                "Could not calculate statistics for groups {} since they only "
                "contain one sample.".format(', '.join(invalid_groups_selected))
            )

        adata_comp = adata
        if layer is not None:
            if use_raw:
                raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
            X = adata_comp.layers[layer]
        else:
            if use_raw and adata.raw is not None:
                adata_comp = adata.raw
            X = adata_comp.X

        # for correct getnnz calculation
        if issparse(X):
            X.eliminate_zeros()

        self.X = X
        self.var_names = adata_comp.var_names

        self.ireference = None
        if reference != 'rest':
            self.ireference = np.where(self.groups_order == reference)[0][0]

        self.means = None
        self.vars = None

        self.means_rest = None
        self.vars_rest = None

        self.comp_pts = comp_pts
        self.pts = None
        self.pts_rest = None

        self.stats = None

        # for logreg only
        self.grouping_mask = adata.obs[groupby].isin(self.groups_order)
        self.grouping = adata.obs.loc[self.grouping_mask, groupby]

    def _basic_stats(self):
        n_genes = self.X.shape[1]
        n_groups = self.groups_masks.shape[0]

        self.means = np.zeros((n_groups, n_genes))
        self.vars = np.zeros((n_groups, n_genes))
        self.pts = np.zeros((n_groups, n_genes)) if self.comp_pts else None

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes))
            self.vars_rest = np.zeros((n_groups, n_genes))
            self.pts_rest = np.zeros((n_groups, n_genes)) if self.comp_pts else None
        else:
            mask_rest = self.groups_masks[self.ireference]
            X_rest = self.X[mask_rest]
            self.means[self.ireference], self.vars[self.ireference] = _get_mean_var(
                X_rest
            )
            # deleting the next line causes a memory leak for some reason
            del X_rest

        if issparse(self.X):
            get_nonzeros = lambda X: X.getnnz(axis=0)
        else:
            get_nonzeros = lambda X: np.count_nonzero(X, axis=0)

        for imask, mask in enumerate(self.groups_masks):
            X_mask = self.X[mask]

            if self.comp_pts:
                self.pts[imask] = get_nonzeros(X_mask) / X_mask.shape[0]

            if self.ireference is not None and imask == self.ireference:
                continue

            self.means[imask], self.vars[imask] = _get_mean_var(X_mask)

            if self.ireference is None:
                mask_rest = ~mask
                X_rest = self.X[mask_rest]
                self.means_rest[imask], self.vars_rest[imask] = _get_mean_var(X_rest)
                # this can be costly for sparse data
                if self.comp_pts:
                    self.pts_rest[imask] = get_nonzeros(X_rest) / X_rest.shape[0]
                # deleting the next line causes a memory leak for some reason
                del X_rest

        
def cosg(
    adata,
    groupby='CellTypes',
    groups: Union[Literal['all'], Iterable[str]] = 'all',

    mu=1,
    remove_lowly_expressed:bool=False,
    expressed_pct:Optional[float] = 0.1,

    n_genes_user:int =50,
    key_added: Optional[str] = None,
    calculate_logfoldchanges: bool = True,
    use_raw: bool = True,
    layer: Optional[str] = None,
    reference: str = 'rest',    

    copy:bool=False
):
    """
    Marker gene identification for single-cell sequencing data using COSG.
    
    Arguments:
        adata: Annotated data matrix. Note: input parameters are similar to the parameters used for scanpy's rank_genes_groups() function.
        groupby: The key of the cell groups in .obs, the default value is set to 'CellTypes'.
        groups: Subset of cell groups, e.g. ['g1', 'g2', 'g3'], to which comparison shall be restricted. The default value is 'all', and all groups will be compared.
        mu: The penalty restricting marker genes expressing in non-target cell groups. Larger value represents more strict restrictions. mu should be >= 0, and by default, mu = 1.
        remove_lowly_expressed: If True, genes that express a percentage of target cells smaller than a specific value (expressed_pct) are not considered as marker genes for the target cells. The default value is False.
        expressed_pct: When remove_lowly_expressed is set to True, genes that express a percentage of target cells smaller than a specific value (expressed_pct) are not considered as marker genes for the target cells. The default value for expressed_pct is 0.1 (10%).
        n_genes_user: The number of genes that appear in the returned tables. The default value is 50.
        key_added: The key in adata.uns information is saved to.
        calculate_logfoldchanges: Calculate logfoldchanges.
        use_raw: Use raw attribute of adata if present.
        layer: Key from adata.layers whose value will be used to perform tests on.
        reference: If 'rest', compare each group to the union of the rest of the group. If a group identifier, compare with respect to this group.

    Returns:
        names : structured np.ndarray (.uns['rank_genes_groups'])
                Structured array to be indexed by group id storing the gene names. Ordered according to scores.
        scores : structured np.ndarray (.uns['rank_genes_groups'])
                Structured array to be indexed by group id storing COSG scores for each gene for each group. Ordered according to scores.

    Notes:
        Contact: Min Dai, daimin@zju.edu.cn

    Examples:
        >>> import cosg as cosg
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc68k_reduced()
        >>> cosg.cosg(adata, key_added='cosg', groupby='bulk_labels')
        >>> sc.pl.rank_genes_groups(adata, key='cosg')

    """
    
    adata = adata.copy() if copy else adata
        
    if layer is not None:
        if use_raw:
            raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
        cellxgene = adata.layers[layer]
    else:
        if use_raw and adata.raw is not None:
             cellxgene = adata.raw.X
        cellxgene = adata.X
    
    
    ### Refer to scanpy's framework
    # https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/tools/_rank_genes_groups.py#L559
    if key_added is None:
        key_added = 'rank_genes_groups'
    adata.uns[key_added] = {}
    adata.uns[key_added]['params'] = dict(
    groupby=groupby,
    reference=reference,
    groups=groups,
    method='cosg',
    use_raw=use_raw,
    layer=layer,
    )
    
    ### Refer to: https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/tools/_rank_genes_groups.py#L543
    if groups == 'all':
        ### group lable for each cell
        group_info=adata.obs[groupby]
    elif isinstance(groups, (str, int)):
        raise ValueError('Specify a sequence of groups')
    else:
        cells_selected=adata.obs[groupby].isin(groups)
        cells_selected=cells_selected.values
        if sparse.issparse(cellxgene):
            cellxgene=cellxgene[cells_selected]
        else:
            cellxgene=cellxgene[cells_selected,:]
            
            

        ### group lable for each cell
        group_info=adata.obs[groupby].copy()
        group_info=group_info[cells_selected]
        

    
    groups_order=np.unique(group_info)
    n_cluster=len(groups_order)
    
    n_cell=cellxgene.shape[0]
    cluster_mat=np.zeros(shape=(n_cluster,n_cell))

    ### To further restrict expression in other clusters, can think about a better way, such as taking the cluster similarities into consideration
    order_i=0
    for group_i in groups_order:    
        idx_i=group_info==group_i 
        cluster_mat[order_i,:][idx_i]=1
        order_i=order_i+1
    
    if sparse.issparse(cellxgene):
        ### Convert to sparse matrix
        from scipy.sparse import csr_matrix
        cluster_mat=csr_matrix(cluster_mat)

        from sklearn.metrics.pairwise import cosine_similarity

        ### the dimension is: Gene x lambda
        cosine_sim=cosine_similarity(X=cellxgene.T,Y=cluster_mat,dense_output=False) 

        pos_nonzero=cosine_sim.nonzero()
        genexlambda=cosine_sim.multiply(cosine_sim)

        e_power2_sum=genexlambda.sum(axis=1)


        if mu==1:
            genexlambda[pos_nonzero]=genexlambda[pos_nonzero]/(np.repeat(e_power2_sum,genexlambda.shape[1],axis=1)[pos_nonzero])
                                                    
        else:
            genexlambda[pos_nonzero]=genexlambda[pos_nonzero]/((1-mu)*genexlambda[pos_nonzero]+
                                                     mu*(
                                                        np.repeat(e_power2_sum,genexlambda.shape[1],axis=1)[pos_nonzero])
                                                    )
        genexlambda[pos_nonzero]=np.multiply(genexlambda[pos_nonzero],cosine_sim[pos_nonzero])
    
    ### If the cellxgene is not a sparse matrix
    else:
         ## Not using sparse matrix
        from sklearn.metrics.pairwise import cosine_similarity    
        cosine_sim=cosine_similarity(X=cellxgene.T,Y=cluster_mat,dense_output=True) 

        pos_nonzero=cosine_sim!=0
        e_power2=np.multiply(cosine_sim,cosine_sim)
        e_power2_sum=np.sum(e_power2,axis=1)
        e_power2[pos_nonzero]=np.true_divide(e_power2[pos_nonzero],(1-mu)*e_power2[pos_nonzero]+mu*(np.dot(e_power2_sum.reshape(e_power2_sum.shape[0],1),np.repeat(1,e_power2.shape[1]).reshape(1,e_power2.shape[1]))[pos_nonzero]))
        e_power2[pos_nonzero]=np.multiply(e_power2[pos_nonzero],cosine_sim[pos_nonzero])
        genexlambda=e_power2

    ### Refer to scanpy
    rank_stats=None
    
    ### Whether to calculate logfoldchanges, because this is required in scanpy 1.8
    if calculate_logfoldchanges:
        ### Calculate basic stats
        ### Refer to Scanpy
        # for clarity, rename variable
        if groups == 'all':
            groups_order2 = 'all'
        elif isinstance(groups, (str, int)):
            raise ValueError('Specify a sequence of groups')
        else:
            groups_order2 = list(groups)
            if isinstance(groups_order2[0], int):
                groups_order2 = [str(n) for n in groups_order2]
            if reference != 'rest' and reference not in set(groups_order2):
                groups_order2 += [reference]
        if reference != 'rest' and reference not in adata.obs[groupby].cat.categories:
            cats = adata.obs[groupby].cat.categories.tolist()
            raise ValueError(
                f'reference = {reference} needs to be one of groupby = {cats}.'
            )
        pts=False
        anndata_obj = _RankGenes(adata, groups_order2, groupby, reference, use_raw, layer, pts)
        anndata_obj._basic_stats()
    
    
    ### Refer to Scanpy
    # for correct getnnz calculation
    ### get non-zeros for columns
    if sparse.issparse(cellxgene):
        cellxgene.eliminate_zeros()
    if sparse.issparse(cellxgene):
        get_nonzeros = lambda X: X.getnnz(axis=0)
    else:
        get_nonzeros = lambda X: np.count_nonzero(X, axis=0)
    
    order_i=0
    for group_i in groups_order:    
        idx_i=group_info==group_i 

        ### Convert to numpy array
        idx_i=idx_i.values

        ## Compare the most ideal case to the worst case
        if sparse.issparse(cellxgene):
            scores=genexlambda[:,order_i].A[:,0]
        else:
            scores=genexlambda[:,order_i]

        
        ### Mask these genes expressed in less than 3 cells in the cluster of interest
        if remove_lowly_expressed:
            n_cells_expressed=get_nonzeros(cellxgene[idx_i])
            n_cells_i=np.sum(idx_i)
            scores[n_cells_expressed<n_cells_i*expressed_pct]= -1

        global_indices = _select_top_n(scores, n_genes_user)

        if rank_stats is None:
            idx = pd.MultiIndex.from_tuples([(group_i,'names')])
            rank_stats = pd.DataFrame(columns=idx)
        rank_stats[group_i, 'names'] = adata.var_names[global_indices]
        rank_stats[group_i, 'scores'] = scores[global_indices]
        
        if calculate_logfoldchanges:
            group_index=np.where(anndata_obj.groups_order==group_i)[0][0]
            if anndata_obj.means is not None:
                mean_group = anndata_obj.means[group_index]
                if anndata_obj.ireference is None:
                    mean_rest = anndata_obj.means_rest[group_index]
                else:
                    mean_rest = anndata_obj.means[anndata_obj.ireference]
                foldchanges = (anndata_obj.expm1_func(mean_group) + 1e-9) / (
                    anndata_obj.expm1_func(mean_rest) + 1e-9
                )  # add small value to remove 0's
                rank_stats[group_i, 'logfoldchanges'] = np.log2(
                    foldchanges[global_indices]
                )
    
        order_i=order_i+1
            
    ## Refer to scanpy
    if calculate_logfoldchanges:
        dtypes = {
        'names': 'O',
        'scores': 'float32',
        'logfoldchanges': 'float32',
        }
    else:
        dtypes = {
        'names': 'O',
        'scores': 'float32',
        }
    ###
    rank_stats.columns = rank_stats.columns.swaplevel()
    for col in rank_stats.columns.levels[0]:
        adata.uns[key_added][col]=rank_stats[col].to_records(
    index=False, column_dtypes=dtypes[col]
    )
    adata.uns[key_added]['pvals']=adata.uns['rank_genes_groups']['pvals']
    adata.uns[key_added]['pvals_adj']=adata.uns['rank_genes_groups']['pvals_adj']
        
    
        
    print('**finished identifying marker genes by COSG**')
        
    ### Return the result
    return adata if copy else None