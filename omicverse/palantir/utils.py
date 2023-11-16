from typing import Iterable, Union, Tuple, List
from warnings import warn
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
import gc

from scipy.sparse import csr_matrix, find, issparse, hstack
from scipy.sparse.linalg import eigs
#import mellon
import scanpy as sc

from .core import run_palantir

from .validation import _validate_obsm_key


class CellNotFoundException(Exception):
    """Exception raised when no valid component is found for the provided cell type."""

    pass


def run_pca(
    data: Union[pd.DataFrame, sc.AnnData],
    n_components: int = 300,
    use_hvg: bool = True,
    pca_key: str = "X_pca",
) -> Union[Tuple[pd.DataFrame, np.array], None]:
    """
    Run PCA on the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Dataframe of cells X genes or sc.AnnData object.
        Typically multi-scale space diffusion components.
    n_components : int, optional
        Number of principal components. Default is 300.
    use_hvg : bool, optional
        Whether to use highly variable genes only for PCA. Default is True.
    pca_key : str, optional
        Key to store the PCA projections in obsm of data if it is a sc.AnnData object. Default is 'X_pca'.

    Returns
    -------
    Union[Tuple[pd.DataFrame, np.array], None]
        Tuple of PCA projections of the data and the explained variance.
        If sc.AnnData is passed as data, the results are also written to the input object and None is returned.
    """
    if isinstance(data, pd.DataFrame):
        ad = sc.AnnData(data.values)
    else:
        ad = data
        if pca_key != "X_pca":
            old_pca = ad.obsm.get("X_pca", None)
        else:
            old_pca = None

    # Run PCA
    if not use_hvg:
        n_comps = n_components
    else:
        l_n_comps = min(1000, ad.n_obs - 1, ad.n_vars - 1)
        sc.pp.pca(ad, n_comps=l_n_comps, use_highly_variable=True, zero_center=False)
        try:
            n_comps = np.where(np.cumsum(ad.uns["pca"]["variance_ratio"]) > 0.85)[0][0]
        except IndexError:
            n_comps = n_components

    # Rerun with selection number of components
    n_comps = min(n_comps, ad.n_obs - 1, ad.n_vars - 1)
    sc.pp.pca(ad, n_comps=n_comps, use_highly_variable=use_hvg, zero_center=False)

    if isinstance(data, sc.AnnData):
        data.obsm[pca_key] = ad.obsm["X_pca"]
        if pca_key != "X_pca":
            del data.obsm["X_pca"]
        elif old_pca is not None:
            data.obsm["X_pca"] = old_pca

    pca_projections = pd.DataFrame(ad.obsm[pca_key], index=ad.obs_names)
    return pca_projections, ad.uns["pca"]["variance_ratio"]


def run_low_density_variability(
    ad: sc.AnnData,
    cell_mask: Union[str, np.ndarray, List[str], pd.Series, pd.Index] = "branch_masks",
    density_key: str = "mellon_log_density",
    localvar_key: str = "local_variability",
    score_key: str = "low_density_gene_variability",
) -> np.ndarray:
    """
    Compute the scores aggregated local gene variability in low-density cell-state transitions.

    Parameters
    ----------
    ad : sc.AnnData
        AnnData object containing the gene expression data and pseudotime.
    cell_mask : str, np.ndarray, list of str, pd.Series, pd.Index, optional
        Key to access the mask matrix in the obsm or obs attributes of the AnnData object.
        If cell_mask is a numpy array with shape (ad.n_obs, ), it is used directly.
        If cell_mask is a list of cell names, a pd.Series, or a pd.Index,
        it is used to create a boolean mask of the same length as ad.n_obs.
        Default is 'branch_masks'.
    density_key : str, optional
        Key to access the density values in the obs attribute of the AnnData object.
        Default is 'mellon_log_density'.
    localvar_key : str, optional
        Key to access local variability matrix in the layers of the AnnData object.
        Default is 'local_variability'.
    score_key : str, optional
        Prefix of the key under which the computed scores are stored in the var attribute of the
        AnnData object. Actual keys are '{score_key}_{branch_name}' if cell_mask points to an ad.obsm.
        Default is 'low_density_gene_variability'.

    Returns
    -------
    low_density_scores : np.ndarray
        A numpy array of scores for each gene.

    Raises
    ------
    ValueError
        If any of the provided keys are not found in the appropriate fields of the AnnData object.
    """
    if density_key not in ad.obs.columns:
        raise ValueError(f"'{density_key}' not found in ad.obs.")
    dip_weight = np.exp(-ad.obs[density_key].values)
    if localvar_key not in ad.layers.keys():
        raise ValueError(f"'{localvar_key}' not found in ad.layers.")
    local_var = ad.layers[localvar_key]
    if isinstance(cell_mask, str):
        if cell_mask in ad.obsm.keys():
            masks, branch_names = _validate_obsm_key(ad, cell_mask, as_df=False)
            if masks.ndim == 1:
                masks = masks[:, None]
        elif cell_mask in ad.obs.columns:
            branch_names = [
                "_" + cell_mask,
            ]
            masks = ad.obs[cell_mask].values[:, None]
        else:
            raise ValueError(f"'{cell_mask}' not found in ad.obsm or ad.obs.columns.")
    elif isinstance(cell_mask, np.ndarray) and cell_mask.shape == (ad.n_obs,):
        masks = cell_mask[:, None]
        branch_names = [
            "",
        ]
    elif isinstance(cell_mask, (list, pd.Series, pd.Index)):
        masks = ad.obs_names.isin(cell_mask)[:, None]
        branch_names = [
            "",
        ]
    else:
        raise ValueError(
            "cell_mask must be either a string key, a numpy array with shape "
            "(ad.n_obs, ), a list of cell names, a pd.Series, or a pd.Index."
        )

    out_columns = list()
    for i, branch in enumerate(branch_names):
        idx = masks[:, i]
        colname = score_key + "_" + branch
        ad.var[colname] = np.mean(
            local_var[idx, :] * dip_weight[idx, None],
            axis=0,
        )
        out_columns.append(colname)

    return ad.var[out_columns].values


def run_density(
    ad: sc.AnnData,
    repr_key: str = "DM_EigenVectors",
    density_key: str = "mellon_log_density",
    **kwargs,
) -> np.ndarray:
    """
    Compute cell-state density with Mellon.

    This function uses the Mellon algorithm to compute the density of cell states, which is stored in the obs attribute of the
    AnnData object. The function returns the computed density. If 'DM_EigenVectors' is not found in the AnnData object,
    an error is raised suggesting the user to run the function `palantir.utils.run_diffusion_maps(ad)`.

    Additionally, the density prediction model is serialized and stored in the `.uns` attribute of the AnnData object
    under the key `'{density_key}_predictor'`. This can be deserialized and used for prediction by using the `mellon.Predictor.from_dict()`
    method.

    Parameters
    ----------
    ad : sc.AnnData
        AnnData object containing the gene expression data and pseudotime.
    repr_key : str, optional
        Key to retrieve cell-state representation from the sc.AnnData object. Default is 'DM_EigenVectors'.
    density_key : str, optional
        Key under which the computed density values are stored in the obs of the AnnData object.
        Default is 'mellon_log_density'.
    **kwargs : dict
        Additional keyword arguments to be passed to mellon.DensityEstimator.

    Returns
    -------
    log_density : np.ndarray
        A numpy array of log density values computed for each cell.

    Raises
    ------
    ValueError
        If `repr_key` is not found in `ad.obsm`.
    """
    if repr_key not in ad.obsm:
        raise ValueError(
            f"'{repr_key}' not found in ad.obsm. "
            "Run `palantir.utils.run_diffusion_maps(ad)` to "
            "compute diffusion map eigenvectors."
        )
    X = ad.obsm[repr_key]

    # Set the default arguments for mellon.DensityEstimator
    mellon_args = dict()
    mellon_args.update(kwargs)
    import mellon
    dest = mellon.DensityEstimator(**mellon_args)
    log_density = np.asarray(dest.fit_predict(X))

    ad.obs[density_key] = log_density
    ad.obs[density_key + "_clipped"] = np.clip(
        log_density, *np.quantile(log_density, [0.01, 1])
    )
    ad.uns[density_key + "_predictor"] = dest.predict.to_dict()

    return log_density


def run_density_evaluation(
    in_ad: sc.AnnData,
    out_ad: sc.AnnData,
    predictor_key: str = "mellon_log_density_predictor",
    repr_key: str = "DM_EigenVectors",
    density_key: str = "cross_log_density",
    **kwargs,
) -> np.ndarray:
    """
    Evaluates the density function of `in_ad.uns[predictor_key]` on the representations of `out_ad.obsm[repr_key]`.

    Parameters
    ----------
    in_ad : sc.AnnData
        AnnData object containing the gene expression data and the serialized predictor.
    out_ad : sc.AnnData
        AnnData object containing the gene expression data and representations to be evaluated.
    predictor_key : str, optional
        Key to access the predictor in the uns of the `in_ad` AnnData object. Default is 'mellon_log_density_predictor'.
    repr_key : str, optional
        Key to access representations in the obsm of the `out_ad` AnnData object. Default is 'DM_EigenVectors'.
    density_key : str, optional
        Key under which the computed density values are stored in the obs of the `out_ad` AnnData object.
        Default is 'cross_log_density'.
    **kwargs : dict
        Additional keyword arguments, unused in this function.

    Returns
    -------
    log_density : np.ndarray
        A numpy array of log density values computed for each cell in `out_ad`.

    Raises
    ------
    ValueError
        If `repr_key` is not found in `out_ad.obsm` or `predictor_key` is not found in `in_ad.uns`.
    """
    if repr_key not in out_ad.obsm:
        raise ValueError(
            f"'{repr_key}' not found in out_ad.obsm. "
            "Run `palantir.utils.run_diffusion_maps(ad)` to "
            "compute diffusion map eigenvectors."
        )
    if predictor_key not in in_ad.uns:
        raise ValueError(
            f"'{predictor_key}' not found in in_ad.uns. "
            "Run `run_density(ad)` to compute the density estimator."
        )

    X = out_ad.obsm[repr_key]
    import mellon
    predictor = mellon.Predictor.from_dict(in_ad.uns[predictor_key])
    log_density = predictor(X)
    out_ad.obs[density_key] = log_density
    out_ad.obs[density_key + "_clipped"] = np.clip(
        log_density, *np.quantile(log_density, [0.01, 1])
    )

    return log_density


def compute_kernel(
    data: Union[pd.DataFrame, sc.AnnData],
    knn: int = 30,
    alpha: float = 0,
    pca_key: str = "X_pca",
    kernel_key: str = "DM_Kernel",
) -> csr_matrix:
    """
    Compute the adaptive anisotropic diffusion kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Data points (rows) in a feature space (columns) for pd.DataFrame.
        For sc.AnnData, it uses the .X attribute.
    knn : int
        Number of nearest neighbors for adaptive kernel calculation. Default is 30.
    alpha : float
        Normalization parameter for the diffusion operator. Default is 0.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    kernel_key : str, optional
        Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.

    Returns
    -------
    csr_matrix
        Computed kernel matrix.
    """

    # If the input is sc.AnnData, convert it to a DataFrame
    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    N = data_df.shape[0]
    temp = sc.AnnData(data_df.values)
    sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
    kNN = temp.obsp["distances"]

    adaptive_k = int(np.floor(knn / 3))
    adaptive_std = np.zeros(N)
    for i in np.arange(N):
        adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
            adaptive_k - 1
        ]

    x, y, dists = find(kNN)
    dists /= adaptive_std[x]
    W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

    kernel = W + W.T

    if alpha > 0:
        D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)

    if isinstance(data, sc.AnnData):
        data.obsp[kernel_key] = kernel

    return kernel


def diffusion_maps_from_kernel(
    kernel: csr_matrix, n_components: int = 10, seed: Union[int, None] = 0
):
    """
    Compute the diffusion map given a kernel matrix.

    Parameters
    ----------
    kernel : csr_matrix
        Precomputed kernel matrix.
    n_components : int
        Number of diffusion components to compute. Default is 10.
    seed : Union[int, None]
        Seed for random initialization. Default is 0.

    Returns
    -------
    dict
        T-matrix (T), Diffusion components (EigenVectors) and corresponding eigenvalues (EigenValues).
    """
    N = kernel.shape[0]
    D = np.ravel(kernel.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

    np.random.seed(seed)
    v0 = np.random.rand(min(T.shape))
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000, v0=v0)

    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    return {"T": T, "EigenVectors": pd.DataFrame(V), "EigenValues": pd.Series(D)}


def run_diffusion_maps(
    data: Union[pd.DataFrame, sc.AnnData],
    n_components: int = 10,
    knn: int = 30,
    alpha: float = 0,
    seed: Union[int, None] = 0,
    pca_key: str = "X_pca",
    kernel_key: str = "DM_Kernel",
    sim_key: str = "DM_Similarity",
    eigval_key: str = "DM_EigenValues",
    eigvec_key: str = "DM_EigenVectors",
):
    """
    Run Diffusion maps using the adaptive anisotropic kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        PCA projections of the data or adjacency matrix.
        If sc.AnnData is passed, its obsm[pca_key] is used and the result is written to
        its obsp[kernel_key], obsm[eigvec_key], and uns[eigval_key].
    n_components : int, optional
        Number of diffusion components. Default is 10.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    alpha : float, optional
        Normalization parameter for the diffusion operator. Default is 0.
    seed : Union[int, None], optional
        Numpy random seed, randomized if None, set to an arbitrary integer for reproducibility.
        Default is 0.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    kernel_key : str, optional
        Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.
    sim_key : str, optional
        Key to store the similarity in obsp of data if it is a sc.AnnData object. Default is 'DM_Similarity'.
    eigval_key : str, optional
        Key to store the EigenValues in uns of data if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to store the EigenVectors in obsm of data if it is a sc.AnnData object. Default is 'DM_EigenVectors'.

    Returns
    -------
    dict
        Diffusion components, corresponding eigen values and the diffusion operator.
        If sc.AnnData is passed as data, these results are also written to the input object
        and returned.
    """

    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    if not isinstance(data_df, pd.DataFrame) and not issparse(data_df):
        raise ValueError("'data_df' should be a pd.DataFrame or sc.AnnData")

    if not issparse(data_df):
        kernel = compute_kernel(data_df, knn, alpha)
    else:
        warn(
            "'data' is a sparse matrix and will be interpreted as kernel. "
            "To avoid this warning compute diffusion maps from a precompued kernel using "
            "palantir.utils.diffusion_maps_from_kernel()."
        )
        kernel = data_df

    res = diffusion_maps_from_kernel(kernel, n_components, seed)

    res["kernel"] = kernel
    if not issparse(data_df):
        res["EigenVectors"].index = data_df.index

    if isinstance(data, sc.AnnData):
        data.obsp[kernel_key] = res["kernel"]
        data.obsp[sim_key] = res["T"]
        data.obsm[eigvec_key] = res["EigenVectors"].values
        data.uns[eigval_key] = res["EigenValues"].values

    return res


def _dot_helper_func(x, y):
    return x.dot(y)


def _local_var_helper(expressions, distances):
    if hasattr(expressions, "todense"):

        def cast(x):
            return x.todense()

        issparse = True
    else:

        def cast(x):
            return x

        issparse = False
    for cell in range(expressions.shape[0]):
        neighbors = distances.getrow(cell).indices if issparse else slice(None)
        try:
            neighbor_expression = cast(expressions[neighbors, :])
            cell_expression = cast(expressions[cell, :])
            expr_deltas = np.array(neighbor_expression - cell_expression)
        except ValueError:
            raise ValueError(f"This cell caused the error: {cell}")
        expr_distance = np.sqrt(np.sum(expr_deltas**2, axis=1, keepdims=True))
        change_rate = expr_deltas / expr_distance
        yield np.max(change_rate**2, axis=0)


def run_local_variability(
    ad: sc.AnnData,
    expression_key: str = "MAGIC_imputed_data",
    distances_key: str = "distances",
    localvar_key: str = "local_variability",
) -> np.ndarray:
    """
    Compute local gene variability scores for each cell.

    This function calculates the variability in gene expression in a local neighbourhood for each cell.
    It adds the result to the layers of the given AnnData object under the specified key.

    Parameters
    ----------
    ad : sc.AnnData
        AnnData object containing the gene expression data and pseudotime.
    expression_key : str, optional
        Key to access the gene expression data in the layers of the AnnData object.
        If None, uses raw expression data in .X. Default is 'MAGIC_imputed_data'.
    distances_key : str, optional
        Key to access the distances matrix in the obsm of the AnnData object.
        Default is 'distances'.
    localvar_key : str, optional
        Key under which the computed local variability matrix is stored in the layers of the AnnData object.
        Default is 'local_variability'.

    Returns
    -------
    local_variability : np.ndarray
        A 2D numpy array of local variability scores for each gene in each cell.
    """

    if expression_key:
        if expression_key not in ad.layers:
            raise KeyError(f"'{expression_key}' not found in .layers.")
        X = ad.layers[expression_key]
    else:
        X = ad.X

    if distances_key not in ad.obsp:
        raise KeyError(f"'{distances_key}' not found in .obsp.")
    X_dists = ad.obsp[distances_key]

    local_variability = np.stack(list(_local_var_helper(X, X_dists)))

    ad.layers[localvar_key] = local_variability

    return local_variability


def run_magic_imputation(
    data: Union[np.ndarray, pd.DataFrame, sc.AnnData, csr_matrix],
    dm_res: Union[dict, None] = None,
    n_steps: int = 3,
    sim_key: str = "DM_Similarity",
    expression_key: str = None,
    imputation_key: str = "MAGIC_imputed_data",
    n_jobs: int = -1,
) -> Union[pd.DataFrame, None, csr_matrix]:
    """
    Run MAGIC imputation on the data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, sc.AnnData, csr_matrix]
        Array or DataFrame of cells X genes, sc.AnnData object, or a sparse csr_matrix.
    dm_res : Union[dict, None], optional
        Diffusion map results from run_diffusion_maps.
        If None and data is a sc.AnnData object, its obsp[kernel_key] is used. Default is None.
    n_steps : int, optional
        Number of steps in the diffusion operator. Default is 3.
    expression_key : str, optional
        Key to access the gene expression data in the layers of the AnnData object.
        If None, uses raw expression data in .X. Default is None.
    sim_key : str, optional
        Key to access the similarity in obsp of data if it is a sc.AnnData object.
        Default is 'DM_Similarity'.
    imputation_key : str, optional
        Key to store the imputed data in layers of data if it is a sc.AnnData object. Default is 'MAGIC_imputed_data'.
    n_jobs : int, optional
        Number of cores to use for parallel processing. If -1, all available cores are used. Default is -1.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, None, csr_matrix]
        Imputed data matrix. If sc.AnnData is passed as data, the result is written to its layers[imputation_key].
    """
    if isinstance(data, sc.AnnData):
        if expression_key is not None:
            if expression_key not in data.layers.keys():
                raise ValueError(
                    f"expression_key '{expression_key}' not found in .layers."
                )
            X = data.layers[expression_key]
        else:
            X = data.X
        if dm_res is None:
            T = data.obsp[sim_key]
    elif isinstance(data, pd.DataFrame):
        X = data.values
    elif issparse(data):  # assuming csr_matrix
        X = data
    else:  # assuming np.ndarray
        X = data

    if dm_res is not None:
        T = dm_res["T"]
    elif not isinstance(data, sc.AnnData):
        raise ValueError(
            "Diffusion map results (dm_res) must be provided if data is not sc.AnnData"
        )

    # Preparing the operator
    T_steps = (T**n_steps).astype(np.float32)

    # Define chunks of columns for parallel processing
    chunks = np.append(np.arange(0, X.shape[1], 100), [X.shape[1]])

    # Run the dot product in parallel on chunks
    res = Parallel(n_jobs=n_jobs)(
        delayed(_dot_helper_func)(T_steps, X[:, chunks[i - 1] : chunks[i]])
        for i in range(1, len(chunks))
    )

    # Stack the results together
    if issparse(X):
        imputed_data = hstack(res).todense()
    else:
        imputed_data = np.hstack(res)

    # Set small values to zero
    imputed_data[imputed_data < 1e-2] = 0

    # Clean up
    gc.collect()

    if isinstance(data, sc.AnnData):
        data.layers[imputation_key] = np.asarray(imputed_data)

    if isinstance(data, pd.DataFrame):
        imputed_data = pd.DataFrame(
            imputed_data, index=data.index, columns=data.columns
        )

    return imputed_data


def determine_multiscale_space(
    dm_res: Union[dict, sc.AnnData],
    n_eigs: Union[int, None] = None,
    eigval_key: str = "DM_EigenValues",
    eigvec_key: str = "DM_EigenVectors",
    out_key: str = "DM_EigenVectors_multiscaled",
) -> Union[pd.DataFrame, None]:
    """
    Determine the multi-scale space of the data.

    Parameters
    ----------
    dm_res : Union[dict, sc.AnnData]
        Diffusion map results from run_diffusion_maps.
        If sc.AnnData is passed, its uns[eigval_key] and obsm[eigvec_key] are used.
    n_eigs : Union[int, None], optional
        Number of eigen vectors to use. If None is specified, the number
        of eigen vectors will be determined using the eigen gap. Default is None.
    eigval_key : str, optional
        Key to retrieve EigenValues from dm_res if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to retrieve EigenVectors from dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors'.
    out_key : str, optional
        Key to store the result in obsm of dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors_multiscaled'.

    Returns
    -------
    Union[pd.DataFrame, None]
        Multi-scale data matrix. If sc.AnnData is passed as dm_res, the result
        is written to its obsm[out_key] and None is returned.
    """
    if isinstance(dm_res, sc.AnnData):
        eigenvectors = dm_res.obsm[eigvec_key]
        if not isinstance(eigenvectors, pd.DataFrame):
            eigenvectors = pd.DataFrame(eigenvectors, index=dm_res.obs_names)
        dm_res_dict = {
            "EigenValues": dm_res.uns[eigval_key],
            "EigenVectors": eigenvectors,
        }
    else:
        dm_res_dict = dm_res

    if not isinstance(dm_res_dict, dict):
        raise ValueError("'dm_res' should be a dict or a sc.AnnData instance")
    if n_eigs is None:
        vals = np.ravel(dm_res_dict["EigenValues"])
        n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

    # Scale the data
    use_eigs = list(range(1, n_eigs))
    eig_vals = np.ravel(dm_res_dict["EigenValues"][use_eigs])
    data = dm_res_dict["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=dm_res_dict["EigenVectors"].index)

    if isinstance(dm_res, sc.AnnData):
        dm_res.obsm[out_key] = data.values

    return data


def _return_cell(ec, obs_names, celltype, mm, dcomp):
    """
    Helper function to print and return the early cell.

    Args:
        ec (int): Index of the early cell.
        obs_names (list): Names of cells.
        celltype (str): The cell type of interest.
        mm (str): Max/min status of the diffusion component.
        dcomp (int): Index of diffusion component.

    Returns:
        str: Name of the early cell.
    """
    early_cell = obs_names[ec]
    print(
        f"Using {early_cell} for cell type {celltype} which is {mm} in "
        f"diffusion component {dcomp}."
    )
    return early_cell


def early_cell(
    ad: sc.AnnData,
    celltype: str,
    celltype_column: str = "celltype",
    eigvec_key: str = "DM_EigenVectors_multiscaled",
    fallback_seed: int = None,
):
    """
    Helper function to determine 'early_cell' for 'run_palantir'.
    It identifies the cell of 'celltype' at the extremes of the state space represented by diffusion maps.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix.
    celltype : str
        The specific cell type of interest for determining the early cell.
    celltype_column : str, optional
        Name of the column in the obs of the Anndata object where the cell type information is stored.
        Default is 'celltype'.
    eigvec_key : str, optional
        Key to access multiscale space diffusion components from obsm of ad.
        Default is 'DM_EigenVectors_multiscaled'.
    fallback_seed : int, optional
        Seed for random number generator in fallback method. If not specified, no seed is used.
        Default is None.

    Returns
    -------
    str
        Name of the early cell for the given cell type.

    Raises
    ------
    CellNotFoundException
        If no valid cell of the specified type can be found at the extremes of the diffusion map.
    """
    if not isinstance(ad, sc.AnnData):
        raise ValueError("'ad' should be an instance of sc.AnnData")

    if eigvec_key not in ad.obsm:
        raise ValueError(
            f"'{eigvec_key}' not found in ad.obsm. "
            "Run `palantir.utils.run_diffusion_maps(ad)` to "
            "compute diffusion map eigenvectors."
        )
    eigenvectors = ad.obsm[eigvec_key]
    if isinstance(eigenvectors, pd.DataFrame):
        eigenvectors = eigenvectors.values

    if not isinstance(celltype_column, str):
        raise ValueError(f"celltype_column='{celltype_column}' should be a string")

    if celltype_column not in ad.obs.columns:
        raise ValueError(
            f"celltype_column='{celltype_column}' should be a column of ad.obs."
        )

    if not isinstance(celltype, str):
        raise ValueError("celltype should be a string")

    if celltype not in ad.obs[celltype_column].values:
        raise ValueError(
            f"Celltype '{celltype}' not found in ad.obs['{celltype_column}']."
        )

    if fallback_seed is not None and not isinstance(fallback_seed, int):
        raise ValueError("'fallback_seed' should be an integer")

    for dcomp in range(eigenvectors.shape[1]):
        ec = eigenvectors[:, dcomp].argmax()
        if ad.obs[celltype_column][ec] == celltype:
            return _return_cell(ec, ad.obs_names, celltype, "max", dcomp)
        ec = eigenvectors[:, dcomp].argmin()
        if ad.obs[celltype_column][ec] == celltype:
            return _return_cell(ec, ad.obs_names, celltype, "min", dcomp)

    if fallback_seed is not None:
        print("Falling back to slow early cell detection.")
        return fallback_terminal_cell(
            ad, celltype, celltype_column=celltype_column, seed=fallback_seed
        )

    raise CellNotFoundException(
        f"No valid component found: {celltype} "
        "Consider increasing the number of diffusion components "
        "('n_components' in palantir.utils.run_diffusion_maps) "
        "or specify a 'fallback_seed' to determine an early cell based on "
        f"reverse pseudotime starting from random non-{celltype} cell."
    )


def fallback_terminal_cell(
    ad: sc.AnnData,
    celltype: str,
    celltype_column: str = "anno",
    eigvec_key: str = "DM_EigenVectors_multiscaled",
    seed: int = 2353,
):
    """
    Fallback method to identify terminal cells when no valid diffusion component
    is found for the specified cell type.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix.
    celltype : str
        The specific cell type of interest for determining the terminal cell.
    celltype_column : str, optional
        Name of the column in the obs of the Anndata object where the cell type information is stored.
        Default is 'anno'.
    eigvec_key : str, optional
        Key to access multiscale space diffusion components from obsm of ad.
        Default is 'DM_EigenVectors_multiscaled'.
    seed : int, optional
        Seed for random number generator in fallback method. If not specified, no seed is used.
        Default is 2353.

    Returns
    -------
    str
        Name of the terminal cell for the given cell type.

    """
    other_cells = ad.obs_names[ad.obs[celltype_column] != celltype]
    fake_early_cell = other_cells.to_series().sample(1, random_state=seed)[0]
    pr_res = run_palantir(
        ad,
        fake_early_cell,
        eigvec_key=eigvec_key,
        terminal_states=None,
        use_early_cell_as_start=True,
    )
    idx = ad.obs[celltype_column] == celltype
    ec = pr_res.pseudotime[idx].argmax()
    early_cell = ad.obs_names[idx][ec]
    print(
        f"Using {early_cell} for cell type {celltype} which is latest cell in "
        f"{celltype} when starting from {fake_early_cell}."
    )
    return early_cell


def find_terminal_states(
    ad: sc.AnnData,
    celltypes: Iterable,
    celltype_column: str = "celltype",
    eigvec_key: str = "DM_EigenVectors_multiscaled",
    fallback_seed: int = None,
):
    """
    Identifies terminal states for a list of cell types in the AnnData object.

    This function iterates over the provided cell types, trying to find a terminal cell for each one
    using the 'early_cell' function. If no valid component is found for a cell type, it emits a warning and
    proceeds to the next cell type.

    Parameters
    ----------
    ad : sc.AnnData
        Annotated data matrix from Scanpy. It should contain computed diffusion maps.
    celltypes : Iterable
        An iterable such as a list or tuple of cell type names for which terminal states should be identified.
    celltype_column : str, optional
        The name of the column in the obs dataframe of the Anndata object where the cell type information is
        stored. By default, it is 'celltype'.
    eigvec_key : str, optional
        Key to access multiscale space diffusion components from obsm of ad.
        Default is 'DM_EigenVectors_multiscaled'.
    fallback_seed : int, optional
        Seed for the random number generator used in the fallback method. Defaults to None, in which case
        the random number generator will be randomly seeded.

    Returns
    -------
    pd.Series
        A pandas Series where the indices are the cell types and the values are the names of the terminal cells.
        If no terminal cell is found for a cell type, it will not be included in the series.
    """
    terminal_states = pd.Series(dtype=str)
    for ct in celltypes:
        try:
            cell = early_cell(ad, ct, celltype_column, eigvec_key, fallback_seed)
        except CellNotFoundException:
            warn(
                f"No valid component found: {ct} "
                "Consider increasing the number of diffusion components "
                "('n_components' in palantir.utils.run_diffusion_maps). "
                f"The cell type {ct} will be skipped."
            )
            continue
        terminal_states[cell] = ct
    return terminal_states
