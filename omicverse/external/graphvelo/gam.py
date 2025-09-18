from typing import Optional, Literal
from pygam import LinearGAM, s

import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from anndata import AnnData

from .utils import flatten
from .kernel_density_smooth import kde2d


def fit_velo_peak(
    adata, 
    genes, 
    tkey: str, 
    layer: str = 'Ms', 
    log_norm: bool = True,
    max_iter: int = 2000,
    **kwargs
):
    """
    Identify the peak velocity (i.e. maximum predicted value) for each gene using GAM.
    
    For each gene in `genes`, a GAM is fit using the independent variable stored in 
    adata.obs[tkey] and the expression values from adata[:, gene].layers[layer]. 
    Then the time (or phase) corresponding to the maximum predicted expression is found 
    by minimizing the negative of the GAM prediction.
    
    Parameters:
        adata (anndata.AnnData): AnnData object containing the data.
        genes (str or list of str): One or more gene names.
        tkey (str): Key in adata.obs for the independent variable (e.g., pseudotime).
        layer (str): Name of the layer to use for expression values (default 'Ms').
        log_norm (bool): Whether to apply log1p normalization to the magnitude.
        max_iter (int): Maximum iterations for GAM fitting.
        **kwargs: Additional keyword arguments for GAM (e.g., n_splines, spline_order).
    
    Returns:
        pd.DataFrame: A DataFrame with gene names as the index and two columns:
                      'phase' for the peak time (or phase) and 'magnitude' for the 
                      predicted expression at the peak (optionally log-normalized).
    """
    
    # Check that tkey is present
    if tkey not in adata.obs.keys():
        raise ValueError(f'{tkey} not found in adata.obs')
    t = adata.obs[tkey].values  # Independent variable
    
    # Process gene list
    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        logging.warning(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    
    # Set up GAM keyword arguments (default: 7 splines, order 3)
    gam_kwargs = {
        'n_splines': 7,
        'spline_order': 3
    }
    gam_kwargs.update(kwargs)
    
    phase = np.zeros(gn)
    magnitude = np.zeros(gn)
        
    for i, gene in tqdm(enumerate(genes), total=gn, desc="Identify velocity peaks using GAM"):
        x = adata[:, gene].layers[layer]
        x = flatten(x)
        
        gam = LinearGAM(s(0, **gam_kwargs), max_iter=max_iter, verbose=False).fit(t, x)
        def neg_gam(x_val):
            return -gam.predict(np.array([[x_val]]))[0]
        
        res = minimize_scalar(neg_gam, bounds=(t.min(), t.max()), method='bounded')
        peak_t = res.x
        phase[i] = peak_t
        magnitude[i] = gam.predict(np.array([[peak_t]]))[0]
    
    # Optionally apply log normalization to the magnitude， genes with negative peak velo can cause error.
    if log_norm:
        magnitude = np.log1p(magnitude)
    
    df = pd.DataFrame({'phase': phase, 'magnitude': magnitude}, index=genes)
    return df
    

def fit_gene_trend(
    adata, 
    genes, 
    tkey:str, 
    layer:str='Ms', 
    max_iter: int = 2000,
    grid_num: int = 200,
    **kwargs):
    """
    Fit gene expression trends over a given time or pseudotime variable using Generalized
    Additive Models (GAMs) and return the predictions in an AnnData object.

    Parameters:
    -----------
    adata : AnnData
        An AnnData object containing the expression data in adata.layers and the
        time/pseudotime information in adata.obs.
    genes : str or list of str
        Gene name (or list of gene names) for which to fit the trend.
    tkey : str
        Key in adata.obs that contains the time or pseudotime variable.
    layer : str, default 'Ms'
        The layer in adata.layers to extract expression values from.
    max_iter : int, default 2000
        Maximum iterations for fitting the GAM.
    grid_num : int, default 200
        Number of points in the grid over which the trend is predicted.
    **kwargs : dict
        Additional keyword arguments to pass to the GAM fitting routine (e.g., n_splines, spline_order).

    Returns:
    --------
    gdata : AnnData
        An AnnData object where .X is a matrix of shape (number of genes, grid_num) containing
        the predicted trend for each gene, and the .obs_names are set to the corresponding gene names.
    """
    
    if tkey not in adata.obs.keys():
        raise ValueError(f'{tkey} not found in adata.obs')
    t = adata.obs[tkey]

    if isinstance(genes, str):
        genes = [genes]
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        print(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)

    gam_kwargs = {
        'n_splines': 7,
        'spline_order': 3
    }
    gam_kwargs.update(kwargs)

    data = np.zeros((gn, grid_num)) # N_genes * N_grid

    for i, gene in tqdm(
        enumerate(genes),
        total=gn,
        desc="Fitting trends using GAM",):
        x = adata[:, gene].layers[layer]
        x = x.A.flatten() if sp.issparse(x) else x.flatten()

        ### GAM fitting
        term  =s(
            0,
            **gam_kwargs)
        gam = LinearGAM(term, max_iter=max_iter, verbose=False).fit(t, x)
        x_lins = np.linspace(t.min(), t.max(), grid_num)
        y_pred = gam.predict(x_lins)
        data[i] = y_pred
    
    gdata = AnnData(data)
    gdata.obs_names = pd.Index(genes)
    return gdata


def fit_response(
    adata: AnnData,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = 'M_sc',
    ykey: Optional[str] = 'jacobian',
    norm: bool = True,
    log: bool = False,
    grid_num: int = 200,
    **kwargs,
):
    """
    Fit response curves for gene pairs and generate a grid of GAM predictions.

    For each gene pair provided in pairs_mat, this function:
      - Retrieves the response values (x) from the specified layer (xkey) for the source gene.
      - Computes the Jacobian (y) for the target gene using dynamo's get_jacobian function.
      - Filters out invalid data points (non-finite or zero sums).
      - Optionally applies log transformation and normalization.
      - Fits a Generalized Additive Model (GAM) to predict the response over a grid.
      - Stores the predicted response curve into a grid (data).

    The function returns an AnnData object where:
      - The .X matrix contains the grid data for each gene pair.
      - The .obs_names are set to the target gene names from each pair.
    """

    try:
        from dynamo.vectorfield.utils import get_jacobian
    except ImportError:
        raise ImportError(
            "If you want to do jacobian analysis related to dynamo, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    if not set([xkey, ykey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )
    
    all_genes_in_pair = np.unique(pairs_mat)
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )
    if not ykey.startswith("jacobian"):
        raise KeyError('The ykey should start with `jacobian`.')
    
    genes = []
    xy = pd.DataFrame()
    id = 0
    for _, gene_pairs in enumerate(pairs_mat):
        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]
        genes.append(gene_pairs[1])

        x = flatten(adata[:, gene_pairs[0]].layers[xkey])
        J_df = get_jacobian(
            adata,
            gene_pairs[0],
            gene_pairs[1],
        )
        jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
        y_ori = flatten(J_df[jkey])

        finite = np.isfinite(x + y_ori)
        nonzero = np.abs(x) + np.abs(y_ori) > 0
        valid_ids = np.logical_and(finite, nonzero)

        x, y_ori = x[valid_ids], y_ori[valid_ids]

        if log:
            x, y_ori = x if sum(x < 0) else np.log(np.array(x) + 1), y_ori if sum(y_ori) < 0 else np.log(
                np.array(y_ori) + 1)

        if norm:
            y_ori = y_ori / y_ori.max()

        y = y_ori

        cur_data = pd.DataFrame({"x": x, "y": y, "type": gene_pair_name})
        xy = pd.concat([xy, cur_data], axis=0)

        id = id + 1

    data = np.zeros((len(pairs_mat), grid_num)) # N_genes * N_grid

    for gene_idx, res_type in enumerate(xy.type.unique()):
        gene_pairs = res_type.split("->")
        xy_subset = xy[xy["type"]==res_type]
        x_val, y_val = xy_subset["x"], xy_subset["y"]

        gam_kwargs = {
            'n_splines': 12,
            'spline_order': 3
        }
        gam_kwargs.update(kwargs)
        term  =s(
            0,
            **gam_kwargs)
        gam = LinearGAM(term, max_iter=1000, verbose=False).fit(x_val, y_val)
        
        x_lins = np.linspace(x_val.min(), x_val.max(), grid_num)
        y_pred = gam.predict(x_lins)
        data[gene_idx] = y_pred
    
    adata = AnnData(data)
    adata.obs_names = pd.Index(genes)

    return adata


def cluster_response(
    adata: AnnData,
    pairs_mat: np.ndarray,
    xkey: Optional[str] = 'M_sc',
    ykey: Optional[str] = 'jacobian',
    grid_num: int = 25,
    kde_backend: Literal['fixbdw', 'scipy', 'statsmodels'] = 'statsmodels',
):
    """
    Cluster responses for gene pairs by computing 2D density estimates on a grid.

    For each gene pair:
      - Retrieves the response values (x) from the specified xkey layer.
      - Retrieves the Jacobian values (y) using dynamo's get_jacobian.
      - Uses a 2D kernel density estimation (kde2d) to compute density over a grid.
      - Normalizes the density over y for each x (row normalization).
      - Flattens the grid and stores it for clustering analysis.

    The function returns a DataFrame where:
      - Each row corresponds to a gene pair (indexed by "gene_pair_name").
      - The columns are the flattened density values on the grid.
    """
    
    try:
        from dynamo.vectorfield.utils import get_jacobian
    except ImportError:
        raise ImportError(
            "If you want to do jacobian analysis related to dynamo, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    if not set([xkey, ykey]) <= set(adata.layers.keys()).union(set(["jacobian"])):
        raise ValueError(
            f"adata.layers doesn't have {xkey, ykey} layers. Please specify the correct layers or "
            "perform relevant preprocessing and vector field analyses first."
        )
    
    all_genes_in_pair = np.unique(pairs_mat)
    if not (set(all_genes_in_pair) <= set(adata.var_names)):
        raise ValueError(
            "adata doesn't include all genes in gene_pairs_mat. Make sure all genes are included in adata.var_names."
        )
    if not ykey.startswith("jacobian"):
        raise KeyError('The ykey should start with `jacobian`.')
    
    flat_data = np.zeros((len(pairs_mat), grid_num**2))
    data_idx = []
    # extract information from dynamo output in adata object
    for gene_idx, gene_pairs in enumerate(pairs_mat):
        gene_pair_name = gene_pairs[0] + "->" + gene_pairs[1]
        data_idx.append(gene_pair_name)

        x = flatten(adata[:, gene_pairs[0]].layers[xkey])

        J_df = get_jacobian(
            adata,
            gene_pairs[0],
            gene_pairs[1],
        )
        jkey = gene_pairs[0] + "->" + gene_pairs[1] + "_jacobian"
        y_ori = flatten(J_df[jkey])
        y = y_ori

        # den_res[0, 0] is at the lower bottom; dens[1, 4]: is the 2nd on x-axis and 5th on y-axis
        x_meshgrid, y_meshgrid, den_res = kde2d(
            x, y, n=[grid_num, grid_num], lims=[-max(x), max(x), -max(y), max(y)], 
            backend=kde_backend
        )
        den_res = np.array(den_res)
        den_x = np.sum(den_res, axis=1)

        data = np.zeros_like(den_res)

        for i in range(len(x_meshgrid)):
            tmp = den_res[i] / den_x[i]  # condition on each input x, normalize over y
            tmp = den_res[i]
            max_val = max(tmp)
            min_val = min(tmp)

            rescaled_val = (tmp - min_val) / (max_val - min_val)
            data[i] = rescaled_val

        data = flatten(data)
        flat_data[gene_idx] = data

    def scale_func(x, X, grid_num):
        return grid_num * (x - np.min(X)) / (np.max(X) - np.min(X))

    data = pd.DataFrame(flat_data, index=data_idx)

    return data