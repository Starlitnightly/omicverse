import scanpy as sc
from ..pp import preprocess
from .._settings import add_reference
from .._registry import register_function

@register_function(
    aliases=["空间变异基因", "svg", "spatially_variable_genes", "空间变异基因检测", "SVG检测"],
    category="space",
    description="Identify spatially variable genes using multiple methods (PROST, Pearson, Spateo, SOMDE, SpatialDE)",
    prerequisites={
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial']  # Spatial coordinates required
    },
    produces={
        'var': ['space_variable_features']
    },
    auto_fix='none',
    examples=[
        "# Basic SVG detection with PROST",
        "adata = ov.space.svg(adata, mode='prost', n_svgs=3000)",
        "# Using Pearson correlation method",
        "adata = ov.space.svg(adata, mode='pearsonr', n_svgs=2000)",
        "# High-resolution analysis",
        "adata = ov.space.svg(adata, mode='prost', n_svgs=5000,",
        "                     target_sum=1e5, platform='visium')",
        "# Using SOMDE (SOM-accelerated SpatialDE)",
        "adata = ov.space.svg(adata, mode='somde', k=20)",
        "# SOMDE with custom threshold and extra training",
        "adata = ov.space.svg(adata, mode='somde', k=20, qval_threshold=0.05, retrain_epoch=100)",
        "# Using SpatialDE (GP-based, direct on single cells)",
        "adata = ov.space.svg(adata, mode='spatialde', qval_threshold=0.05)",
        "# SpatialDE with custom gene filter and regress formula",
        "adata = ov.space.svg(adata, mode='spatialde', min_total_count=3,",
        "                     regress_formula='np.log(total_counts)')",
        "# Access identified SVGs",
        "svgs = adata.var_names[adata.var['space_variable_features']]"
    ],
    related=["pp.preprocess", "space.clusters", "space.pySTAGATE"]
)
def svg(adata,mode='prost',n_svgs=3000,target_sum=50*1e4,platform="visium",
        mt_startwith='MT-',**kwargs):
    # somde kwargs: k=20, qval_threshold=0.05, retrain_epoch=0
    r"""Identify spatially variable genes using multiple methods.
    
    This function identifies genes that show significant spatial variation in their
    expression patterns across the tissue. It supports multiple methods including
    PROST (Pattern RecognitiOn of Spatial Transcriptomics), Pearson correlation,
    and Spateo-based analysis.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing expression matrix and coordinates in
        ``adata.obsm['spatial']``.
    mode : {'prost', 'pearsonr', 'spateo', 'somde', 'spatialde'}, default='prost'
        SVG detection backend.
    n_svgs : int, default=3000
        Number of spatially variable genes to select.
    target_sum : float, default=50*1e4
        Target-sum used during normalization.
    platform : str, default='visium'
        Platform identifier used by PROST preprocessing.
    mt_startwith : str, default='MT-'
        Mitochondrial gene prefix excluded by default.
    **kwargs
        Additional method-specific options.
        For ``somde``: ``k``, ``qval_threshold``, ``retrain_epoch``.
        For ``spatialde``: ``qval_threshold``, ``min_total_count``, ``regress_formula``.

    Returns
    -------
    AnnData
        Updated AnnData with SVG flags in
        ``adata.var['space_variable_features']`` and ``adata.var['highly_variable']``.

    Notes:
        - PROST mode requires opencv-python package
        - Different modes use different statistical approaches:
            - PROST: Pattern recognition and spatial autocorrelation
            - pearsonr: Correlation between gene expression and spatial coordinates
            - spateo: Wasserstein distance-based spatial variation
            - somde: SOM-compressed SpatialDE GP test (fast for large datasets)
        - SOMDE kwargs: ``k`` (cells/node, default 20), ``qval_threshold`` (default 0.05),
          ``retrain_epoch`` (extra SOM epochs, default 0)
        - SOMDE stores ``adata.var['somde_LLR']``, ``somde_pval``, ``somde_qval``, ``somde_FSV``
        - SOMDE requires ``somoclu`` and ``patsy``
        - spatialde: GP-based test directly on single-cell coordinates (no SOM compression),
          uses bundled NaiveDE + SpatialDE packages from ``omicverse/external/``
        - SpatialDE kwargs: ``qval_threshold`` (default 0.05), ``min_total_count`` (default 3),
          ``regress_formula`` (default ``'np.log(total_counts)'``)
        - SpatialDE stores ``adata.var['spatialde_LLR']``, ``spatialde_pval``, ``spatialde_qval``,
          ``spatialde_FSV``, ``spatialde_l``
        - SpatialDE requires ``patsy`` and ``tqdm``
        - Mitochondrial genes are excluded by default
        - Results are normalized and log-transformed

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load spatial data
        >>> adata = sc.read_visium(...)
        >>> # Find SVGs using PROST
        >>> adata = ov.space.svg(
        ...     adata,
        ...     mode='prost',
        ...     n_svgs=2000,
        ...     platform='visium'
        ... )
        >>> # Access SVGs
        >>> svgs = adata.var_names[adata.var['space_variable_features']]
    """
    if mode=='prost':
        from ..external.PROST import prepare_for_PI,cal_PI,spatial_autocorrelation,feature_selection

        if 'counts' not in adata.layers.keys():
            adata.layers['counts'] = adata.X.copy()
        # Calculate PI
        try:
            import cv2
        except ImportError:
            print("Please install the package cv2 by \"pip install opencv-python\"")
            import sys
            sys.exit(1)
        adata = prepare_for_PI(adata, platform=platform)
        adata = cal_PI(adata, platform=platform)
        print('PI calculation is done!')

        # Spatial autocorrelation test
        #spatial_autocorrelation(adata)
        #print('Spatial autocorrelation test is done!')

        # Remove MT-gene
        drop_gene_name = mt_startwith
        selected_gene_name=list(adata.var_names[adata.var_names.str.contains(mt_startwith)==False])
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print('normalization and log1p are done!')
        #adata.raw = adata
        adata = feature_selection(adata, 
                                  by = mode, n_top_genes = n_svgs)
        add_reference(adata,'PROST','spatial variable gene selection with PROST')
        #print(f'{n_svgs} SVGs are selected!')
    elif mode=='pearsonr':
        from ..pp import preprocess
        adata=preprocess(adata,mode='shiftlog|pearson',n_HVGs=n_svgs,target_sum=target_sum)
        adata.var['space_variable_features']=adata.var['highly_variable_features']
        add_reference(adata,'scanpy','spatial variable gene selection with pearsonr')
        #adata.raw = adata
        #adata = adata[:, adata.var.highly_variable_features]
    elif mode=='spateo':
        import spateo as st
        from ..pp import preprocess
        adata=preprocess(adata,mode='shiftlog|pearson',n_HVGs=n_svgs,target_sum=target_sum)
        e16_w, _ = st.svg.cal_wass_dis_bs(adata, **kwargs)
        # Add positive rate before smoothing for each gene
        st.svg.add_pos_ratio_to_adata(adata, layer='counts')
        e16_w['pos_ratio_raw'] = adata.var['pos_ratio_raw']
        # We obtain 529 significant SVGs
        sig_df = e16_w[(e16_w['log2fc']>=1) & (e16_w['rank_p']<=0.05) & (e16_w['pos_ratio_raw']>=0.05) & (e16_w['adj_pvalue']<=0.05)]
        adata.var['space_variable_features'] = False
        adata.var.loc[sig_df.index, 'space_variable_features'] = True
        print(f'{len(sig_df)} SVGs are selected!')
        print('In mode of spateo, the SVGs are selected based on the spatial expression pattern.')
        add_reference(adata,'spateo','spatial variable gene selection with spateo')
    elif mode == 'somde':
        import numpy as np
        import pandas as pd
        import scipy.sparse as _sp

        try:
            from ..external.somde import SomNode
        except ImportError as e:
            raise ImportError(
                "SOMDE requires `somoclu` and `patsy`. "
                "Install with: pip install somoclu patsy"
            ) from e

        k             = kwargs.get('k', 20)
        qval_thresh   = kwargs.get('qval_threshold', 0.05)
        retrain_epoch = kwargs.get('retrain_epoch', 0)
        n_jobs        = kwargs.get('n_jobs', 1)

        # --- spatial coordinates (cells × 2) ---
        X = adata.obsm['spatial'].astype(np.float32)

        # --- expression matrix (genes × cells) as DataFrame ---
        if 'counts' in adata.layers:
            mat = adata.layers['counts']
        else:
            mat = adata.X
        if _sp.issparse(mat):
            mat = mat.toarray()
        df = pd.DataFrame(
            mat.T,
            index=adata.var_names,
            columns=adata.obs_names,
        )

        print(f'Running SOMDE: {adata.n_obs} cells → ~{adata.n_obs // k} SOM nodes (k={k})')

        # --- SOM training ---
        som = SomNode(X, k)
        if retrain_epoch > 0:
            print(f'Re-training SOM for {retrain_epoch} additional epochs...')
            som.reTrain(retrain_epoch)

        # --- aggregate → normalize → SpatialDE test ---
        ndf, ninfo = som.mtx(df)
        nres = som.norm()
        result, SVnum = som.run(n_jobs=n_jobs)

        # --- store statistics back to adata.var ---
        result_indexed = result.set_index('g')
        for col in ('LLR', 'pval', 'qval', 'FSV'):
            if col in result_indexed.columns:
                adata.var[f'somde_{col}'] = result_indexed.reindex(adata.var_names)[col].values

        # --- select SVGs ---
        sig_mask = result_indexed.reindex(adata.var_names)['qval'] < qval_thresh
        adata.var['space_variable_features'] = sig_mask.fillna(False).values

        n_sig = adata.var['space_variable_features'].sum()
        if n_sig == 0 and n_svgs is not None:
            # fallback: top n_svgs ranked by LLR
            top_genes = set(result.head(n_svgs)['g'].tolist())
            adata.var['space_variable_features'] = adata.var_names.isin(top_genes)
            print(f'No genes with qval < {qval_thresh}; selected top {n_svgs} by LLR instead.')
        else:
            print(f'{n_sig} SVGs selected (qval < {qval_thresh}), total tested: {SVnum}')

        add_reference(adata, 'SOMDE', 'spatial variable gene selection with SOMDE')
    elif mode == 'spatialde':
        import numpy as np
        import pandas as pd
        import scipy.sparse as _sp

        try:
            from ..external.SpatialDE import run as _spatialde_run
            from ..external.NaiveDE import stabilize as _stabilize, regress_out as _regress_out
        except ImportError as e:
            raise ImportError(
                "SpatialDE requires `patsy` and `tqdm`. "
                "Install with: pip install patsy tqdm"
            ) from e

        qval_thresh     = kwargs.get('qval_threshold', 0.05)
        min_total_count = kwargs.get('min_total_count', 3)
        regress_formula = kwargs.get('regress_formula', 'np.log(total_counts)')
        n_jobs          = kwargs.get('n_jobs', 1)

        # --- raw counts (cells × genes) ---
        if 'counts' in adata.layers:
            mat = adata.layers['counts']
        else:
            mat = adata.X
        if _sp.issparse(mat):
            mat = mat.toarray()
        counts_df = pd.DataFrame(mat, index=adata.obs_names, columns=adata.var_names)

        # filter genes with too few total counts
        gene_mask = counts_df.sum(0) >= min_total_count
        counts_filt = counts_df.loc[:, gene_mask]
        n_pass = int(gene_mask.sum())
        print(f'SpatialDE: {n_pass}/{adata.n_vars} genes pass '
              f'min_total_count={min_total_count} filter')

        # --- sample_info: spatial coordinates + total_counts ---
        coords = adata.obsm['spatial'].astype(float)
        sample_info = pd.DataFrame(coords, index=adata.obs_names, columns=['x', 'y'])
        sample_info['total_counts'] = counts_filt.sum(1).values

        # --- NaiveDE: variance stabilize → regress out library size ---
        norm_expr  = _stabilize(counts_filt.T).T
        resid_expr = _regress_out(sample_info, norm_expr.T, regress_formula).T

        # --- SpatialDE GP test ---
        # pass numpy array to avoid pandas multi-dim indexing deprecation in older spatialDE
        X_coords = sample_info[['x', 'y']].values.astype(float)
        print(f'Running SpatialDE on {resid_expr.shape[1]} genes × '
              f'{resid_expr.shape[0]} cells...')
        results = _spatialde_run(X_coords, resid_expr, n_jobs=n_jobs)

        # --- store per-gene statistics back to adata.var ---
        results_idx = results.set_index('g')
        for col in ('LLR', 'pval', 'qval', 'FSV', 'l'):
            if col in results_idx.columns:
                adata.var[f'spatialde_{col}'] = (
                    results_idx.reindex(adata.var_names)[col].values
                )

        # --- select SVGs by qval threshold ---
        sig_mask = results_idx.reindex(adata.var_names)['qval'] < qval_thresh
        adata.var['space_variable_features'] = sig_mask.fillna(False).values

        n_sig = int(adata.var['space_variable_features'].sum())
        if n_sig == 0 and n_svgs is not None:
            top_genes = set(results.head(n_svgs)['g'].tolist())
            adata.var['space_variable_features'] = adata.var_names.isin(top_genes)
            print(f'No genes with qval < {qval_thresh}; '
                  f'selected top {n_svgs} by LLR instead.')
        else:
            print(f'{n_sig} SVGs selected (qval < {qval_thresh}), '
                  f'total tested: {n_pass}')

        add_reference(adata, 'SpatialDE', 'spatial variable gene selection with SpatialDE')
    else:
        raise ValueError(f"mode {mode} is not supported")

    adata.var['highly_variable'] = adata.var['space_variable_features']
    return adata
    # End-of-file (EOF)
