import scanpy as sc
from ..external.PROST import prepare_for_PI,cal_PI,spatial_autocorrelation,feature_selection
from ..pp import preprocess
from .._settings import add_reference
from ..utils.registry import register_function

@register_function(
    aliases=["空间变异基因", "svg", "spatially_variable_genes", "空间变异基因检测", "SVG检测"],
    category="space",
    description="Identify spatially variable genes using multiple methods (PROST, Pearson, Spateo)",
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
        "# Access identified SVGs",
        "svgs = adata.var_names[adata.var['space_variable_features']]"
    ],
    related=["pp.preprocess", "space.clusters", "space.pySTAGATE"]
)
def svg(adata,mode='prost',n_svgs=3000,target_sum=50*1e4,platform="visium",
        mt_startwith='MT-',**kwargs):
    r"""Identify spatially variable genes using multiple methods.
    
    This function identifies genes that show significant spatial variation in their
    expression patterns across the tissue. It supports multiple methods including
    PROST (Pattern RecognitiOn of Spatial Transcriptomics), Pearson correlation,
    and Spateo-based analysis.

    Arguments:
        adata: AnnData
            Annotated data matrix containing spatial transcriptomics data.
            Must contain:
            - Raw counts in adata.X or adata.layers['counts']
            - Spatial coordinates in adata.obsm['spatial']
        mode: str, optional (default='prost')
            Method for identifying spatially variable genes:
            - 'prost': Pattern RecognitiOn of Spatial Transcriptomics
            - 'pearsonr': Pearson correlation-based method
            - 'spateo': Spateo-based analysis using Wasserstein distance
        n_svgs: int, optional (default=3000)
            Number of spatially variable genes to select.
        target_sum: float, optional (default=50*1e4)
            Target sum for library size normalization.
        platform: str, optional (default="visium")
            Spatial transcriptomics platform type.
        mt_startwith: str, optional (default='MT-')
            Prefix for mitochondrial genes to exclude from analysis.
        **kwargs:
            Additional arguments passed to specific SVG methods:
            For 'spateo':
                - log2fc: Minimum log2 fold change (default: 1)
                - rank_p: Maximum rank-based p-value (default: 0.05)
                - adj_pvalue: Maximum adjusted p-value (default: 0.05)

    Returns:
        AnnData
            Input AnnData object updated with:
            - adata.var['space_variable_features']: Boolean mask of selected SVGs
            - adata.var['highly_variable']: Alias for space_variable_features
            For 'prost' mode:
                - Additional PROST-specific metrics in adata.var
            For 'spateo' mode:
                - Wasserstein distance statistics in adata.var

    Notes:
        - PROST mode requires opencv-python package
        - Different modes use different statistical approaches:
            - PROST: Pattern recognition and spatial autocorrelation
            - pearsonr: Correlation between gene expression and spatial coordinates
            - spateo: Wasserstein distance-based spatial variation
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
    else:
        raise ValueError(f"mode {mode} is not supported")
    
    adata.var['highly_variable'] = adata.var['space_variable_features']
    return adata
    # End-of-file (EOF)
