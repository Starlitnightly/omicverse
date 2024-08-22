import scanpy as sc
from ..externel.PROST import prepare_for_PI,cal_PI,spatial_autocorrelation,feature_selection

def svg(adata,mode='prost',n_svgs=3000,target_sum=50*1e4,platform="visium",
        mt_startwith='MT-'):
    """
    Find the spatial variable genes.
    """
    if mode=='prost':
        if 'counts' not in adata.layers.keys():
            adata.layers['counts'] = adata.X.copy()
        # Calculate PI
        adata = prepare_for_PI(adata, platform=platform)
        adata = cal_PI(adata, platform=platform)
        print('PI calculation is done!')

        # Spatial autocorrelation test
        spatial_autocorrelation(adata)
        print('Spatial autocorrelation test is done!')

        # Remove MT-gene
        drop_gene_name = mt_startwith
        selected_gene_name=list(adata.var_names[adata.var_names.str.contains(mt_startwith)==False])
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print('normalization and log1p are done!')
        #adata.raw = adata
        adata = feature_selection(adata, 
                                  by = mode, n_top_genes = n_svgs)
        #print(f'{n_svgs} SVGs are selected!')
    else:
        raise ValueError(f"mode {mode} is not supported")
    return adata
    # End-of-file (EOF)
