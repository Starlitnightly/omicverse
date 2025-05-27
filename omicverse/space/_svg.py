import scanpy as sc
from ..externel.PROST import prepare_for_PI,cal_PI,spatial_autocorrelation,feature_selection
from ..pp import preprocess
from .._settings import add_reference
def svg(adata,mode='prost',n_svgs=3000,target_sum=50*1e4,platform="visium",
        mt_startwith='MT-',**kwargs):
    """
    Find the spatial variable genes.
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
