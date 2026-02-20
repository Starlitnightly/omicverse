from ..pp import scale,pca
import scanpy as sc
import numpy as np
import anndata
from .._settings import add_reference,settings
from .._registry import register_function
from .._monitor import monitor

@monitor
@register_function(
    aliases=["批次校正", "batch_correction", "batch_correct", "数据整合", "去批次效应"],
    category="single",
    description="Comprehensive batch effect correction using multiple methods including Harmony, Combat, Scanorama, scVI, and CellANOVA",
    prerequisites={
        'optional_functions': ['preprocess', 'scale', 'pca']
    },
    requires={
        'obsm': [],  # Flexible - some methods use X_pca, others raw data
        'obs': []    # Requires batch_key column (user-specified)
    },
    produces={
        'obsm': []  # Dynamic: X_pca_harmony, X_combat, X_scanorama, X_scVI, or X_cellanova
    },
    auto_fix='none',
    examples=[
        "# Harmony batch correction (recommended for most cases)",
        "ov.single.batch_correction(adata, batch_key='batch', methods='harmony')",
        "# Combat batch correction",
        "ov.single.batch_correction(adata, batch_key='batch', methods='combat')",
        "# Scanorama integration",
        "ov.single.batch_correction(adata, batch_key='batch', methods='scanorama')",
        "# GPU-accelerated scVI (requires GPU)",
        "model = ov.single.batch_correction(adata, batch_key='batch',",
        "                                   methods='scVI', n_layers=2, n_latent=30)",
        "# CellANOVA with control samples",
        "control_dict = {'pool1': ['batch1', 'batch2']}",
        "ov.single.batch_correction(adata, batch_key='batch', methods='CellANOVA',",
        "                           control_dict=control_dict)"
    ],
    related=["pp.preprocess", "utils.mde", "utils.embedding"]
)
def batch_correction(adata:anndata.AnnData,batch_key:str,
                     use_rep='scaled|original|X_pca',
                     methods:str='harmony',n_pcs:int=50,**kwargs)->anndata.AnnData:
    """
    Batch correction for single-cell data

    Arguments:
        adata: AnnData object
        batch_key: batch key
        methods: harmony,combat,scanorama
        n_pcs: number of PCs
        kwargs: other parameters for harmony`harmonypy.run_harmony()`,combat`sc.pp.combat()`,scanorama`scanorama.integrate_scanpy()`

    Returns:
        adata: AnnData object
    
    """

    print(f'...Begin using {methods} to correct batch effect')

    if methods=='harmony':
        from ..external.harmony import run_harmony
        
        adata3=adata.copy()
        if 'scaled|original|X_pca' not in adata3.obsm.keys() and use_rep=='scaled|original|X_pca':
            scale(adata3)
            pca(adata3,layer='scaled',n_pcs=n_pcs)
        
        
        harmony_out = run_harmony(adata3.obsm[use_rep], adata3.obs, batch_key, **kwargs)
        adata.obsm['X_pca_harmony'] = harmony_out.result()
        adata.obsm['X_harmony'] = harmony_out.result()
        
        add_reference(adata,'Harmony','batch correction with Harmony')
        
    elif methods=='combat':
        adata2=adata.copy()
        sc.pp.combat(adata2, key=batch_key,**kwargs)
        scale(adata2)
        pca(adata2,layer='scaled',n_pcs=n_pcs)
        adata2.obsm['X_combat']=adata2.obsm[use_rep].copy()
        adata.obsm['X_combat']=adata2.obsm['X_combat'].copy()
        del adata2
        add_reference(adata,'Combat','batch correction with Combat')
        #return adata2
    elif methods=='scanorama':
        try:
            import intervaltree
            import fbpca
            #print('mofax have been install version:',mfx.__version__)
        except ImportError:
            raise ImportError(
                'Please install the intervaltree: `pip install intervaltree fbpca`.'
            )
        from ..external.scanorama import integrate_scanpy
        batches = adata.obs[batch_key].cat.categories.tolist()
        alldata = {}
        for batch in batches:
            alldata[batch] = adata[adata.obs[batch_key] == batch,]
        alldata2 = dict()
        for ds in alldata.keys():
            print(ds)
            alldata2[ds] = alldata[ds]
        
        #convert to list of AnnData objects
        adatas = list(alldata2.values())
        
        # run scanorama.integrate
        integrate_scanpy(adatas, dimred = n_pcs,**kwargs)
        scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]

        # make into one matrix.
        all_s = np.concatenate(scanorama_int)
        print(all_s.shape)
        
        # add to the AnnData object, create a new object first
        adata.obsm["X_scanorama"] = all_s
        add_reference(adata,'Scanorama','batch correction with Scanorama')
        return adata
    elif methods=='scVI':
        try:
            import scvi 
            #print('mofax have been install version:',mfx.__version__)
        except ImportError:
            raise ImportError(
                'Please install the scVI: `pip install scvi-tools`. or `conda install scvi-tools -c conda-forge`'
            )
        import scvi
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
        model = scvi.model.SCVI(adata, **kwargs)
        model.train()
        SCVI_LATENT_KEY = "X_scVI"
        adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
        add_reference(adata,'scVI','batch correction with scVI')
        return model
    elif methods=='CellANOVA':
        from ..external.cellanova.model import calc_ME,calc_BE,calc_TE
        if ('highly_variable_features' in adata.var.columns) and ('highly_variable' not in adata.var.columns):
            adata.var['highly_variable']=adata.var['highly_variable_features']
        adata= calc_ME(adata, integrate_key=batch_key)
        adata = calc_BE(adata,  integrate_key=batch_key, **kwargs)
        adata = calc_TE(adata,  integrate_key=batch_key)
        from scipy.sparse import csr_matrix
        adata.layers['denoised']=csr_matrix(adata.layers['denoised'])
        ## create an independent anndata object for cellanova-integrated data
        pca(adata,layer='denoised',n_pcs=n_pcs)
        adata.obsm['X_cellanova']=adata.obsm['denoised|original|X_pca'].copy()
        add_reference(adata,'CellANOVA','batch correction with CellANOVA')
    elif methods=='Concord' or methods=='concord':
        try:
            import concord as ccd
        except ImportError:
            raise ImportError(
                'Please install the concord: `pip install concord-sc`.'
            )
        import torch
        # Set device to cpu or to gpu (if your torch has been set up correctly to use GPU), for mac you can use either torch.device('mps') or torch.device('cpu')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if 'highly_variable' in adata.var.columns:
            feature_list = adata.var.loc[adata.var['highly_variable']==True].index.tolist()
        elif 'highly_variable_features' in adata.var.columns:
            feature_list = adata.var.loc[adata.var['highly_variable_features']==True].index.tolist()
        else:
            print('No highly variable features found, using all features')
            feature_list = adata.var_names.tolist()

        # Initialize Concord with an AnnData object, skip input_feature to use all features, set preload_dense=False if your data is very large
        # Provide 'domain_key' if integrating across batches, see below
        cur_ccd = ccd.Concord(
            adata=adata, input_feature=feature_list, 
            preload_dense=True,domain_key=batch_key, **kwargs
        ) 

        # Encode data, saving the latent embedding in adata.obsm['Concord']
        cur_ccd.fit_transform(output_key='X_concord')
        add_reference(adata,'Concord','batch correction with Concord')
        return cur_ccd
    else:
        print('Not supported')

    