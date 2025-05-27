from ..pp import *
import scanpy as sc
import numpy as np
import anndata
from .._settings import add_reference

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
        try:
            import harmonypy
            #print('mofax have been install version:',mfx.__version__)
        except ImportError:
            raise ImportError(
                'Please install the harmonypy: `pip install harmonypy`.'
            )
        
        adata3=adata.copy()
        if 'scaled|original|X_pca' not in adata3.obsm.keys():
            scale(adata3)
            pca(adata3,layer='scaled',n_pcs=n_pcs)
        sc.external.pp.harmony_integrate(adata3, batch_key,basis=use_rep,**kwargs)
        adata.obsm['X_harmony']=adata3.obsm['X_pca_harmony'].copy()
        del adata3
        add_reference(adata,'Harmony','batch correction with Harmony')
        #return adata3
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
        from ..externel.scanorama import integrate_scanpy
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
        from ..externel.cellanova.model import calc_ME,calc_BE,calc_TE
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
    else:
        print('Not supported')

    