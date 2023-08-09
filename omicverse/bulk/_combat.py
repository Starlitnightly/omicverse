import anndata


def batch_correction(adata:anndata.AnnData,
                     batch_key=None,
                     key_added:str='batch_correction'):
    
    
    try:
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError(
            'Please install the combat: `pip install combat`.'
        )
    adata.layers[key_added]=pycombat(adata.to_df().T,adata.obs[batch_key].values).T
    print(f"Storing batch correction result in adata.layers['{key_added}']")