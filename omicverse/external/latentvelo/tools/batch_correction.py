"""
These are 'standard' batch correction methods to compare with.
"""
import scanpy as sc
import scvelo as scv
import numpy as np
import scipy as scp
import anndata as ad


def compute_scvi(adata_raw, counts_key = 'counts_spliced', batch_key = 'batch', n_latent=20, gene_likelihood='nb',
                 encode_covariates=True, max_epochs = 500, scanvi_epochs=20, n_top_genes = None, scANVI=False, labels_key = None, **kwargs):
    """
    Fit scVI and scANVI with mostly default settings
    """
    
    import scvi
    
    adata=adata_raw.copy()

    
    if n_top_genes != None:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=2000,
            layer=counts_key,
            batch_key=batch_key,
            subset=True
        )
    
    
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=counts_key,
        batch_key=batch_key
    )

    from tqdm import tqdm
    tqdm(disable=True, total=0) 

    model = scvi.model.SCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood, encode_covariates=encode_covariates, **kwargs)

    model.train(max_epochs=max_epochs)

    latent_scvi = model.get_latent_representation()
    expression_scvi = model.get_normalized_expression()

    if scANVI and labels_key != None:
    
        lvae = scvi.model.SCANVI.from_scvi_model(
            model,
            adata=adata,
            labels_key=labels_key,
            unlabeled_category="Unknown",
        )
        
        lvae.train(max_epochs=scanvi_epochs, n_samples_per_label=100)

        latent_scANVI = lvae.get_latent_representation(adata)
        expression_scANVI = lvae.get_normalized_expression()

        return expression_scvi, latent_scvi, expression_scANVI, latent_scANVI
    
    else:
        #adata.obsm["X_scVI"] = latent
        return expression_scvi, latent_scvi



def combat_velocity(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = 'batch', log=True):
    """Batch correct both U and S with ComBat for velocity, following Ranek 2022 and Hansen https://www.hansenlab.org/velocity_batch"""
    
    M = np.array(adata.layers[spliced_key].todense() + adata.layers[unspliced_key].todense())
    masked_M = (M>0)*M + (M==0)
    R = np.array(adata.layers[spliced_key].todense())/masked_M
        
    if log:
        M = np.log(1 + M)
        
    new_adata = ad.AnnData(M, obs=adata.obs)
    
    corrected_M = np.array(sc.pp.combat(new_adata, key=batch_key, inplace=False))
    
    if log:
        corrected_M = np.exp(corrected_M)-1
    
    corrected_S = corrected_M * R
    corrected_U = corrected_M * (1-R)

    adata.layers[spliced_key + 'b'] = corrected_S
    adata.layers[unspliced_key + 'b'] = corrected_U


def scgen_velocity(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = 'batch', cluster_key='celltype', log=True, normalize = True, n_latent=20, max_epochs=100):
    """Batch correct both U and S with scGen for velocity, following Ranek 2022 and Hansen https://www.hansenlab.org/velocity_batch"""
    
    import scgen
    
    M = np.array(adata.layers[spliced_key].todense() + adata.layers[unspliced_key].todense())
    masked_M = (M>0)*M + (M==0)
    R = np.array(adata.layers[spliced_key].todense())/masked_M
    
    if normalize:
        normalization = M.sum(1)[:,None]
        M = M/normalization
    
    if log:
        M = np.log(1 + M)
    
    train_adata = ad.AnnData(M, obs=adata.obs)
    train_adata.obs['batch'] = adata.obs[batch_key].values
    train_adata.obs['cell_type'] = train_adata.obs[cluster_key].values

    scgen.SCGEN.setup_anndata(train_adata, batch_key="batch", labels_key="cell_type")
    model = scgen.SCGEN(train_adata, n_latent=n_latent)

    model.train(
        max_epochs=max_epochs,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=25,
    )

    corrected_adata = model.batch_removal()
    corrected_M = corrected_adata.X
    
    if log:
        corrected_M = np.exp(corrected_M)-1

    if normalize:
        corrected_M = corrected_M * normalization
    
    corrected_S = corrected_M * R
    corrected_U = corrected_M * (1-R)

    adata.layers[spliced_key + 'b'] = corrected_S
    adata.layers[unspliced_key + 'b'] = corrected_U
    
