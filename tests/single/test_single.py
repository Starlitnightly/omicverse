import pytest 
import omicverse as ov



def test_pp():
    adata=ov.datasets.pbmc3k()
    adata=ov.pp.qc(adata,
              tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
    adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
    ov.pp.scale(adata)
    #ov.pp.pca(adata,layer='scaled',n_pcs=50)
    assert adata.layers['scaled'] is not None

def test_pca(adata):
    ov.pp.pca(adata,layer='scaled',n_pcs=5)
    assert adata.obsm['scaled|original|X_pca'] is not None

def test_umap(adata):
    ov.pp.neighbors(adata,use_rep='scaled|original|X_pca')
    ov.pp.umap(adata)

    assert adata.obsm['X_umap'] is not None