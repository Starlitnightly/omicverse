import pytest 
import omicverse as ov



def test_pp():
    adata=ov.utils.pancreas()
    adata=ov.pp.qc(adata,
              tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
    ov.utils.store_layers(adata,layers='counts')
    adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
    ov.pp.scale(adata)
    #ov.pp.pca(adata,layer='scaled',n_pcs=50)
    assert adata.layers['scaled'] is not None

