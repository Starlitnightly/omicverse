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
    assert adata.layer['scaled'] is not None


def test_anno():
    adata=ov.utils.pancreas()
    adata=ov.single.scanpy_lazy(adata)
    scsa=ov.single.pySCSA(adata=adata,
                          foldchange=1.5,
                          pvalue=0.01,
                          celltype='normal',
                          target='cellmarker',
                          tissue='All',
    )
    anno=scsa.cell_anno(clustertype='leiden',
               cluster='all',rank_rep=True)
    assert anno is not None

def test_metatime():
    adata=ov.utils.pancreas()
    adata=ov.single.scanpy_lazy(adata)
    TiME_object=ov.single.MetaTiME(adata,mode='table')
    TiME_object.overcluster(resolution=8,clustercol = 'overcluster',)
    TiME_object.predictTiME(save_obs_name='MetaTiME')
    assert TiME_object.adata.obs['MetaTiME'].shape[0]==adata.shape[0]



