from scipy.sparse import csr_matrix,issparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def CAST(adata,sample_key=None,basis='spatial',layer='norm_1e4',
        output_path = 'output/CAST_Mark',gpu_t=0,device='cuda:0',**kwargs):
    if issparse(adata.obsm[basis]):
        adata.obsm[basis]=adata.obsm[basis].toarray()
    adata.obs['x'] = adata.obsm[basis][:,0]
    adata.obs['y'] = adata.obsm[basis][:,1]

    
    # Get the coordinates and expression data for each sample
    samples = np.unique(adata.obs[sample_key]) # used samples in adata
    coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs[sample_key] == sample_t] for sample_t in samples}
    exp_dict = {sample_t: adata[adata.obs[sample_key] == sample_t].layers[layer] for sample_t in samples}

    
    os.makedirs(output_path, exist_ok=True)
    
    from ..externel.CAST import CAST_MARK
    embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,gpu_t=gpu_t,device=device,**kwargs)

    
    adata.obsm['X_cast']=np.zeros((adata.shape[0],512))
    
    adata.obsm['X_cast']=pd.DataFrame(adata.obsm['X_cast'],index=adata.obs.index)
    for key in tqdm(embed_dict.keys()):
        adata.obsm['X_cast'].loc[adata.obs[sample_key]==key]+=embed_dict[key].cpu().numpy()
    adata.obsm['X_cast']=adata.obsm['X_cast'].values
    print('CAST embedding is saved in adata.obsm[\'X_cast\']')
    #adata.obs['cast_clusters']=adata.obs['cast_clusters'].astype('category')