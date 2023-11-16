#聚类
from sklearn.mixture import GaussianMixture
import scanpy as sc
import pandas as pd
import anndata

#初始化聚类位置，这个很重要
def get_initial_means(X, n_components,init_params, r):
    # Run a GaussianMixture with max_iter=0 to output the initialization means
    gmm = GaussianMixture(
        n_components=n_components, init_params=init_params, tol=1e-9, max_iter=0, random_state=r
    ).fit(X)
    return gmm.means_


def cluster(adata:anndata.AnnData,method:str='leiden',
            use_rep:str='X_pca',random_state:int=1024,
            n_components=None, **kwargs):
    
    if method=='leiden':
        sc.tl.leiden(adata,**kwargs)
    elif method=='louvain':
        sc.tl.louvain(adata,**kwargs)
    elif method=='GMM':
        if n_components is None:
            print('You need to input the `n_components` when methods is `GMM`')
            return
        print(f"""running GaussianMixture clustering""")
        data=adata.obsm[use_rep].copy()
        ini = get_initial_means(data,n_components, 'k-means++', 0)
        gmm = GaussianMixture(n_components = n_components,random_state=random_state,
                     means_init=ini, **kwargs)
        gmm.fit(data)
        adata.obs['gmm_cluster']=gmm.predict(data)
        adata.obs['gmm_cluster']=adata.obs['gmm_cluster'].astype(str)
        
        #new_num=adata.obs['gmm_cluster'].value_counts()[adata.obs['gmm_cluster'].value_counts()>10].shape[0]
        #adata.obs.loc[adata.obs['gmm_cluster'].isin(adata.obs['gmm_cluster'].value_counts()[adata.obs['gmm_cluster'].value_counts()<10].index.tolist()),'gmm_cluster']='-1'
        
        #adata.obs['gmm_cluster']=adata.obs['gmm_cluster'].astype('category')
        #adata.obs['gmm_cluster'].cat.categories=pd.Index(list(range(len(adata.obs['gmm_cluster'].cat.categories))))
        
        print(f"""finished: found {n_components} clusters and added
    'gmm_cluster', the cluster labels (adata.obs, categorical)""")
        
def filtered(adata:anndata.AnnData,
             cluster_key:str,
             cluster_minsize:int=10):
    new_num=adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].value_counts()<cluster_minsize].shape[0]
    adata.obs.loc[adata.obs[cluster_key].isin(adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].value_counts()<cluster_minsize].index.tolist()),cluster_key]='-1'
    adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
    print(f"""filtered {new_num} clusters and changed the cluster labels to '-1'(adata.obs, categorical)""")
        