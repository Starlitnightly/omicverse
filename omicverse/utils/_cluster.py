#聚类
from sklearn.mixture import GaussianMixture
import scanpy as sc
import pandas as pd
import anndata

mira_install=False
def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

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
    elif method=='schist':
        try:
            import schist
        except ImportError:
            raise ImportError(
                'Please install the schist using conda `conda install -c conda-forge schist` \nor `pip install git+https://github.com/dawe/schist.git`'
            )
        schist.inference.nested_model(adata, **kwargs)

        
def filtered(adata:anndata.AnnData,
             cluster_key:str,
             cluster_minsize:int=10):
    new_num=adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].value_counts()<cluster_minsize].shape[0]
    adata.obs['gmm_cluster']=adata.obs['gmm_cluster'].astype(str)
    adata.obs.loc[adata.obs[cluster_key].isin(adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].value_counts()<cluster_minsize].index.tolist()),cluster_key]='-1'
    adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
    adata.obs['gmm_cluster'].cat.categories=pd.Index(list(range(len(adata.obs['gmm_cluster'].cat.categories))))
    print(f"""filtered {new_num} clusters and changed the cluster labels to '-1'(adata.obs, categorical)""")


class LDA_topic(object):

    def __init__(self,adata,feature_type='expression',
                  highly_variable_key='highly_variable_features',
                 layers='counts',batch_key=None,learning_rate=1e-3,ondisk=False):
        global mira_install
        try:
            import mira
            mira_install=True
            print('mira have been install version:',mira.__version__)
        except ImportError:
            raise ImportError(
                """Please install the mira: `conda install -c bioconda mira-multiome` or 
                `pip install mira-multiome`.'"""
            )
        if mira_install==True:
            global_imports("mira")
        self.adata=adata
        self.model = mira.topics.make_model(
            adata.n_obs, adata.n_vars, # helps MIRA choose reasonable values for some hyperparameters which are not tuned.
            feature_type = feature_type,
            highly_variable_key=highly_variable_key,
            counts_layer=layers,
            categorical_covariates=batch_key
        )
        self.ondisk=ondisk
        if self.ondisk!=True:
        
            self.model.get_learning_rate_bounds(adata)
            self.model.set_learning_rates(learning_rate, 0.25) # for larger datasets, the default of 1e-3, 0.1 usually works well.
        else:
            train, test = self.model.train_test_split(adata)
            import os 
            os.mkdir('topic_train')
            os.mkdir('topic_test')
            self.model.write_ondisk_dataset(train, dirname='topic_train')
            self.model.write_ondisk_dataset(test, dirname='topic_test')
            self.model.get_learning_rate_bounds('topic_train')
            self.model.set_learning_rates(learning_rate, 0.25) # for larger datasets, the default of 1e-3, 0.1 usually works well.

        self.model.plot_learning_rate_bounds(figsize=(6,3))
        
    def plot_topic_contributions(self,num_topics=6):
        NUM_TOPICS = num_topics
        if self.ondisk!=True:
            topic_contributions = mira.topics.gradient_tune(self.model, self.adata)
        else:
            topic_contributions = mira.topics.gradient_tune(self.model, 'topic_train')
        mira.pl.plot_topic_contributions(topic_contributions, NUM_TOPICS)
        
    def predicted(self,num_topics=6):
        print(f"""running LDA topic predicted""")
        self.model = self.model.set_params(num_topics = num_topics).fit(self.adata)
        self.model.predict(self.adata)
        # 找到每行中最大值所在的列（topic）
        df=self.adata.obs[self.model.topic_cols].copy()
        max_topic = df.idxmax(axis=1)
        # 将结果添加到DataFrame中
        self.adata.obs['LDA_cluster'] = max_topic
        print(f"""finished: found {num_topics} clusters and added
    'LDA_cluster', the cluster labels (adata.obs, categorical)""")