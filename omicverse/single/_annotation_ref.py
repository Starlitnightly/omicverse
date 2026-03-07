from .._settings import Colors
import numpy as np
import anndata as ad
from anndata import AnnData
import scanpy as sc
from .._registry import register_function


@register_function(
    aliases=[
        "参考映射注释",
        "reference mapping annotation",
        "细胞类型参考映射注释",
        "cell type reference mapping annotation",
        "细胞类型图谱注释",
        "cell type atlas annotation",
        "AnnotationRef",
        "reference transfer annotation",
        "query to reference label transfer",
    ],
    category="single",
    description="Reference-based annotation transfer class that aligns query and reference data, integrates embeddings, and transfers cell-type labels via weighted kNN.",
    prerequisites={
        'optional_functions': ['pp.preprocess', 'single.batch_correction']
    },
    requires={
        'var': ['shared genes between query and reference'],
        'obs': ['reference celltype labels']
    },
    produces={
        'obsm': ['X_pca_harmony_anno', 'X_scVI_anno', 'X_scanorama_anno'],
        'obs': ['harmony_prediction', 'scVI_prediction', 'scanorama_prediction']
    },
    auto_fix='escalate',
    examples=[
        "ref_anno = ov.single.AnnotationRef(adata_query, adata_ref, celltype_key='celltype')",
        "ref_anno.train(method='harmony')",
        "adata_query = ref_anno.predict(method='harmony', n_neighbors=15)"
    ],
    related=[
        "single.Annotation",
        "single.batch_correction",
        "utils.weighted_knn_trainer",
        "utils.weighted_knn_transfer",
    ]
)
class AnnotationRef(object):
    """
    Reference-based annotation transfer helper.
    
    Parameters
    ----------
    adata_query : AnnData
        Initialization argument for `AnnotationRef`.
    adata_ref : AnnData
        Initialization argument for `AnnotationRef`.
    celltype_key : str, optional, default='celltype'
        Initialization argument for `AnnotationRef`.
    
    Returns
    -------
    None
        Creates a configured class instance.
    """
    def __init__(self, adata_query: AnnData, adata_ref: AnnData, celltype_key: str = 'celltype'):
        self.adata_query = adata_query
        self.adata_ref = adata_ref
        self.celltype_key = celltype_key

        #check var names 
        both_var_names = list(set(adata_query.var_names) & set(adata_ref.var_names))
        if len(both_var_names) == 0:
            raise ValueError("Query and reference adata have no common var names")
        else:
            self.adata_query = adata_query[:, both_var_names]
            self.adata_ref = adata_ref[:, both_var_names]

        #check values is log normalized or not
        if self.adata_query.X.max() < np.log1p(1e4):
            print(f"{Colors.WARNING}Query adata is log normalized{Colors.ENDC}")
            print(f"{Colors.WARNING}Please run `ov.pp.recover_counts` to recover the counts before concatenation{Colors.ENDC}")
        if self.adata_ref.X.max() < np.log1p(1e4):
            print(f"{Colors.WARNING}Reference adata is log normalized{Colors.ENDC}")
            print(f"{Colors.WARNING}Please run `ov.pp.recover_counts` to recover the counts before concatenation{Colors.ENDC}")
        
        
        self.adata_new=sc.concat(
            {'ref':self.adata_ref,
            'query':self.adata_query},
            label='integrate_batch'
        )
        print(f"Concatenated adata saved to self.adata_new")

    def preprocess(self,mode='shiftlog|pearson',n_HVGs=3000,batch_key='integrate_batch'):
        """
        Preprocess concatenated query+reference AnnData for integration.
        
        Parameters
        ----------
        mode : Any, optional, default='shiftlog|pearson'
            Input parameter for `preprocess`.
        n_HVGs : Any, optional, default=3000
            Input parameter for `preprocess`.
        batch_key : Any, optional, default='integrate_batch'
            Input parameter for `preprocess`.
        
        Returns
        -------
        Any
            Output produced by `preprocess`.
        """
        from ..pp._preprocess import preprocess,scale,pca
        self.adata_new=preprocess(self.adata_new,mode=mode,
                       n_HVGs=n_HVGs,batch_key=batch_key)
        self.adata_new = self.adata_new[:, self.adata_new.var.highly_variable_features]
        scale(self.adata_new)
        pca(self.adata_new,layer='scaled',n_pcs=50)

    def train(
        self,method='harmony',
        **kwargs
    ):
        """
        Train/compute integrated embedding for transfer annotation.
        
        Parameters
        ----------
        method : Any, optional, default='harmony'
            Input parameter for `train`.
        **kwargs : Any
            Input parameter for `train`.
        
        Returns
        -------
        Any
            Output produced by `train`.
        """
        from ._batch import batch_correction
        if method=='harmony':
            batch_correction(self.adata_new,batch_key='integrate_batch',methods='harmony',**kwargs)
            self.adata_query.obsm['X_pca_harmony_anno']=self.adata_new[self.adata_query.obs.index].obsm['X_pca_harmony']
            self.adata_ref.obsm['X_pca_harmony_anno']=self.adata_new[self.adata_ref.obs.index].obsm['X_pca_harmony']
            print(f"Harmony integrated embeddings saved to self.adata_query.obsm['X_pca_harmony'] and self.adata_ref.obsm['X_pca_harmony']")
        elif method=='scVI':
            batch_correction(self.adata_new,batch_key='integrate_batch',methods='scVI',**kwargs)
            self.adata_query.obsm['X_scVI_anno']=self.adata_new[self.adata_query.obs.index].obsm['X_scVI']
            self.adata_ref.obsm['X_scVI_anno']=self.adata_new[self.adata_ref.obs.index].obsm['X_scVI']
            print(f"scVI integrated embeddings saved to self.adata_query.obsm['X_scVI'] and self.adata_ref.obsm['X_scVI']")
        elif method=='scanorama':
            batch_correction(self.adata_new,batch_key='integrate_batch',methods='scanorama',**kwargs)
            self.adata_query.obsm['X_scanorama_anno']=self.adata_new[self.adata_query.obs.index].obsm['X_scanorama']
            self.adata_ref.obsm['X_scanorama_anno']=self.adata_new[self.adata_ref.obs.index].obsm['X_scanorama']
            print(f"Scanorama integrated embeddings saved to self.adata_query.obsm['X_scanorama'] and self.adata_ref.obsm['X_scanorama']")
        else:
            raise ValueError(f"Unsupported method: {method}")
        return self.adata_query

    def predict(self,method='harmony',n_neighbors=15,pred_key=None,uncert_key=None):
        """
        Transfer reference labels to query cells using weighted kNN.
        
        Parameters
        ----------
        method : Any, optional, default='harmony'
            Input parameter for `predict`.
        n_neighbors : Any, optional, default=15
            Input parameter for `predict`.
        pred_key : Any, optional, default=None
            Input parameter for `predict`.
        uncert_key : Any, optional, default=None
            Input parameter for `predict`.
        
        Returns
        -------
        Any
            Output produced by `predict`.
        """
        if method=='harmony':
            if pred_key is None:
                pred_key='harmony_prediction'
            if uncert_key is None:
                uncert_key='harmony_uncertainty'
            emb_key='X_pca_harmony_anno'
        elif method=='scVI':
            if pred_key is None:
                pred_key='scVI_prediction'
            if uncert_key is None:
                uncert_key='scVI_uncertainty'
            emb_key='X_scVI_anno'
        elif method=='scanorama':
            if pred_key is None:
                pred_key='scanorama_prediction'
            if uncert_key is None:
                uncert_key='scanorama_uncertainty'
            emb_key='X_scanorama_anno'
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        from ..utils._knn import weighted_knn_trainer,weighted_knn_transfer
        knn_transformer=weighted_knn_trainer(
            train_adata=self.adata_ref,
            train_adata_emb=emb_key,
            n_neighbors=n_neighbors,
        )
        labels,uncert=weighted_knn_transfer(
            query_adata=self.adata_query,
            query_adata_emb=emb_key,
            label_keys=self.celltype_key,
            knn_model=knn_transformer,
            ref_adata_obs=self.adata_ref.obs,
        )
        self.adata_query.obs[pred_key]=labels.loc[self.adata_query.obs.index,self.celltype_key]
        self.adata_query.obs[uncert_key]=uncert.loc[self.adata_query.obs.index,self.celltype_key]
        print(f"{pred_key} saved to adata.obs['{pred_key}']")
        print(f"{uncert_key} saved to adata.obs['{uncert_key}']")
        return self.adata_query
    
    

    
