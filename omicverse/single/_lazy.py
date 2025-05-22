from ..pp import *
import scanpy as sc
import numpy as np

def lazy(adata,
         species='human',
         reforce_steps=[],
         sample_key=None,
         qc_kwargs=None,
         preprocess_kwargs=None,
         pca_kwargs=None,
         harmony_kwargs=None,
         scvi_kwargs=None,
         ):
    """
    This is a very interesting function. We can use this function to avoid many unnecessary steps.

    arguments:
        adata: the data to analysis
        reforce_steps: we can reforce run lazy step, because some step have been run and will be skipped. 
                        ['qc','pca','preprocess','scaled','Harmony','scVI','eval_bench','eval_clusters']
        sample_key: the key store in `adata.obs` to batch correction.
    
    """
    #step 1: qc:
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if ('qc' in adata.uns['status'].keys() and adata.uns['status']['qc'] == False) or ('qc' in reforce_steps):
        print('❌ QC step didn\'t start, we will start it now')
        if qc_kwargs is None:
            qc_kwargs = {
                'tresh': {'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250},
                'doublets_method': 'scrublet',
                'batch_key': sample_key
            }
        print(f'🔧 The argument of qc we set '
              f'   mito_perc: {qc_kwargs["tresh"]["mito_perc"]} '
              f'   nUMIs: {qc_kwargs["tresh"]["nUMIs"]} '
              f'   detected_genes: {qc_kwargs["tresh"]["detected_genes"]}'
              f'   doublets_method: {qc_kwargs["doublets_method"]}'
              f'   batch_key: {qc_kwargs["batch_key"]}'
              )
        adata = qc(adata,
                **qc_kwargs)
    else:
        print('✅ QC step already finished, skipping it')
        print(f'🔧 The argument of qc we set '
              f'   mito_perc: {adata.uns["status_args"]["qc"]["mito_perc"]} '
              f'   nUMIs: {adata.uns["status_args"]["qc"]["nUMIs"]} '
              f'   detected_genes: {adata.uns["status_args"]["qc"]["detected_genes"]}'
              f'   doublets_method: {adata.uns["status_args"]["qc"]["doublets_method"]}'
              f'   batch_key: {adata.uns["status_args"]["qc"]["batch_key"]}'
              )

    #step 2: normalization and highly variable genes:
    if ('preprocess' in adata.uns['status'].keys() and adata.uns['status']['preprocess'] == False)  or ('preprocess' in reforce_steps):
        print('❌ Preprocess step didn\'t start, we will start it now')
        if preprocess_kwargs is None:
            preprocess_kwargs = {
                'mode': 'shiftlog|pearson',
                'n_HVGs': 2000,
                'target_sum': 50*1e4
            }
        print(f'🔧 The argument of preprocess we set '
              f'   mode: {preprocess_kwargs["mode"]} '
              f'   n_HVGs: {preprocess_kwargs["n_HVGs"]} '
              f'   target_sum: {preprocess_kwargs["target_sum"]} '
              )
        adata = preprocess(adata,**preprocess_kwargs)
    else:
        print('✅ Preprocess step already finished, skipping it')
        print(f'🔧 The argument of preprocess in data'
              f'   mode: {adata.uns["status_args"]["preprocess"]["mode"]} '
              f'   n_HVGs: {adata.uns["status_args"]["preprocess"]["n_HVGs"]} '
              f'   target_sum: {adata.uns["status_args"]["preprocess"]["target_sum"]} '
              )
        
    if ('scaled' in adata.uns['status'].keys() and adata.uns['status']['scaled'] == False)  or ('scaled' in reforce_steps):
        print('❌ Scaled step didn\'t start, we will start it now')
        scale(adata)
    else:
        print('✅ Scaled step already finished, skipping it')
    
    #step 3: PCA:
    if ('pca' in adata.uns['status'].keys() and adata.uns['status']['pca'] == False)  or ('pca' in reforce_steps):
        print('❌ PCA step didn\'t start, we will start it now')
        if pca_kwargs is None:
            pca_kwargs = {
                'layer':'scaled',
                'n_pcs':50,
                'use_highly_variable': True,
            }
        if ('highly_variable' not in adata.var.columns) and ('highly_variable_features' in adata.var.columns):
            adata.var['highly_variable'] = adata.var['highly_variable_features'].tolist()
        print(f'🔧 The argument of PCA we set '
              f'   layer: {pca_kwargs["layer"]} '
              f'   n_pcs: {pca_kwargs["n_pcs"]} '
              f'   use_highly_variable: {pca_kwargs["use_highly_variable"]} '
              )
        pca(adata,**pca_kwargs)
        adata.obsm['X_pca']=adata.obsm["scaled|original|X_pca"]
    else:
        print('✅ PCA step already finished, skipping it')

    #step 4 Score cell cycle:
    if ('cell_cycle' in adata.uns['status'].keys() and adata.uns['status']['cell_cycle'] == False)  or ('cell_cycle' in reforce_steps):
        print('❌ Cell cycle scoring step didn\'t start, we will start it now')
        score_genes_cell_cycle(adata,species=species)
    else:
        print('✅ Cell cycle scoring step already finished, skipping it')

    #step 5 batch remove:
    adata_hvg=adata.copy()
    if 'highly_variable_features' in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable_features]
    elif 'highly_variable' in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable]
    from ..single import batch_correction
    if ('X_harmony' not in adata.obsm.keys()) or ('Harmony' in reforce_steps):
        print('❌ Batch Correction: `Harmony` step didn\'t start, we will start it now')
        if harmony_kwargs is None:
            harmony_kwargs={
                'n_pcs':50
            }
        batch_correction(
            adata_hvg,
            batch_key=sample_key,
            methods='harmony',
            **harmony_kwargs
        )
        adata.obsm['X_harmony']=adata_hvg.obsm['X_harmony']
    else:
        print('✅ Batch Correction: `Harmony` step already finished, skipping it')
    
    if ('X_scVI' not in adata.obsm.keys())  or ('scVI' in reforce_steps):
        print('❌ Batch Correction: `scVI` step didn\'t start, we will start it now')
        if scvi_kwargs is None:
            scvi_kwargs={
                'n_layers':2, 
                'n_latent':30, 
                'gene_likelihood':"nb"
            }
        batch_correction(
            adata_hvg,
            batch_key=sample_key,
            methods='scVI',
            **scvi_kwargs
        )
        adata.obsm['X_scVI']=adata_hvg.obsm['X_scVI']
    else:
        print('✅ Batch Correction: `scVI` step already finished, skipping it')
    del adata_hvg

    if ('bench_best_res' not in adata.uns.keys()) or ('eval_bench' in reforce_steps):
        print('❌ Best Bench Correction Eval step didn\'t start, we will start it now')
        from scib_metrics.benchmark import Benchmarker
        
        emb_keys=["X_harmony",'X_scVI']
        bm = Benchmarker(
            adata,
            batch_key=sample_key,
            label_key="phase",
            embedding_obsm_keys=emb_keys,
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=-1,
        )
        bm.benchmark()
        bench_res = bm.get_results(min_max_scale=False)
        adata.uns['bench_res']=bench_res.loc[emb_keys]
        adata.uns['bench_best_res']=bench_res.loc[emb_keys,'Batch correction'].sort_values().index[-1]
        for col in adata.uns['bench_res']:
            adata.uns['bench_res'][col]=adata.uns['bench_res'][col].astype(float)
        import matplotlib.pyplot as plt
        bm.plot_results_table(min_max_scale=False,show=False)
        print(f'The Best Bench Correction Method is {adata.uns["bench_best_res"]}')
        print("We can found it in `adata.uns['bench_best_res']`")
    else:
        print('✅ Best Bench Correction Eval step already finished, skipping it')

    #step 6 clusters:
    if ('best_clusters' not in adata.obs.columns)  or ('eval_clusters' in reforce_steps):
        print('❌ Best Clusters step didn\'t start, we will start it now')
        method_test=adata.uns['bench_best_res']
        print(f"Automatic clustering using sccaf")
        print(f"Dimensionality using :{method_test}")
        mde(adata,embedding_dim=2,n_neighbors=15, basis='X_mde',
                    n_pcs=30, use_rep=adata.uns['bench_best_res'],)
        #预聚类
        print(f"Automatic clustering using leiden for preprocessed")
        sc.tl.leiden(adata, resolution=1.5, key_added = 'leiden_r1.5')
        #self.adata.obs['L1_Round0'] = self.adata.obs['leiden_r1.5']
        adata.obs['L1_result_smooth']=adata.obs['leiden_r1.5']
        #自动聚类
        for idx in range(10):
            if (np.unique(len(adata.obs['L1_result_smooth'].tolist()))>3) and idx>0:
                break
            else:
                adata.obs['L1_Round0']=adata.obs['L1_result_smooth']
                print(f"Automatic clustering using sccaf, Times: {idx}")
                from ..single import SCCAF_optimize_all
                SCCAF_optimize_all(min_acc=0.95, ad=adata, classifier='RF',n_jobs=4,
                                        use=adata.uns['bench_best_res'], basis ='X_mde',
                                        method='leiden',prefix='L1',plot=True)
                #平滑聚类效果
                print(f"Smoothing the effect of clustering, Times: {idx}")
                adata.obs['L1_result_smooth'] = adata.obs['L1_result'].tolist()
                
        #获取最佳聚类
        adata.obs['best_clusters']=adata.obs['L1_result_smooth'].copy()
    else:
        print('✅ Best Clusters step already finished, skipping it')

    #step 7 anno celltype automatically:


