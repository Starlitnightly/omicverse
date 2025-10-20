# Command snippets by analysis

## AUCell
```python
# score a single pathway
geneset_name = 'response to vitamin (GO:0033273)'
ov.single.geneset_aucell(
    adata,
    geneset_name=geneset_name,
    geneset=pathway_dict[geneset_name],
)
sc.pl.embedding(adata, basis='umap', color=[f"{geneset_name}_aucell"])

# score multiple pathways
geneset_names = ['response to vitamin (GO:0033273)', 'response to vitamin D (GO:0033280)']
ov.single.pathway_aucell(
    adata,
    pathway_names=geneset_names,
    pathways_dict=pathway_dict,
)
sc.pl.embedding(adata, basis='umap', color=[f"{name}_aucell" for name in geneset_names])

# library-wide enrichment and export
adata_aucs = ov.single.pathway_aucell_enrichment(
    adata,
    pathways_dict=pathway_dict,
    num_workers=8,
)
adata_aucs.write_h5ad('data/pancreas_aucell.h5ad', compression='gzip')
```

## scRNA-seq DEG (meta-cell)
```python
# preprocessing
adata = ov.pp.qc(adata, tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)

# bulk-style DEG on selected cells
dds = ov.bulk.pyDEG(test_adata.to_df(layer='lognorm').T)
dds.plot_volcano(title='DEG Analysis', figsize=(4, 4), plot_genes_num=8)
dds.plot_boxplot(
    genes=['Irx1', 'Adra2a'],
    treatment_groups=treatment_groups,
    control_groups=control_groups,
)

# metacell workflow
meta_obj = ov.single.MetaCell(adata, use_rep='scaled|original|X_pca', n_metacells=150, use_gpu=True)
dds_meta = ov.bulk.pyDEG(test_adata.to_df().T)
```

## scRNA-seq DEG (cell-type & composition)
```python
# differential expression
deg_obj = ov.single.DEG(
    adata,
    condition='condition',
    ctrl_group='Control',
    test_group='Salmonella',
    method='memento-de',
)
deg_obj.run(
    celltype_key='cell_label',
    celltype_group=['TA'],
    capture_rate=0.07,
    num_cpus=12,
    num_boot=5000,
)

# differential composition (scCODA)
dct_obj = ov.single.DCT(
    adata,
    condition='condition',
    ctrl_group='Control',
    test_group='Salmonella',
    cell_type_key='cell_label',
    method='sccoda',
    sample_key='batch',
)
dct_obj.model.plot_boxplots(...)

# Milo setup
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000, target_sum=50 * 1e4)
adata = adata[:, adata.var.highly_variable_features]
ov.single.batch_correction(adata, batch_key='batch', methods='harmony', n_pcs=50)
ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='X_harmony')
```

## scDrug
```python
# data preparation
adata = cnv.datasets.maynard2020_3k()
ov.utils.get_gene_annotation(adata, gtf='gencode.v43.basic.annotation.gtf.gz', gtf_by='gene_name')
ov.utils.download_GDSC_data()
ov.utils.download_CaDRReS_model()

# clustering resolution & persistence
adata, res, plot_df = ov.single.autoResolution(adata, cpus=4)
adata.write('scanpyobj.h5ad')

# drug response prediction
job = ov.single.Drug_Response(
    adata,
    scriptpath='CaDRReS-Sc',
    modelpath='models/',
    output='result',
)
```

## SCENIC
```python
# initialize SCENIC
scenic_obj = ov.single.SCENIC(
    adata=adata,
    db_glob='/path/to/scenic/databases/mm10/*.feather',
    motif_path='/path/to/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
    n_jobs=12,
)

# inspect and export
scenic_obj.auc_mtx.head()
ov.pl.embedding(regulon_ad, basis='X_draw_graph_fa', color=['cell_type_roughly', 'E2f8(+)'])
ov.utils.save(scenic_obj, 'results/scenic_obj.pkl')
regulon_ad.write('results/scenic_regulon_ad.h5ad')
```

## cNMF
```python
cnmf_obj = ov.single.cNMF(
    adata,
    components=np.arange(5, 11),
    n_iter=20,
    seed=14,
    num_highvar_genes=2000,
    output_dir='example_dg/cNMF',
    name='dg_cNMF',
)
cnmf_obj.factorize(worker_i=0, total_workers=2)
cnmf_obj.combine(skip_missing_files=True)
cnmf_obj.k_selection_plot()
cnmf_obj.consensus(k=selected_K, density_threshold=density_threshold, show_clustering=True)
result_dict = cnmf_obj.load_results(K=selected_K, density_threshold=density_threshold)
cnmf_obj.get_results(adata, result_dict)
```

## NOCD
```python
scbrca = ov.single.scnocd(adata)
scbrca.matrix_transform()
scbrca.matrix_normalize()
scbrca.GNN_configure()
scbrca.GNN_preprocess()
scbrca.GNN_model()
scbrca.GNN_result()
scbrca.GNN_plot()
scbrca.cal_nocd()
scbrca.calculate_nocd()
sc.pl.umap(scbrca.adata, color=['leiden', 'nocd'], wspace=0.4, palette=sc_color)
```

## Lazy pipeline
```python
ov.settings.cpu_gpu_mixed_init()
adata = ov.single.lazy(
    adata,
    species='mouse',
    reforce_steps=['qc', 'preprocess', 'scaled', 'pca', 'cell_cycle', 'Harmony', 'scVI', 'eval_bench', 'umap', 'tsne'],
    sample_key='batch',
    qc_kwargs={
        'tresh': {'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250},
        'doublets_method': 'scrublet',
        'batch_key': 'batch',
    },
)
ov.pl.embedding(adata, basis='X_umap', color=['leiden_clusters_L1', 'cell_label'])
html_report = ov.single.generate_scRNA_report(
    adata,
    species='mouse',
    sample_key='batch',
    output_path='scRNA_analysis_report.html',
)
ov.generate_reference_table(adata)
```
