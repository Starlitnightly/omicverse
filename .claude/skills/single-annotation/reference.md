# Quick reference: single-cell annotation workflows

## SCSA (pySCSA)
```python
import scanpy as sc
import omicverse as ov

adata = sc.read_10x_mtx('data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)
adata = ov.pp.qc(adata, tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)
ov.pp.scale(adata)
ov.pp.pca(adata, layer='scaled', n_pcs=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='scaled|original|X_pca')
sc.tl.leiden(adata)
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=False)

scsa = ov.single.pySCSA(
    adata,
    foldchange=1.5,
    pvalue=0.01,
    celltype='normal',
    target='cellmarker',
    tissue='All',
    model_path='temp/pySCSA_2024_v1_plus.db',
)
scsa.cell_auto_anno(adata, key='scsa_celltype_cellmarker')
```

## MetaTiME
```python
import scanpy as sc
import omicverse as ov

data_path = 'TiME_adata_scvi.h5ad'  # download from https://figshare.com/ndownloader/files/41440050
adata = sc.read(data_path)
sc.pp.neighbors(adata, use_rep='X_scVI')
adata.obsm['X_mde'] = ov.utils.mde(adata.obsm['X_scVI'])

TiME = ov.single.MetaTiME(adata, mode='table')
TiME.overcluster(resolution=8, clustercol='overcluster')
TiME.predictTiME(save_obs_name='MetaTiME')
```

## CellVote
```python
import anndata as ad
import omicverse as ov

adata = ad.read_h5ad('data/pbmc3k.h5ad')
ov.single.CellVote(adata).vote(
    clusters_key='leiden',
    cluster_markers=ov.single.get_celltype_marker(adata, clustertype='leiden', rank=True),
    celltype_keys=['scsa_annotation', 'gpt_celltype', 'gbi_celltype'],
    species='human',
    organization='PBMC',
    provider='openai',
    model='gpt-4o-mini',
)
```

## CellMatch / CellOntologyMapper
```bash
mkdir -p new_ontology
wget http://purl.obolibrary.org/obo/cl/cl.json -O new_ontology/cl.json
wget https://download.cncb.ac.cn/celltaxonomy/Cell_Taxonomy_resource.txt -O new_ontology/Cell_Taxonomy_resource.txt
```
```python
import pertpy as pt
import omicverse as ov

data = pt.dt.haber_2017_regions()
mapper = ov.single.CellOntologyMapper(
    cl_obo_file='new_ontology/cl.json',
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    local_model_dir='./my_models',
)
mapper.load_cell_taxonomy_resource(
    'new_ontology/Cell_Taxonomy_resource.txt',
    species_filter=['Homo sapiens', 'Mus musculus'],
)
mapper.map_adata_with_taxonomy(
    data,
    cell_name_col='cell_label',
    new_col_name='enhanced_cell_ontology',
    expand_abbreviations=True,
    use_taxonomy=True,
    species='Mus musculus',
    tissue_context='Gut',
    threshold=0.3,
)
```

## GPTAnno (remote or local LLM)
```python
import os
import omicverse as ov

os.environ['AGI_API_KEY'] = 'sk-your-key'
markers = ov.single.get_celltype_marker(
    adata,
    clustertype='leiden',
    rank=True,
    key='rank_genes_groups',
    foldchange=2,
    topgenenumber=5,
)
result = ov.single.gptcelltype(
    markers,
    tissuename='PBMC',
    speciename='human',
    model='qwen-plus',
    provider='qwen',
    topgenenumber=5,
)
adata.obs['gpt_celltype'] = adata.obs['leiden'].map({
    k: v.split(': ')[-1].split(' (')[0].split('. ')[1]
    for k, v in result.items()
})
```
```python
# Offline alternative
ov.single.gptcelltype_local(
    markers,
    tissuename='PBMC',
    speciename='human',
    model_name='~/models/Qwen2-7B-Instruct',
    topgenenumber=5,
)
```

## Weighted KNN label transfer
```python
import scanpy as sc
import omicverse as ov

rna = sc.read('data/analysis_lymph/rna-emb.h5ad')
atac = sc.read('data/analysis_lymph/atac-emb.h5ad')
knn_model = ov.utils.weighted_knn_trainer(rna, train_adata_emb='X_glue', n_neighbors=15)
labels, uncert = ov.utils.weighted_knn_transfer(
    query_adata=atac,
    query_adata_emb='X_glue',
    label_keys='major_celltype',
    knn_model=knn_model,
    ref_adata_obs=rna.obs,
)
atac.obs['transf_celltype'] = labels.loc[atac.obs.index, 'major_celltype']
atac.obs['transf_celltype_unc'] = uncert.loc[atac.obs.index, 'major_celltype']
```
