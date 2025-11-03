# OmicVerse visualization quick reference

```python
import omicverse as ov
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

ov.ov_plot_set()  # fall back to ov.plot_set() on older versions
```

## Bulk RNA-seq plots
```python
# venn diagram
sets = {
    'Control vs A': control_deg,
    'Control vs B': control_deg_b,
    'Control vs C': control_deg_c,
}
ov.pl.venn(sets=sets, palette=['#f59d7f', '#f0c14b', '#7db9de'])

# volcano plot
result = ov.read('data/dds_result.csv', index_col=0)
ov.pl.volcano(
    result,
    pval_name='qvalue',
    fc_name='log2FoldChange',
    sig_pvalue=0.05,
    sig_fc=1.0,
    palette={'up': '#d62828', 'down': '#1d3557', 'stable': '#adb5bd'},
    annotate_top=10,
)

# box plot using seaborn tips dataset
data = sns.load_dataset('tips')
fig, ax = plt.subplots(figsize=(4, 4))
ov.pl.boxplot(
    data,
    x_value='day',
    y_value='total_bill',
    hue='sex',
    palette={'Male': '#4271ae', 'Female': '#ff9896'},
    ax=ax,
)
fig.tight_layout()
```

## Forbidden City color utilities
```python
fb = ov.pl.ForbiddenCity()
print(fb.available_names[:6])
royal_purple = fb.get_color(name='凝夜紫')
segment = ov.pl.get_cmap_seg(['#ef6f6c', '#f7c59f', '#458f69'], name='warm_to_green')

color_dict = {
    'Astrocytes': fb.get_color('石英粉红'),
    'Microglia': fb.get_color('胭脂紫'),
    'OPC': fb.get_color('藤黄'),
    'Neurons': fb.get_color('霁蓝'),
}
```

## Single-cell visualizations
```python
adata = ov.read('data/DentateGyrus/10X43_1.h5ad')
adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)

# optimize palette for clusters
palette = ov.pl.optim_palette(adata, basis='X_umap', colors='clusters')

# stacked proportions
fig, ax = plt.subplots(figsize=(3, 3))
ov.pl.cellproportion(
    adata,
    celltype_clusters='clusters',
    groupby='orig.ident',
    palette=palette,
    ax=ax,
)

# embedding with convex hulls
fig, ax = plt.subplots(figsize=(4, 4))
ov.pl.ConvexHull(
    adata,
    basis='X_umap',
    groupby='clusters',
    palette=palette,
    ax=ax,
)

# density overlay and gene density
ov.pl.embedding_density(adata, basis='X_umap', groupby='clusters', adjust=0.8)
ov.pl.calculate_gene_density(adata, genes=['Sox4'], basis='X_umap', key_added='Sox4_density')
ov.pl.embedding(
    adata,
    basis='X_umap',
    color='Sox4_density',
    layer='Sox4_density',
    cmap=segment,
    frameon=False,
)

# export figure
plt.savefig('omicverse_visualization.png', dpi=300, bbox_inches='tight')
```
