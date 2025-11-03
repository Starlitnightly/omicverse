# ComBat batch correction quick commands

```python
import pandas as pd
import anndata
import omicverse as ov
import matplotlib.pyplot as plt

ov.plot_set()  # use ov.ov_plot_set() on older releases

# load batches (replace with user files)
dataset_1 = pd.read_pickle('data/combat/GSE18520.pickle')
dataset_2 = pd.read_pickle('data/combat/GSE66957.pickle')
dataset_3 = pd.read_pickle('data/combat/GSE69428.pickle')

adata1 = anndata.AnnData(dataset_1.T)
adata1.obs['batch'] = '1'
adata2 = anndata.AnnData(dataset_2.T)
adata2.obs['batch'] = '2'
adata3 = anndata.AnnData(dataset_3.T)
adata3.obs['batch'] = '3'

adata = anndata.concat([adata1, adata2, adata3], merge='same')

ov.bulk.batch_correction(adata, batch_key='batch')

raw = adata.to_df().T
corrected = adata.to_df(layer='batch_correction').T
raw.to_csv('raw_data.csv')
corrected.to_csv('removing_data.csv')
adata.write_h5ad('adata_batch.h5ad', compression='gzip')

adata.layers['raw'] = adata.X.copy()
ov.pp.pca(adata, layer='raw', n_pcs=50)
ov.pp.pca(adata, layer='batch_correction', n_pcs=50)

ov.utils.embedding(adata, basis='raw|original|X_pca', color='batch', frameon='small')
ov.utils.embedding(adata, basis='batch_correction|original|X_pca', color='batch', frameon='small')
```

```python
# boxplot comparison
color_dict = {
    '1': ov.utils.red_color[1],
    '2': ov.utils.blue_color[1],
    '3': ov.utils.green_color[1],
}
fig, ax = plt.subplots(figsize=(20, 4))
bp = plt.boxplot(adata.to_df().T, patch_artist=True)
for i, batch in zip(range(adata.shape[0]), adata.obs['batch']):
    bp['boxes'][i].set_facecolor(color_dict[batch])
ax.axis(False)
plt.show()

fig, ax = plt.subplots(figsize=(20, 4))
bp = plt.boxplot(adata.to_df(layer='batch_correction').T, patch_artist=True)
for i, batch in zip(range(adata.shape[0]), adata.obs['batch']):
    bp['boxes'][i].set_facecolor(color_dict[batch])
ax.axis(False)
plt.show()
```
