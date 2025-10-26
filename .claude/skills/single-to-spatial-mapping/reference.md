# Single2Spatial quick commands

```python
import pandas as pd
import anndata
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import omicverse as ov

ov.utils.ov_plot_set()

# Load processed PDAC matrices (replace with user paths)
sc_expr = pd.read_csv('data/pdac/sc_data.csv', index_col=0)
sc_meta = pd.read_csv('data/pdac/sc_meta.csv', index_col=0)

tag_expr = pd.read_csv('data/pdac/st_data.csv', index_col=0)
tag_meta = pd.read_csv('data/pdac/st_meta.csv', index_col=0)

single_data = anndata.AnnData(sc_expr.T)
single_data.obs = sc_meta[['Cell_type']].copy()

spatial_data = anndata.AnnData(tag_expr.T)
spatial_data.obs = tag_meta.copy()

st_model = ov.bulk2single.Single2Spatial(
    single_data=single_data,
    spatial_data=spatial_data,
    celltype_key='Cell_type',
    spot_key=['xcoord', 'ycoord'],
)

sp_adata = st_model.train(
    spot_num=500,
    cell_num=10,
    df_save_dir='data/pdac/predata_net/save_model',
    df_save_name='pdac_df',
    k=10,
    num_epochs=1000,
    batch_size=1000,
    predicted_size=32,
)

# To reuse saved weights later
sp_adata = st_model.load(
    modelsize=14478,
    df_load_dir='data/pdac/predata_net/save_model/pdac_df.pth',
    k=10,
    predicted_size=32,
)

sp_adata_spot = st_model.spot_assess()

sc.pl.embedding(
    sp_adata,
    basis='X_spatial',
    color=['REG1A', 'CLDN1', 'KRT16', 'MUC5B'],
    frameon=False,
    ncols=4,
    show=False,
)

sc.pl.embedding(
    sp_adata_spot,
    basis='X_spatial',
    color=['Acinar cells', 'Cancer clone A', 'Cancer clone B', 'Ductal'],
    frameon=False,
    ncols=4,
    show=False,
)

sc.pl.embedding(
    sp_adata,
    basis='X_spatial',
    color=['Cell_type'],
    frameon=False,
    palette=ov.utils.ov_palette()[11:],
    show=False,
)

sp_adata.write_h5ad('single2spatial_cells.h5ad', compression='gzip')
sp_adata_spot.write_h5ad('single2spatial_spots.h5ad', compression='gzip')
```
