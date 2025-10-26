# Representative commands

## PAGA
```python
import scanpy as sc
adata = sc.read_h5ad("t_cells.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)
sc.tl.leiden(adata, resolution=1.0)
sc.tl.paga(adata, groups="leiden")
sc.pl.paga(adata, color=["leiden", "geneX"])
```

## Palantir
```python
import palantir
ms_data = palantir.utils.prepare_pca(adata.X, adata.obsm["X_pca"], adata.obs_names)
pr_res = palantir.utils.run_diffusion_maps(ms_data, n_components=30)
start_cells = ["cellA"]
pr_res = palantir.core.run_palantir(pr_res, start_cells=start_cells)
pr_res.pseudotime.to_csv("palantir_pseudotime.csv")
pr_res.entropy.plot()
```

## VIA
```python
from via import VIA
v0 = VIA(adata.obsm["X_pca"], labels=adata.obs["leiden"].values, knn=30)
v0.run_VIA()
adata.obs["via_pseudotime"] = v0.single_cell_pt_markov
v0.plot_scatter(color="via_pseudotime")
```

## scVelo velocity + VIA coupling
```python
import scvelo as scv
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata, n_pcs=40, n_neighbors=30)
scv.tl.velocity(adata, mode="stochastic")
scv.tl.velocity_graph(adata)
scv.tl.latent_time(adata)
v1 = VIA(adata.obsm["X_pca"], labels=adata.obs["leiden"].values,
         velocity_weight=adata.layers["velocity"], knn=30)
v1.run_VIA()
```

## CytoTRACE notebook snippet
```python
from cytotrace import cytotrace
scores = cytotrace.score_cells(adata)
adata.obs["cytotrace"] = scores
scores.to_csv("cytotrace_scores.csv")
```
