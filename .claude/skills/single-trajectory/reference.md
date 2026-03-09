# Trajectory analysis quick commands

## PAGA
```python
import scanpy as sc
import omicverse as ov
adata = ov.read("t_cells.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata)
# n_neighbors: controls local vs global topology. 15-30 is typical.
# Higher = smoother trajectories but may miss fine branches.
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)
sc.tl.leiden(adata, resolution=1.0)
sc.tl.paga(adata, groups="leiden")  # Computes cluster connectivity graph
sc.pl.paga(adata, color=["leiden", "geneX"])
# Use init_pos='paga' in umap/draw_graph for PAGA-initialized embedding
# sc.tl.umap(adata, init_pos='paga')
```

## Palantir
```python
import palantir
ms_data = palantir.utils.prepare_pca(adata.X, adata.obsm["X_pca"], adata.obs_names)
# n_components: diffusion components. 20-30 typical. More = captures fine structure but slower.
pr_res = palantir.utils.run_diffusion_maps(ms_data, n_components=30)
# start_cells: manually selected root cell(s). Use marker genes and velocity to confirm choice.
# Bad root selection → misleading pseudotime ordering.
start_cells = ["cellA"]
pr_res = palantir.core.run_palantir(pr_res, start_cells=start_cells)
pr_res.pseudotime.to_csv("palantir_pseudotime.csv")
pr_res.entropy.plot()  # Differentiation potential: high entropy = multipotent
```

## VIA
```python
from via import VIA
# knn: neighbor count for VIA's internal graph. Should match sc.pp.neighbors for consistency.
# root_user: provide a list of cluster labels (not cell barcodes) for root selection.
# Must be a value in the labels array, e.g., ['nIPC'] not ['ATCG...'].
v0 = VIA(adata.obsm["X_pca"], labels=adata.obs["leiden"].values, knn=30)
v0.run_VIA()
adata.obs["via_pseudotime"] = v0.single_cell_pt_markov  # Markov-chain pseudotime
v0.plot_scatter(color="via_pseudotime")
```

## scVelo velocity + VIA coupling
```python
import scvelo as scv
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata, n_pcs=40, n_neighbors=30)  # Must match PAGA/VIA neighbor settings
# mode='stochastic': faster, works for most datasets.
# mode='dynamical': slower but more accurate for complex dynamics.
scv.tl.velocity(adata, mode="stochastic")
scv.tl.velocity_graph(adata)
scv.tl.latent_time(adata)  # Compare with VIA pseudotime to validate directionality

# velocity_weight: incorporates RNA velocity into VIA's graph traversal.
# This refines lineage directionality — cells move "with" the velocity flow.
v1 = VIA(adata.obsm["X_pca"], labels=adata.obs["leiden"].values,
         velocity_weight=adata.layers["velocity"], knn=30)
v1.run_VIA()
```

## CytoTRACE differentiation potential
```python
from cytotrace import cytotrace
# CytoTRACE scores: high = more differentiated, low = more stem-like.
# Use to cross-validate with velocity-informed root selection.
scores = cytotrace.score_cells(adata)
adata.obs["cytotrace"] = scores
scores.to_csv("cytotrace_scores.csv")
```

## OmicVerse unified Velo class — dynamo backend
```python
import omicverse as ov

velo = ov.single.Velo(adata)
# recipe='monocle': standard Dynamo preprocessing with cell-cycle scoring
velo.preprocess(recipe='monocle', n_neighbors=30, n_pcs=30)
velo.moments(backend='dynamo', n_pcs=30, n_neighbors=30)
velo.dynamics(backend='dynamo')
velo.cal_velocity(method='dynamo')
velo.velocity_graph(basis='umap')
velo.velocity_embedding(basis='umap')
```

## OmicVerse unified Velo class — latentvelo backend
```python
import omicverse as ov

# latentvelo: VAE + neural ODE for latent-space velocity estimation
# Requires: pip install torchdiffeq
velo = ov.single.Velo(adata)
velo.filter_genes(min_shared_counts=20)
velo.moments(backend='scvelo', n_pcs=30, n_neighbors=30)
velo.cal_velocity(
    method='latentvelo',
    celltype_key='cell_type',    # AnnotVAE uses cell type for better training
    batch_key='batch',           # Optional batch correction
    velocity_key='velocity_S',
    n_top_genes=2000,
)
velo.velocity_graph(basis='umap', vkey='velocity_S')
velo.velocity_embedding(basis='umap', vkey='velocity_S')
```

## OmicVerse unified Velo class — graphvelo refinement
```python
import omicverse as ov

# graphvelo: refines velocity from any base method using graph structure
# Step 1: Compute base velocity with scvelo
velo = ov.single.Velo(adata)
velo.filter_genes(min_shared_counts=20)
velo.moments(backend='scvelo', n_pcs=30, n_neighbors=30)
velo.dynamics(backend='scvelo')
velo.cal_velocity(method='scvelo')

# Step 2: Refine with graphvelo
# xkey: spliced moments key (default 'Ms')
# vkey: base velocity to refine (default 'velocity_S')
# basis_keys: project refined velocity to these embeddings
velo.graphvelo(xkey='Ms', vkey='velocity_S', basis_keys=['X_umap', 'X_pca'])
```
