---
name: scenic-gene-regulatory-network
title: SCENIC gene regulatory network
description: "SCENIC gene regulatory network: RegDiffusion GRN inference, cisTarget regulon pruning, AUCell scoring, RSS, regulon embeddings in OmicVerse."
---

# SCENIC Gene Regulatory Network Analysis

Use this skill when the user wants to infer transcription factor (TF) regulatory networks, identify regulons (TF + target gene sets), score regulon activity per cell, or find master regulators for specific cell types. SCENIC reconstructs gene regulatory networks from scRNA-seq data using a 3-stage pipeline.

## Overview: 3-Stage Pipeline

1. **GRN inference** — Predict TF → target gene links using RegDiffusion (deep learning, 10x faster than legacy GRNBoost2)
2. **Regulon pruning** — Validate links with cisTarget motif enrichment databases, keeping only direct targets
3. **AUCell scoring** — Quantify regulon activity per cell, enabling regulon-based clustering and cell type characterization

## Prerequisites

### Data requirements
- **Raw counts** (NOT log-transformed). RegDiffusion needs count-level variance structure.
- HVG-filtered to ~3000 genes for tractable runtime.
- Cell type annotations in `adata.obs` (for downstream RSS analysis).

### Database downloads (CRITICAL — most common failure point)

cisTarget ranking databases and motif annotations must be downloaded before analysis. These are species-specific and ~1-2 GB each.

**Mouse (mm10)**:
```bash
mkdir -p scenic_db
# Ranking databases (two resolution variants)
wget -P scenic_db/ https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/gene_based/mm10_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
wget -P scenic_db/ https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/gene_based/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
# Motif annotations
wget -P scenic_db/ https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl
```

**Human (hg38)**:
```bash
mkdir -p scenic_db
wget -P scenic_db/ https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
wget -P scenic_db/ https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
wget -P scenic_db/ https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
```

## Pipeline Steps

### 1. Initialize SCENIC

```python
import omicverse as ov
import glob

db_glob = 'scenic_db/*.feather'       # Glob pattern matching ranking DBs
motif_path = 'scenic_db/motifs-*.tbl' # Path to motif annotation file
# n_jobs: CPU workers for parallel processing. Match to available cores.
scenic = ov.single.SCENIC(adata, db_glob=db_glob, motif_path=motif_path, n_jobs=8)
```

### 2. GRN inference with RegDiffusion

```python
edgelist = scenic.cal_grn(method='regdiffusion', layer='counts')
# method: 'regdiffusion' (recommended, deep learning) or legacy methods
# layer: must contain RAW counts — not log-normalized. Use 'counts', 'raw_count', or 'X' if X has raw counts.
# Returns: DataFrame with columns [TF, target, importance]
```

### 3. Regulon discovery and AUCell scoring

```python
regulon_ad = scenic.cal_regulons(rho_mask_dropouts=True, seed=42)
# rho_mask_dropouts: ignore zero entries in correlation (handles dropout noise)
# Internally: builds co-expression modules → cisTarget motif pruning → AUCell activity scoring
# Typical compression: ~10k modules → ~70 regulons
```

After this step, `scenic` stores:
- `scenic.adjacencies` — TF→target edge list with importance scores
- `scenic.regulons` — list of Regulon objects (TF, target genes, weights)
- `scenic.auc_mtx` — cells × regulons activity matrix (AUCell scores)
- `scenic.modules` — raw co-expression modules before pruning

### 4. Downstream analysis

**Regulon Specificity Scores (RSS)** — identify master regulators per cell type:
```python
from pyscenic.utils import modules_to_regulons
from pyscenic.binarize import binarize
rss = ov.single.regulon_specificity_scores(scenic.auc_mtx, adata.obs['cell_type'])
# Returns: (cell_types × regulons) DataFrame, Jensen-Shannon divergence scores (0-1)
# Higher = more specific to that cell type
```

**Binary activity matrix** — convert continuous AUCell to on/off:
```python
binary_mtx, thresholds = binarize(scenic.auc_mtx, num_workers=8)
```

**Visualization**:
```python
# Regulon activity on embedding
ov.pl.embedding(regulon_ad, basis='X_umap', color=['Ets1(+)', 'E2f8(+)'])

# GRN network graph
ov.single.plot_grn(G, pos, tf_list, temporal_df, tf_gene_dict, top_tf_target_num=5)
```

## Critical API Reference

### `db_glob` must be a glob pattern matching .feather files

```python
# CORRECT — glob pattern
scenic = ov.single.SCENIC(adata, db_glob='scenic_db/*.feather', motif_path='scenic_db/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl', n_jobs=8)

# WRONG — single file path (needs at least one .feather match)
# scenic = ov.single.SCENIC(adata, db_glob='scenic_db/mm10_500bp.feather', ...)  # May work but misses second DB
```

### `layer` in cal_grn must contain RAW counts

```python
# CORRECT — specify raw count layer
edgelist = scenic.cal_grn(layer='counts')      # If raw counts in adata.layers['counts']
edgelist = scenic.cal_grn(layer='raw_count')   # Alternative layer name

# WRONG — using log-normalized data
# edgelist = scenic.cal_grn(layer='X')  # If X is log-normalized, GRN inference quality degrades severely
```

### Gene names must match the species database

Mouse databases expect mixed-case gene symbols (e.g., `Tp53`, `Cd4`). Human databases expect uppercase (e.g., `TP53`, `CD4`). A mismatch → most genes unmatched → very few regulons recovered.

## Defensive Validation Patterns

```python
import os, glob as gl

# Verify cisTarget databases exist
db_files = gl.glob(db_glob)
assert len(db_files) >= 1, f"No cisTarget .feather files found matching '{db_glob}'. Download databases first."

# Verify motif annotation file exists
assert os.path.isfile(motif_path), f"Motif file not found: {motif_path}. Download species-specific .tbl file."

# Verify raw counts (not log-transformed)
import numpy as np
if hasattr(adata.X, 'toarray'):
    max_val = adata.X.toarray().max()
else:
    max_val = adata.X.max()
if max_val < 20:
    print("WARNING: Max expression value is low — data may be log-transformed. SCENIC needs raw counts.")

# Verify gene name format matches database species
sample_genes = list(adata.var_names[:5])
if 'mm10' in db_glob or 'mgi' in motif_path:
    # Mouse DB expects mixed-case (Tp53)
    if all(g.isupper() for g in sample_genes):
        print("WARNING: Gene names are all uppercase but using mouse database. Check gene name format.")
elif 'hg38' in db_glob or 'hgnc' in motif_path:
    # Human DB expects uppercase (TP53)
    if any(g[0].isupper() and g[1:].islower() for g in sample_genes if len(g) > 1):
        print("WARNING: Gene names look like mouse format but using human database.")
```

## Troubleshooting

- **`No ranking databases found`**: The `db_glob` pattern doesn't match any `.feather` files. Check the path and ensure databases are downloaded. Use `glob.glob(db_glob)` to debug.
- **`Empty regulons (0 regulons after pruning)`**: Usually means gene names don't match the database species. Mouse genes are mixed-case (Actb), human are uppercase (ACTB). Also check: are you using raw counts?
- **`Log-transformed data passed to RegDiffusion`**: RegDiffusion needs variance structure from raw counts. If `adata.X.max() < 20`, the data is likely log-transformed. Use `adata.layers['counts']` or expm1 to recover raw counts.
- **`MemoryError during regulon pruning`**: cisTarget enrichment is memory-intensive. Reduce HVG count (2000 instead of 3000) or increase swap. Also try reducing `n_jobs`.
- **`Low regulon recovery rate (10k modules → <10 regulons)`**: Pruning thresholds may be too strict. Adjust `cal_regulons(thresholds=(0.5, 0.75))` for more permissive filtering. Or increase `top_n_targets=(100,)`.
- **`GUROBI license required` (CEFCON only)**: CEFCON's integer linear programming prefers GUROBI (academic license free). Fallback to SCIP is slower but works without license.

## Examples
- "Run SCENIC on my mouse hematopoiesis data to find master regulators per cell type."
- "Infer gene regulatory networks from scRNA-seq and visualize TF-target relationships."
- "Score regulon activity per cell and identify cell-type-specific transcription factors."

## References
- Tutorial: `t_scenic.ipynb`
- Quick copy/paste commands: [`reference.md`](reference.md)
