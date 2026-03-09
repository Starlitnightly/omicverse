---
name: cellfate-pseudotime-gene-analysis
title: CellFateGenie pseudotime gene analysis
description: "CellFateGenie: Adaptive Threshold Regression for pseudotime-associated gene discovery, Mellon density, lineage scoring via ov.single.Fate."
---

# CellFateGenie Analysis

Use this skill when the user wants to identify genes that drive cell fate decisions along a developmental trajectory. CellFateGenie discovers pseudotime-associated genes using adaptive ridge regression and then scores lineage-specific fate-driving genes via manifold density estimation.

This skill is used **after trajectory inference** — the user must already have pseudotime values computed (e.g., from Palantir, VIA, or diffusion pseudotime).

## Overview

CellFateGenie answers: "Which genes change most significantly along pseudotime, and which are specifically driving a particular lineage?" It works in two phases:

1. **Gene selection** — Adaptive Threshold Regression (ATR) iteratively removes low-coefficient genes while monitoring R² to find the minimal gene set that explains pseudotime
2. **Lineage scoring** — Mellon density estimation on the manifold identifies low-density transition regions, and lineage-specific variability scoring pinpoints fate-driving genes

## Prerequisites

- **Pseudotime**: Must exist as a column in `adata.obs`. Compute first using Palantir, VIA, DPT, or any trajectory method.
- **Mellon** (optional but important): `pip install mellon` for density estimation. Without it, `low_density()` will fail.
- **Expression data**: Works on the `.X` matrix. Log-normalized data is fine (unlike SCENIC which needs raw counts).

## Pipeline Steps

### 1. Initialize Fate object

```python
import omicverse as ov

# pseudotime: column name in adata.obs containing pseudotime values
fate = ov.single.Fate(adata, pseudotime='dpt_pseudotime')
# Automatically uses GPU (PyTorchRidge) if CUDA available, else sklearn Ridge on CPU
```

### 2. Initial ridge regression (model_init)

```python
coef_df = fate.model_init(
    test_size=0.3,           # Train/test split ratio
    alpha=0.1,               # Ridge regularization strength
    use_data_augmentation=False,  # Enable for noisy pseudotime
)
# Returns: DataFrame of gene coefficients
# Stores: fate.coef (all coefficients), fate.raw_r2, fate.raw_mse
```

### 3. Adaptive Threshold Regression (ATR) — feature selection

This is the core innovation. ATR iteratively removes genes with the smallest coefficients and monitors when R² starts dropping significantly:

```python
threshold_df = fate.ATR(
    test_size=0.4,
    alpha=0.1,
    stop=500,    # Maximum iterations. Increase for more genes (default 100).
    flux=0.01,   # R² drop tolerance. When R² drops by more than flux from max, stop.
)
# Sets fate.coef_threshold internally
# Visualize the filtering curve:
fate.plot_filtering()  # Shows R² vs iteration, marks optimal threshold
```

### 4. Refit on selected genes (model_fit)

```python
filter_coef_df = fate.model_fit(
    test_size=0.3,
    alpha=0.1,
)
# Returns: DataFrame of coefficients for genes above threshold only
# Stores: fate.filter_coef
# Compare: fate.get_r2('raw') vs fate.get_r2('filter') — filter R² should be close to raw
```

### 5. Statistical validation (kendalltau_filter)

```python
kendall_df = fate.kendalltau_filter()
# Computes Kendall's tau rank correlation for each filtered gene vs pseudotime
# Returns: DataFrame with kendalltau_sta and pvalue per gene
# Confirms monotonic relationship — genes with high |tau| are truly pseudotime-associated
```

### 6. Mellon density estimation (low_density)

```python
fate.low_density(
    n_components=10,     # Diffusion map components for manifold representation
    knn=30,              # k-nearest neighbors for density estimation
    alpha=0.0,           # Mellon regularization
    seed=0,
    pca_key='X_pca',     # PCA embedding to use
)
# Stores: adata.obs['mellon_log_density_lowd']
# Low-density regions = developmental transition points (branching, commitment)
```

### 7. Lineage-specific scoring (lineage_score)

```python
fate.lineage_score(
    cluster_key='leiden',              # Clustering column in adata.obs
    lineage=['20', '17'],              # Cluster labels defining the lineage of interest
    cell_mask='specification',         # How to select cells: 'specification' uses lineage list
    density_key='mellon_log_density_lowd',
)
# Stores: adata.var['change_scores_lineage']
# High scores = genes with high expression variability specifically in that lineage
```

### 8. Identify fate-driving genes

```python
# Intersect ATR-selected genes with lineage-specific scores
fate_genes = adata.var.loc[fate.filter_coef.index, 'change_scores_lineage']
top_fate_genes = fate_genes.sort_values(ascending=False).head(20)
print(top_fate_genes)
```

## Data Augmentation

For noisy pseudotime estimates, enable augmentation to improve robustness:

```python
fate.model_init(
    use_data_augmentation=True,
    augmentation_strategy='jitter_pseudotime_noise',  # or 'gene_expression_noise', 'both'
    augmentation_intensity=0.05,  # Noise magnitude (fraction of range)
)
# Same parameters available in ATR() and model_fit()
```

## ATAC Mode

CellFateGenie also works with scATAC-seq data:

```python
fate.atac_init(...)          # Initialize for ATAC peak data
fate.get_related_peak(...)   # Find peaks associated with fate genes
```

## Visualization

```python
# ATR filtering curve — shows R² vs iteration
fate.plot_filtering(figsize=(3, 3))

# Model fit quality
fate.plot_fitting(type='raw')     # All genes
fate.plot_fitting(type='filter')  # ATR-selected genes only

# Color-coded by cluster
fate.plot_color_fitting(type='filter', cluster_key='leiden')
```

## Critical API Reference

### Pseudotime column must exist in adata.obs

```python
# CORRECT
fate = ov.single.Fate(adata, pseudotime='dpt_pseudotime')

# WRONG — column doesn't exist
# fate = ov.single.Fate(adata, pseudotime='pseudotime')  # KeyError if not in adata.obs
```

### ATR flux controls sensitivity

The `flux` parameter (default 0.01) determines when ATR stops removing genes. Lower flux = more genes retained (stricter R² preservation). Higher flux = fewer genes (more aggressive filtering).

### low_density requires mellon package

```python
# WRONG — mellon not installed
# fate.low_density()  # ImportError: No module named 'mellon'

# FIX
# pip install mellon
```

## Defensive Validation Patterns

```python
# Verify pseudotime column exists
assert pseudotime_col in adata.obs.columns, \
    f"Pseudotime column '{pseudotime_col}' not in adata.obs. Compute trajectory first."

# Verify pseudotime has valid values (no NaN)
import numpy as np
assert not adata.obs[pseudotime_col].isna().any(), \
    f"Pseudotime column contains NaN. Filter cells or impute missing values."

# Verify mellon is installed (before low_density)
try:
    import mellon
except ImportError:
    print("WARNING: mellon not installed. Run: pip install mellon")

# Verify PCA exists (needed for low_density)
assert 'X_pca' in adata.obsm, "PCA required for low_density(). Run ov.pp.pca(adata) first."

# Verify lineage clusters exist (before lineage_score)
for cl in lineage_list:
    assert cl in adata.obs[cluster_key].values, \
        f"Cluster '{cl}' not found in adata.obs['{cluster_key}']. Available: {adata.obs[cluster_key].unique()}"
```

## Troubleshooting

- **`KeyError` on pseudotime column**: The column name passed to `Fate()` doesn't exist in `adata.obs`. Check with `adata.obs.columns.tolist()`.
- **`ImportError: No module named 'mellon'`**: Install with `pip install mellon`. This is required for `low_density()` but not for ATR/model_fit.
- **Low R² after ATR (<0.3)**: Pseudotime may not be well-correlated with gene expression. Try a different trajectory method or increase `stop` iterations.
- **`plot_filtering()` shows flat curve**: The dataset may have too few variable genes. Ensure HVG selection was done before CellFateGenie.
- **GPU out of memory**: CellFateGenie uses PyTorchRidge when CUDA is available. For large datasets, it falls back to CPU automatically, but you can force CPU by setting `CUDA_VISIBLE_DEVICES=""`.
- **`lineage_score` returns all zeros**: The specified lineage clusters may have too few cells. Check cluster sizes with `adata.obs[cluster_key].value_counts()`.

## Examples
- "Find genes driving erythroid differentiation along my Palantir pseudotime."
- "Run CellFateGenie to identify fate-associated genes and plot the ATR filtering curve."
- "Score lineage-specific genes for the monocyte branch in my trajectory."

## References
- Tutorials: `t_cellfate.ipynb`, `t_cellfate_gene.ipynb`, `t_cellfate_genesets.ipynb`
- Quick copy/paste commands: [`reference.md`](reference.md)
