# CellFateGenie quick commands

## Full RNA pipeline

```python
import omicverse as ov
import scanpy as sc

ov.plot_set()

# --- Load preprocessed data with pseudotime ---
adata = ov.read("trajectory_data.h5ad")
# Pseudotime must be in adata.obs (e.g., from Palantir, VIA, DPT)

# --- Initialize ---
# pseudotime: column name in adata.obs
fate = ov.single.Fate(adata, pseudotime='dpt_pseudotime')

# --- Step 1: Initial model ---
# alpha: ridge regularization. Higher = more regularized, fewer extreme coefficients.
coef_df = fate.model_init(test_size=0.3, alpha=0.1)
print(f"Initial R²: {fate.get_r2('raw'):.3f}")

# --- Step 2: Adaptive Threshold Regression ---
# stop: max iterations of gene removal. 500 is generous; 100 for quick runs.
# flux: R² drop tolerance. 0.01 means stop when R² drops >1% from peak.
threshold_df = fate.ATR(test_size=0.4, alpha=0.1, stop=500, flux=0.01)

# Visualize the threshold selection
fate.plot_filtering(figsize=(4, 3))

# --- Step 3: Refit on selected genes ---
filter_coef_df = fate.model_fit(test_size=0.3, alpha=0.1)
print(f"Filtered R²: {fate.get_r2('filter'):.3f} (was {fate.get_r2('raw'):.3f})")
print(f"Genes retained: {len(fate.filter_coef)}")

# --- Step 4: Statistical validation ---
kendall_df = fate.kendalltau_filter()
# High |kendalltau_sta| and low pvalue = truly pseudotime-associated

# --- Step 5: Mellon density (requires: pip install mellon) ---
# n_components: diffusion map components. 10-20 typical.
# knn: neighbors for density estimation. 30 is standard.
fate.low_density(n_components=10, knn=30, alpha=0.0, seed=0)

# --- Step 6: Lineage-specific scoring ---
# cluster_key: clustering column in adata.obs
# lineage: list of cluster labels defining the lineage of interest
fate.lineage_score(
    cluster_key='leiden',
    lineage=['20', '17'],  # Replace with your lineage cluster labels
    cell_mask='specification',
    density_key='mellon_log_density_lowd',
)

# --- Identify top fate-driving genes ---
fate_genes = adata.var.loc[fate.filter_coef.index, 'change_scores_lineage']
top_fate_genes = fate_genes.sort_values(ascending=False).head(20)
print("Top fate-driving genes:")
print(top_fate_genes)
```

## With data augmentation (noisy pseudotime)

```python
# Enable augmentation for robustness against pseudotime estimation noise
fate = ov.single.Fate(adata, pseudotime='palantir_pseudotime')

coef_df = fate.model_init(
    use_data_augmentation=True,
    augmentation_strategy='jitter_pseudotime_noise',  # 'gene_expression_noise', 'both'
    augmentation_intensity=0.05,
)

threshold_df = fate.ATR(
    stop=500, flux=0.01,
    use_data_augmentation=True,
    augmentation_strategy='jitter_pseudotime_noise',
    augmentation_intensity=0.05,
)

filter_coef_df = fate.model_fit(
    use_data_augmentation=True,
    augmentation_strategy='jitter_pseudotime_noise',
    augmentation_intensity=0.05,
)
```

## ATAC mode

```python
fate = ov.single.Fate(adata_atac, pseudotime='pseudotime')
fate.atac_init(...)
# Find peaks associated with pseudotime-driving genes
fate.get_related_peak(...)
```

## Visualization

```python
# ATR filtering curve (R² vs iteration)
fate.plot_filtering(figsize=(3, 3), color='#5ca8dc', fontsize=12)

# Model fit: predicted vs actual pseudotime
fate.plot_fitting(type='raw')      # Before filtering
fate.plot_fitting(type='filter')   # After ATR selection

# Color-coded by cluster identity
fate.plot_color_fitting(type='filter', cluster_key='leiden')
```

## Metrics access

```python
# Raw model (all genes)
print(f"R²: {fate.get_r2('raw')}, MSE: {fate.get_mse('raw')}, MAE: {fate.get_mae('raw')}")

# Filtered model (ATR-selected genes)
print(f"R²: {fate.get_r2('filter')}, MSE: {fate.get_mse('filter')}, MAE: {fate.get_mae('filter')}")
```
