---
name: single-popv-annotation
title: PopV population-level cell type annotation
description: "PopV population-level cell annotation: 10 algorithms (SCVI, SCANVI, CellTypist, OnClass, RF, SVM, XGBoost, BBKNN, HARMONY, SCANORAMA), consensus voting, pretrained hub models."
---

# PopV Population-Level Cell Type Annotation

PopV (Population Voting) annotates cell types by running up to 10 classification algorithms and aggregating predictions via majority voting. Unlike single-method annotation (SCSA, MetaTiME, CellTypist alone), PopV produces a consensus prediction that is more robust to individual algorithm failures. The module also supports ontology-aware voting via the Cell Ontology (CL) for hierarchical label resolution.

## Defensive Validation

```python
# Before PopV: verify reference has the cell type column
assert ref_labels_key in ref_adata.obs.columns, \
    f"ref_adata.obs['{ref_labels_key}'] not found. Available: {list(ref_adata.obs.columns)}"

# Verify no NaN in reference labels
assert ref_adata.obs[ref_labels_key].notna().all(), \
    f"NaN values in ref_adata.obs['{ref_labels_key}']. Use fillna() or drop these cells."

# Verify gene overlap
overlap = query_adata.var_names.intersection(ref_adata.var_names)
assert len(overlap) > 100, \
    f"Only {len(overlap)} overlapping genes between query and reference. Check var_names format (ENSEMBL vs symbol)."
```

## Stage 1: Data Preparation

```python
import omicverse as ov

# Process_Query preprocesses and concatenates query + reference
process_obj = ov.popv.Process_Query(
    query_adata=query_adata,
    ref_adata=ref_adata,
    ref_labels_key='cell_type',           # REQUIRED: column in ref_adata.obs
    ref_batch_key='batch',                # batch column in ref_adata.obs
    query_batch_key='batch',              # batch column in query_adata.obs (optional)
    cl_obo_folder=False,                  # False to skip ontology, or path to CL .obo file
    prediction_mode='retrain',            # 'retrain' | 'inference' | 'fast'
    unknown_celltype_label='unknown',     # label for query cells
    n_samples_per_label=300,              # subsample reference per cell type
    hvg=4000,                             # number of highly variable genes
    save_path_trained_models='tmp/',      # where to save models
    pretrained_scvi_path=None,            # path to pretrained scVI model (optional)
)
```

**prediction_mode choices:**
- `'retrain'` — Train all models from scratch on reference+query. Most accurate, slowest.
- `'inference'` — Load previously saved models. Requires `save_path_trained_models` from prior run.
- `'fast'` — Skip integration-heavy algorithms. Uses FAST_ALGORITHMS subset.

**Preprocessing applied automatically:**
- Filters cells with < 30 total counts
- Log1p normalization (target_sum=1e4)
- PCA on reference (50 components)
- Stores raw counts in `layers['scvi_counts']`

## Stage 2: Annotation

```python
# Run all algorithms and compute consensus
ov.popv.annotate_data(
    process_obj.adata,
    methods='all',                        # or list of specific algorithms
    save_path='results/popv/',            # saves predictions.csv here
    methods_kwargs=None,                  # dict of per-method overrides
)
```

### Available Algorithms (10 total)

| Algorithm | Result Key | Type | Speed |
|-----------|-----------|------|-------|
| `KNN_SCVI` | `popv_knn_on_scvi_prediction` | Deep learning + KNN | Medium |
| `SCANVI_POPV` | `popv_scanvi_prediction` | Semi-supervised DL | Medium |
| `CELLTYPIST` | `popv_celltypist_prediction` | Logistic regression | Fast |
| `ONCLASS` | `popv_onclass_prediction` | Ontology-guided | Medium |
| `Support_Vector` | `popv_svm_prediction` | SVM | Fast |
| `XGboost` | `popv_xgboost_prediction` | Gradient boosting | Fast |
| `KNN_HARMONY` | `popv_knn_harmony_prediction` | Harmony + KNN | Fast |
| `KNN_BBKNN` | `popv_knn_bbknn_prediction` | BBKNN + KNN | Fast |
| `Random_Forest` | `popv_rf_prediction` | Random forest | Fast |
| `KNN_SCANORAMA` | `popv_knn_scanorama_prediction` | Scanorama + KNN | Medium |

**Algorithm subsets:**
- `FAST_ALGORITHMS`: KNN_SCVI, SCANVI_POPV, Support_Vector, XGboost, ONCLASS, CELLTYPIST (used with `prediction_mode='fast'`)
- `CURRENT_ALGORITHMS`: All except Random_Forest and KNN_SCANORAMA (outdated)
- `'all'` or `None`: Uses CURRENT_ALGORITHMS (or FAST_ALGORITHMS in fast mode)

### Selecting Specific Methods

```python
# Run only fast classical methods
ov.popv.annotate_data(
    process_obj.adata,
    methods=['CELLTYPIST', 'Support_Vector', 'XGboost'],
)

# Override per-method parameters
ov.popv.annotate_data(
    process_obj.adata,
    methods=['KNN_SCVI', 'SCANVI_POPV'],
    methods_kwargs={
        'KNN_SCVI': {'train_kwargs': {'max_epochs': 50}},
        'SCANVI_POPV': {'train_kwargs': {'max_epochs': 50}},
    },
)
```

## Stage 3: Consensus Results & Visualization

After `annotate_data()`, these columns appear in `adata.obs`:

| Column | Description |
|--------|-------------|
| `popv_majority_vote_prediction` | Majority vote across all methods |
| `popv_majority_vote_score` | Number of agreeing methods |
| `popv_prediction` | Ontology-aggregated consensus (if CL enabled) |
| `popv_prediction_score` | Ontology consensus score |

```python
# Agreement plots: confusion matrices per method vs consensus
ov.popv.make_agreement_plots(
    process_obj.adata,
    prediction_keys=process_obj.adata.uns['prediction_keys'],
    popv_prediction_key='popv_prediction',
    save_folder='results/popv/',
    show=True,
)

# Bar plot: agreement score per cell type
ov.popv.agreement_score_bar_plot(
    process_obj.adata,
    popv_prediction_key='popv_prediction',
    save_folder='results/popv/',
)

# Bar plot: prediction score distribution
ov.popv.prediction_score_bar_plot(
    process_obj.adata,
    popv_prediction_score='popv_prediction_score',
    save_folder='results/popv/',
)

# Bar plot: cell type proportions (ref vs query)
ov.popv.celltype_ratio_bar_plot(
    process_obj.adata,
    popv_prediction='popv_prediction',
    save_folder='results/popv/',
)
```

## Stage 4: Pretrained Hub Models (Optional)

For large references (e.g., Human Cell Atlas), use pretrained models to skip training:

```python
from omicverse.popv.hub import HubModel

# Pull pretrained model from HuggingFace
model = HubModel.pull_from_huggingface_hub(
    repo_name='popv/immune_all',
    cache_dir='models/popv/',
)

# Annotate query data directly (fast mode)
result_adata = model.annotate_data(
    query_adata=query_adata,
    query_batch_key='batch',
    prediction_mode='fast',
    methods=None,  # uses model's default methods
)
```

## Critical API Reference

```python
# CORRECT: methods as list of strings matching class names
ov.popv.annotate_data(adata, methods=['KNN_SCVI', 'CELLTYPIST', 'Support_Vector'])

# WRONG: passing class objects or lowercase names
# ov.popv.annotate_data(adata, methods=[KNN_SCVI, CELLTYPIST])  # TypeError
# ov.popv.annotate_data(adata, methods=['knn_scvi'])             # KeyError

# CORRECT: ref_labels_key must exist in ref_adata.obs before Process_Query
assert 'cell_type' in ref_adata.obs.columns
process_obj = ov.popv.Process_Query(ref_labels_key='cell_type', ...)

# WRONG: forgetting to set unknown_celltype_label causes NaN in voting
# process_obj = ov.popv.Process_Query(..., unknown_celltype_label=None)  # NaN errors

# CORRECT: access consensus results after annotation
final_labels = process_obj.adata.obs['popv_majority_vote_prediction']
# or ontology-refined:
final_labels = process_obj.adata.obs['popv_prediction']

# WRONG: looking for results on the original query_adata
# query_adata.obs['popv_prediction']  # KeyError: results are on process_obj.adata
```

## GPU Acceleration

```python
import omicverse.popv as popv
popv.settings.accelerator = 'gpu'   # for scVI/scANVI training
popv.settings.cuml = True           # for KNN/SVM/RF via cuML
popv.settings.n_jobs = 10           # parallel jobs for CPU methods
```

## Troubleshooting

- **`RuntimeError: CUDA out of memory` during scVI/scANVI training**: Reduce `hvg` (try 2000), decrease `n_samples_per_label` (try 100), or switch to `prediction_mode='fast'` which uses fewer epochs.
- **CellTypist model download fails**: Set `methods_kwargs={'CELLTYPIST': {'method_kwargs': {'model': '/path/to/local/model.pkl'}}}` to use a local model file.
- **Low consensus agreement (<50% cells agree)**: Some algorithms may not suit your tissue. Exclude underperforming methods: check per-method predictions and drop outliers from the `methods` list.
- **`KeyError: 'gene_name'` — gene identifier mismatch**: Harmonize var_names between reference and query before calling `Process_Query`. Use `adata.var_names = adata.var['gene_symbols']` if ENSEMBL IDs are in var_names.
- **`ValueError: batch_key contains NaN`**: Clean batch columns before PopV. Apply the batch validation pattern from the single-preprocessing skill: `adata.obs['batch'] = adata.obs['batch'].fillna('unknown').astype('category')`.
- **`FileNotFoundError` in inference mode**: Ensure `save_path_trained_models` points to the same directory used during the original `retrain` run. Check that model files (.pt, .pkl, .joblib) exist.

## Dependencies
- Core: `omicverse`, `scanpy`, `anndata`, `numpy`, `pandas`
- Deep learning: `scvi-tools`, `torch` (for KNN_SCVI, SCANVI_POPV)
- Classical ML: `scikit-learn`, `xgboost` (for RF, SVM, XGBoost)
- Integration: `harmonypy`, `bbknn`, `scanorama` (for respective KNN methods)
- Annotation: `celltypist`, `OnClass` (optional per method)
- Ontology: `obonet`, `pronto` (for ontology-aware voting)
- Hub: `huggingface_hub` (for pretrained models)

## Examples
- "Annotate my PBMC query data against a reference atlas using PopV with all 10 algorithms and visualize the consensus."
- "Use a pretrained PopV hub model to quickly annotate my lung tissue scRNA-seq data."
- "Run PopV with only classical methods (SVM, XGBoost, CellTypist) to annotate my query cells without GPU."

## References
- Quick copy/paste commands: [`reference.md`](reference.md)
