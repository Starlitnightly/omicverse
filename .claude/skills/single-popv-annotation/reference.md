# Quick commands: PopV annotation

## Complete pipeline (retrain mode)

```python
import omicverse as ov
ov.plot_set(font_path='Arial')

# Load data
ref_adata = ov.read('data/reference_atlas.h5ad')
query_adata = ov.read('data/query_cells.h5ad')

# Preprocess and concatenate
process_obj = ov.popv.Process_Query(
    query_adata=query_adata,
    ref_adata=ref_adata,
    ref_labels_key='cell_type',
    ref_batch_key='batch',
    query_batch_key=None,
    cl_obo_folder=False,
    prediction_mode='retrain',
    unknown_celltype_label='unknown',
    n_samples_per_label=300,
    hvg=4000,
    save_path_trained_models='models/popv/',
)

# Run annotation with all current algorithms
ov.popv.annotate_data(
    process_obj.adata,
    methods='all',
    save_path='results/popv/',
)

# Extract query cell predictions
query_mask = process_obj.adata.obs['_dataset'] == 'query'
query_results = process_obj.adata[query_mask].obs[[
    'popv_majority_vote_prediction',
    'popv_majority_vote_score',
]]
print(query_results['popv_majority_vote_prediction'].value_counts())
```

## Fast mode (subset of algorithms)

```python
process_obj = ov.popv.Process_Query(
    query_adata=query_adata,
    ref_adata=ref_adata,
    ref_labels_key='cell_type',
    ref_batch_key='batch',
    cl_obo_folder=False,
    prediction_mode='fast',
    n_samples_per_label=200,
    hvg=3000,
    save_path_trained_models='models/popv_fast/',
)

# Fast mode automatically selects FAST_ALGORITHMS:
# KNN_SCVI, SCANVI_POPV, Support_Vector, XGboost, ONCLASS, CELLTYPIST
ov.popv.annotate_data(process_obj.adata)
```

## Custom algorithm subset (no GPU needed)

```python
process_obj = ov.popv.Process_Query(
    query_adata=query_adata,
    ref_adata=ref_adata,
    ref_labels_key='cell_type',
    ref_batch_key='batch',
    cl_obo_folder=False,
    prediction_mode='retrain',
    save_path_trained_models='models/popv_cpu/',
)

# CPU-only classical methods
ov.popv.annotate_data(
    process_obj.adata,
    methods=['CELLTYPIST', 'Support_Vector', 'XGboost', 'KNN_HARMONY', 'KNN_BBKNN'],
)
```

## Inference mode (reuse trained models)

```python
# Second run: reuse saved models from a previous retrain
process_obj = ov.popv.Process_Query(
    query_adata=new_query_adata,
    ref_adata=ref_adata,
    ref_labels_key='cell_type',
    ref_batch_key='batch',
    cl_obo_folder=False,
    prediction_mode='inference',
    save_path_trained_models='models/popv/',  # same path as retrain
)

ov.popv.annotate_data(process_obj.adata)
```

## Hub model (pretrained)

```python
from omicverse.popv.hub import HubModel

# Download pretrained model
model = HubModel.pull_from_huggingface_hub(
    repo_name='popv/immune_all',
    cache_dir='models/hub/',
)

# Annotate with pretrained model
result_adata = model.annotate_data(
    query_adata=query_adata,
    query_batch_key='batch',
    prediction_mode='fast',
)

print(result_adata.obs['popv_majority_vote_prediction'].value_counts())
```

## Visualization

```python
# Confusion matrices: each method vs consensus
ov.popv.make_agreement_plots(
    process_obj.adata,
    prediction_keys=process_obj.adata.uns['prediction_keys'],
    popv_prediction_key='popv_majority_vote_prediction',
    save_folder='figures/popv/',
)

# Agreement score per cell type
ov.popv.agreement_score_bar_plot(
    process_obj.adata,
    popv_prediction_key='popv_majority_vote_prediction',
    save_folder='figures/popv/',
)

# Prediction score distribution
ov.popv.prediction_score_bar_plot(
    process_obj.adata,
    popv_prediction_score='popv_majority_vote_score',
    save_folder='figures/popv/',
)

# Cell type ratio comparison (ref vs query)
ov.popv.celltype_ratio_bar_plot(
    process_obj.adata,
    popv_prediction='popv_majority_vote_prediction',
    save_folder='figures/popv/',
    normalize=True,
)
```

## GPU settings

```python
import omicverse.popv as popv
popv.settings.accelerator = 'gpu'
popv.settings.cuml = True
popv.settings.n_jobs = 10
popv.settings.return_probabilities = True
popv.settings.compute_umap_embedding = True
```

## Key function signatures

```python
# Process_Query
ov.popv.Process_Query(
    query_adata, ref_adata, ref_labels_key, ref_batch_key,
    cl_obo_folder=False,
    query_batch_key=None, query_layer_key=None, ref_layer_key=None,
    prediction_mode='retrain', unknown_celltype_label='unknown',
    n_samples_per_label=300, save_path_trained_models='tmp/',
    pretrained_scvi_path=None, relabel_reference_cells=False, hvg=4000,
)

# annotate_data
ov.popv.annotate_data(
    adata, methods=None, save_path=None, methods_kwargs=None,
)

# Visualization
ov.popv.make_agreement_plots(adata, prediction_keys, popv_prediction_key='popv_prediction',
                              save_folder=None, show=True)
ov.popv.agreement_score_bar_plot(adata, popv_prediction_key='popv_prediction',
                                  consensus_score_key='popv_prediction_score', save_folder=None)
ov.popv.prediction_score_bar_plot(adata, popv_prediction_score='popv_prediction_score',
                                   save_folder=None)
ov.popv.celltype_ratio_bar_plot(adata, popv_prediction='popv_prediction',
                                 save_folder=None, normalize=True)

# Hub
HubModel.pull_from_huggingface_hub(repo_name, cache_dir=None, revision=None)
model.annotate_data(query_adata, query_batch_key=None, save_path='tmp',
                     prediction_mode='fast', methods=None, gene_symbols=None)
```
