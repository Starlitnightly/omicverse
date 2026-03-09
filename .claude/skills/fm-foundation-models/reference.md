# Foundation model quick commands

## Discovery and profiling

```python
import omicverse as ov
import scanpy as sc

ov.plot_set()

# --- List all available models ---
all_models = ov.fm.list_models()  # All 22 models
embed_models = ov.fm.list_models(task="embed", skill_ready_only=True)  # Only production-ready

# --- Get detailed model spec ---
spec = ov.fm.describe_model("scgpt")
# Returns: input_contract (gene_id_scheme, species), output_contract (embedding_key, dim), resources

# --- Profile your data ---
profile = ov.fm.profile_data("data.h5ad")
# Auto-detects: species, gene_scheme, modality, n_cells, n_genes
# model_compatibility: per-model {compatible, issues, recommendations}
```

## Full end-to-end pipeline

```python
import omicverse as ov
import scanpy as sc

# 1. Profile
profile = ov.fm.profile_data("pbmc3k.h5ad")
print(f"Species: {profile['species']}, Gene scheme: {profile['gene_scheme']}")
print(f"Cells: {profile['n_cells']}, Genes: {profile['n_genes']}")

# 2. Select best model for task
selection = ov.fm.select_model(
    "pbmc3k.h5ad",
    task="embed",
    prefer_zero_shot=True,
    max_vram_gb=8,
)
model_name = selection['recommended']['name']
print(f"Recommended: {model_name} — {selection['recommended']['rationale']}")

# 3. Validate compatibility
validation = ov.fm.preprocess_validate("pbmc3k.h5ad", model_name, "embed")
print(f"Status: {validation['status']}")
if validation.get('auto_fixes'):
    for fix in validation['auto_fixes']:
        print(f"  Suggested fix: {fix}")

# 4. Run embedding
result = ov.fm.run(
    task="embed",
    model_name=model_name,
    adata_path="pbmc3k.h5ad",
    output_path="pbmc3k_embedded.h5ad",
    device="auto",
    batch_size=64,
)
print(f"Output: {result['output_path']}, Keys: {result['output_keys']}")

# 5. Visualize
adata = ov.read("pbmc3k_embedded.h5ad")
embed_key = result['output_keys'][0]  # e.g., 'X_scGPT'
sc.pp.neighbors(adata, use_rep=embed_key)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['cell_type'])

# 6. Interpret
metrics = ov.fm.interpret_results("pbmc3k_embedded.h5ad", task="embed")
print(f"Metrics: {metrics['metrics']}")
```

## Per-model embedding snippets

### scGPT (general RNA, multi-modal)
```python
result = ov.fm.run(
    task="embed",
    model_name="scgpt",
    adata_path="data.h5ad",
    output_path="data_scgpt.h5ad",
    device="auto",
    batch_size=64,
)
# Embedding at adata.obsm['X_scGPT'], dim=512
```

### Geneformer (Ensembl IDs, CPU-friendly)
```python
# Data MUST use Ensembl gene IDs (ENSG...)
result = ov.fm.run(
    task="embed",
    model_name="geneformer",
    adata_path="data_ensembl.h5ad",
    output_path="data_geneformer.h5ad",
    device="cpu",  # Full-speed on CPU
    batch_size=64,
)
# Embedding at adata.obsm['X_geneformer'], dim=512
```

### UCE (cross-species, 7 species)
```python
# Supports: human, mouse, zebrafish, mouse_lemur, macaque, frog, pig
result = ov.fm.run(
    task="embed",
    model_name="uce",
    adata_path="zebrafish_data.h5ad",
    output_path="zebrafish_uce.h5ad",
    device="cuda",  # GPU required (16GB VRAM)
    batch_size=64,
)
# Embedding at adata.obsm['X_uce'], dim=1280
```

### CellPLM (fastest inference)
```python
result = ov.fm.run(
    task="embed",
    model_name="cellplm",
    adata_path="large_data.h5ad",
    output_path="large_cellplm.h5ad",
    device="auto",
    batch_size=128,  # CellPLM supports higher batch size
)
# Embedding at adata.obsm['X_cellplm'], dim=512
```

## Integration task example

```python
# Batch integration with a foundation model
result = ov.fm.run(
    task="integrate",
    model_name="scgpt",
    adata_path="multi_batch.h5ad",
    output_path="integrated.h5ad",
    batch_key="batch",     # Column in adata.obs with batch labels
    device="auto",
)
# Integration embedding at adata.obsm['X_scGPT_integrated']
```

## Annotation task example

```python
# Cell type annotation (zero-shot if model supports it)
result = ov.fm.run(
    task="annotate",
    model_name="scgpt",
    adata_path="query.h5ad",
    output_path="annotated.h5ad",
    label_key="cell_type",  # Reference labels (for fine-tuned models)
    device="auto",
)
# Predictions at adata.obs['scgpt_pred'], confidence at adata.obs['scgpt_pred_score']
```

## Environment variable configuration

```bash
# Set checkpoint directory for all models
export OV_FM_CHECKPOINT_DIR=/path/to/all/checkpoints

# Or per-model
export OV_FM_CHECKPOINT_DIR_SCGPT=/path/to/scgpt/weights
export OV_FM_CHECKPOINT_DIR_GENEFORMER=/path/to/geneformer/weights

# Disable conda subprocess isolation (if dependencies are in current env)
export OV_FM_DISABLE_CONDA_SUBPROCESS=1
```
