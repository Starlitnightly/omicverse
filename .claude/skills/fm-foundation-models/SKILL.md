---
name: foundation-model-analysis
title: Foundation model analysis
description: "Foundation model workflows: scGPT, Geneformer, UCE, CellPLM cell embedding, annotation, integration via ov.fm unified API. 22 models."
---

# Foundation Model Analysis

Use this skill when a user wants to generate cell embeddings, annotate cell types, integrate batches, or predict perturbation effects using single-cell foundation models. The `ov.fm` module provides a unified 6-step API that works identically across all 22 supported models.

## Model Selection Guide

Pick a model based on your task, species, and hardware. The 5 skill-ready models have full adapter support:

| Model | Tasks | Species | Gene IDs | Min VRAM | CPU? | Best when |
|-------|-------|---------|----------|----------|------|-----------|
| **scGPT** | embed, integrate | human, mouse | symbol | 8 GB | Yes | General RNA, multi-modal (RNA+ATAC+Spatial) |
| **Geneformer** | embed, integrate | human | **ensembl** | 4 GB | **Yes** | Ensembl IDs, CPU-only environments, network biology |
| **UCE** | embed, integrate | 7 species | symbol | 16 GB | No | Cross-species (zebrafish, macaque, pig, frog, lemur) |
| **scFoundation** | embed, integrate | human | custom | 16 GB | No | xTrimoGene architecture, perturbation tasks |
| **CellPLM** | embed, integrate | human | symbol | 8 GB | Yes | Fastest inference (batch_size=128), cell-centric |

12 additional partial models (scBERT, GeneCompass, Nicheformer, scMulan, tGPT, CellFM, scCello, scPrint, AiDocell, Pulsar, Atacformer, scPlantLLM) and 5+ experimental models are also registered.

### Quick decision tree
- Cross-species → **UCE**
- Ensembl gene IDs or CPU-only → **Geneformer**
- ATAC-seq data → **Atacformer** (partial) or **scGPT**
- Multi-omics (RNA+ATAC+Protein) → **scMulan** (partial)
- Spatial transcriptomics → **Nicheformer** (partial) or **scGPT**
- Fastest throughput on large datasets → **CellPLM**
- General RNA, no special needs → **scGPT**

## 6-Step Unified Workflow

Every FM analysis follows the same pipeline regardless of model choice:

### Step 1: Discover available models
```python
import omicverse as ov
models = ov.fm.list_models(task="embed", skill_ready_only=True)
# Returns: {"count": int, "models": [{name, version, tasks, species, ...}]}
```

### Step 2: Profile your data
```python
profile = ov.fm.profile_data("pbmc3k.h5ad")
# Auto-detects: species, gene_scheme (symbol/ensembl), modality, batch/celltype columns
# Returns model_compatibility per model (compatible: bool, issues: [], recommendations: [])
```

### Step 3: Select best model
```python
selection = ov.fm.select_model(
    "pbmc3k.h5ad",
    task="embed",
    prefer_zero_shot=True,   # No labeled data needed
    max_vram_gb=8,            # Hardware constraint
)
model_name = selection['recommended']['name']
# Returns: recommended model with rationale, fallbacks list, preprocessing_notes
```

### Step 4: Validate compatibility
```python
validation = ov.fm.preprocess_validate("pbmc3k.h5ad", model_name, "embed")
# Returns: status ("ready"/"needs_preprocessing"), diagnostics, auto_fixes suggestions
```

### Step 5: Execute
```python
result = ov.fm.run(
    task="embed",
    model_name=model_name,
    adata_path="pbmc3k.h5ad",
    output_path="pbmc3k_embedded.h5ad",
    device="auto",          # auto-detects cuda/mps/cpu
    batch_size=64,          # CellPLM can use 128
    batch_key=None,         # For integration tasks
    label_key=None,         # For annotation tasks
    checkpoint_dir=None,    # Auto-resolved from env vars or cache
)
# Returns: {output_path, output_keys, statistics} or {error: str}
```

### Step 6: Interpret results
```python
metrics = ov.fm.interpret_results(
    "pbmc3k_embedded.h5ad",
    task="embed",
    generate_umap=True,
    color_by=["cell_type"],
)
# Returns: metrics (silhouette scores), visualizations, embedding_keys
```

## Gene ID Resolution

Gene ID mismatch is the most common failure mode. The `profile_data()` function detects your data's gene scheme automatically.

| Model | Expected IDs | Example | Auto-convert? |
|-------|-------------|---------|---------------|
| scGPT | HGNC symbols | TP53, CD4 | No — data must use symbols |
| Geneformer | Ensembl IDs | ENSG00000141510 | No — data must use Ensembl |
| UCE | HGNC symbols | TP53, CD4 | No |
| scFoundation | Custom vocab | 19,264 gene vocab | Adapter handles mapping |

If `profile_data()` reports `gene_scheme: "ensembl"` but you selected scGPT (which needs symbols), either convert gene IDs first or switch to Geneformer.

## Hardware Requirements

| Model | GPU Required | Min VRAM | CPU Fallback | Default Batch Size |
|-------|-------------|----------|--------------|-------------------|
| scGPT | Recommended | 8 GB | Yes (slow) | 64 |
| Geneformer | No | 4 GB | **Yes (full speed)** | 64 |
| UCE | Yes | 16 GB | No | 64 |
| scFoundation | Yes | 16 GB | No | 64 |
| CellPLM | Recommended | 8 GB | Yes | 128 |

Device auto-detection priority: CUDA → MPS (Apple Silicon) → CPU.

Checkpoint resolution priority:
1. `checkpoint_dir` parameter in `ov.fm.run()`
2. Model-specific env var: `OV_FM_CHECKPOINT_DIR_SCGPT`
3. Base env var + subfolder: `OV_FM_CHECKPOINT_DIR/scgpt/`
4. OmicVerse model cache (auto-download if available)

## Critical API Reference

### TaskType values
`"embed"`, `"annotate"`, `"integrate"`, `"perturb"`, `"spatial"`, `"drug_response"`

### Output keys are model-specific
```python
# After running, embeddings are stored in adata.obsm with model-specific keys:
# scGPT → adata.obsm['X_scGPT']
# Geneformer → adata.obsm['X_geneformer']
# UCE → adata.obsm['X_uce']
# Check result['output_keys'] for the exact keys written
```

### Provenance tracking
```python
# Every run writes provenance to adata.uns['fm']:
# {"runs_json": [...], "latest_json": "..."}
# Contains: model_name, version, task, timestamp, output_keys
```

## Defensive Validation Patterns

```python
import os

# Before any FM workflow: verify input file exists
assert os.path.isfile(adata_path), f"Input file not found: {adata_path}"

# After profile_data: check species was detected
profile = ov.fm.profile_data(adata_path)
assert 'unknown' not in profile.get('species', 'unknown').lower(), \
    f"Species not detected. Check gene names — use HGNC symbols (human) or standard names (mouse)."

# After select_model: verify a model was recommended
selection = ov.fm.select_model(adata_path, task="embed")
assert 'recommended' in selection and selection['recommended'], \
    "No compatible model found. Check species, gene IDs, and hardware constraints."

# Before run: verify checkpoint exists (for models requiring local weights)
desc = ov.fm.describe_model(model_name)
if desc.get('resources', {}).get('checkpoint_url'):
    print(f"Model may need checkpoint download. Check OV_FM_CHECKPOINT_DIR env var.")
```

## Troubleshooting

- **`Gene ID mismatch` warning in profile**: Your data uses Ensembl IDs but the selected model expects symbols (or vice versa). Convert with `adata.var_names = adata.var['gene_symbols']` or switch to a compatible model.
- **`CUDA out of memory`**: Reduce `batch_size` (try 32 or 16). For UCE/scFoundation (16GB VRAM), ensure no other GPU processes are running.
- **`Model not installed` or `ImportError`**: Some models need separate packages. Install via `pip install scgpt` / `pip install geneformer` or use conda isolation (`OV_FM_DISABLE_CONDA_SUBPROCESS=0`).
- **`Species unsupported`**: Most models only support human. For mouse, use scGPT or UCE. For zebrafish/pig/frog, only UCE works.
- **Empty embeddings (all zeros)**: Input data may have constant or near-zero expression. Filter genes with `sc.pp.filter_genes(adata, min_cells=10)` before running.
- **`Checkpoint not found`**: Set the environment variable `OV_FM_CHECKPOINT_DIR_<MODEL>=/path/to/weights` or pass `checkpoint_dir` directly to `ov.fm.run()`.
- **`device='auto'` picks CPU despite GPU available**: Check `torch.cuda.is_available()`. If False, verify CUDA drivers and PyTorch CUDA build.

## Examples
- "Generate scGPT embeddings for my PBMC dataset and visualize on UMAP."
- "Which foundation model works best for my mouse brain scRNA-seq data?"
- "Embed my ATAC-seq data using a foundation model — I only have 8GB VRAM."
- "Profile my h5ad file and tell me which models are compatible."

## References
- Quick copy/paste commands: [`reference.md`](reference.md)
- FM API source: `omicverse/fm/api.py`
- Model registry: `omicverse/fm/registry.py`