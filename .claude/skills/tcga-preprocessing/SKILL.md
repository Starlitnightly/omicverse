---
name: tcga-bulk-data-preprocessing-with-omicverse
title: TCGA bulk data preprocessing with omicverse
description: "TCGA bulk RNA-seq preprocessing with pyTCGA: GDC sample sheets, expression archives, clinical metadata, Kaplan-Meier survival analysis, and annotated AnnData export."
---

# TCGA Bulk Data Preprocessing with OmicVerse

## Overview
Use this skill for loading TCGA data from GDC downloads, building normalised expression matrices, attaching clinical metadata, and running survival analyses through `ov.bulk.pyTCGA`.

## Instructions

### 1. Gather required downloads
Confirm the user has three items from the GDC Data Portal:
- `gdc_sample_sheet.<date>.tsv` — the sample sheet export
- Decompressed `gdc_download_xxxxx/` directory with expression archives
- `clinical.cart.<date>/` directory with clinical XML/JSON files

### 2. Initialise the TCGA helper
```python
import omicverse as ov
import scanpy as sc
ov.plot_set()

aml_tcga = ov.bulk.pyTCGA(sample_sheet_path, download_dir, clinical_dir)
aml_tcga.adata_init()  # Builds AnnData with raw counts, FPKM, and TPM layers
```

### 3. Persist and reload
```python
aml_tcga.adata.write_h5ad('data/ov_tcga_raw.h5ad', compression='gzip')

# To reload later:
new_tcga = ov.bulk.pyTCGA(sample_sheet_path, download_dir, clinical_dir)
new_tcga.adata_read('data/ov_tcga_raw.h5ad')
```

### 4. Initialise metadata and survival
```python
aml_tcga.adata_meta_init()   # Gene ID → symbol mapping, patient info
aml_tcga.survial_init()      # NOTE: "survial" spelling — see Critical API Reference below
```

### 5. Run survival analysis
```python
# Single gene
aml_tcga.survival_analysis('MYC', layer='deseq_normalize', plot=True)

# All genes (can take minutes for large gene sets)
aml_tcga.survial_analysis_all()  # NOTE: "survial" spelling
```

### 6. Export results
```python
aml_tcga.adata.write_h5ad('data/ov_tcga_survival.h5ad', compression='gzip')
```

## Critical API Reference

### IMPORTANT: Method Name Spelling Inconsistency

The pyTCGA API has an intentional spelling inconsistency. Two methods use "survial" (missing the 'v') while one uses the correct "survival":

| Method | Spelling | Purpose |
|--------|----------|---------|
| `survial_init()` | **survial** (no 'v') | Initialize survival metadata columns |
| `survival_analysis(gene, layer, plot)` | **survival** (correct) | Single-gene Kaplan-Meier curve |
| `survial_analysis_all()` | **survial** (no 'v') | Sweep all genes for survival significance |

```python
# CORRECT — use the exact method names as documented
aml_tcga.survial_init()                    # "survial" — no 'v'
aml_tcga.survival_analysis('MYC', layer='deseq_normalize', plot=True)  # "survival" — correct
aml_tcga.survial_analysis_all()            # "survial" — no 'v'

# WRONG — these will raise AttributeError
# aml_tcga.survival_init()                 # AttributeError! Use survial_init()
# aml_tcga.survival_analysis_all()         # AttributeError! Use survial_analysis_all()
```

### Survival Analysis Methodology

`survival_analysis()` performs Kaplan-Meier analysis:
1. Splits patients into high/low expression groups using the **median** as cutoff
2. Computes a **log-rank test** p-value to assess significance
3. If `plot=True`, renders survival curves with confidence intervals

**Layer selection matters**: Use `layer='deseq_normalize'` (recommended) because DESeq2 normalization accounts for library size and composition bias, making expression comparable across samples. Alternative: `layer='tpm'` for TPM-normalized values.

## Defensive Validation Patterns

```python
import os

# Before pyTCGA init: verify all paths exist
for name, path in [('sample_sheet', sample_sheet_path),
                    ('downloads', download_dir),
                    ('clinical', clinical_dir)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TCGA {name} path not found: {path}")

# After adata_init(): verify expected layers were created
expected_layers = ['counts', 'fpkm', 'tpm']
for layer in expected_layers:
    if layer not in aml_tcga.adata.layers:
        print(f"WARNING: Missing layer '{layer}' — check if TCGA archives are fully extracted")

# Before survival analysis: verify metadata is initialized
if 'survial_init' not in dir(aml_tcga) or aml_tcga.adata.obs.shape[1] < 5:
    print("WARNING: Run adata_meta_init() and survial_init() before survival analysis")
```

## Troubleshooting

- **`AttributeError: 'pyTCGA' object has no attribute 'survival_init'`**: Use the misspelled name `survial_init()` (missing 'v'). Same for `survial_analysis_all()`. See Critical API Reference above.
- **`KeyError` during `adata_meta_init()`**: Gene IDs in the expression matrix don't match expected format. TCGA uses ENSG IDs; the method maps them to symbols internally. Ensure archives are from the same GDC download.
- **Empty survival plot or NaN p-values**: Clinical XML files are missing date fields (days_to_death, days_to_last_follow_up). Check that the `clinical.cart.*` directory contains complete XML files, not just metadata JSONs.
- **`survial_analysis_all()` runs very slowly**: This tests every gene individually. For a genome with ~20,000 genes, expect 5-15 minutes. Consider filtering to genes of interest first.
- **Sample sheet column mismatch**: Verify the TSV uses tab separators and the header row matches GDC's expected format. Re-download from GDC if column names differ.
- **Missing `deseq_normalize` layer**: This layer is created during `adata_meta_init()`. If absent, re-run the metadata initialization step.

## Examples
- "Read my TCGA OV download, initialise metadata, and plot MYC survival curves using DESeq-normalised counts."
- "Reload a saved AnnData file, attach survival annotations, and export the updated `.h5ad`."
- "Run survival analysis for all genes and store the enriched dataset."

## References
- Tutorial notebook: `t_tcga.ipynb`
- Quick copy/paste commands: [`reference.md`](reference.md)
