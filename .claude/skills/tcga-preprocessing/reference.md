# TCGA Preprocessing Quick Commands

```python
import omicverse as ov
import scanpy as sc
import os

ov.plot_set()

# --- Paths to GDC downloads ---
gdc_sample_sheet = 'data/TCGA_OV/gdc_sample_sheet.2024-07-05.tsv'
gdc_download_files = 'data/TCGA_OV/gdc_download_20240705_180129.081531'
clinical_cart = 'data/TCGA_OV/clinical.cart.2024-07-05'

# Validate paths before init
for name, path in [('sample_sheet', gdc_sample_sheet),
                    ('downloads', gdc_download_files),
                    ('clinical', clinical_cart)]:
    assert os.path.exists(path), f"TCGA {name} not found: {path}"

# --- Initialize and build AnnData ---
aml_tcga = ov.bulk.pyTCGA(gdc_sample_sheet, gdc_download_files, clinical_cart)
aml_tcga.adata_init()  # Creates AnnData with counts, FPKM, TPM layers

# Save for later reuse
aml_tcga.adata.write_h5ad('data/TCGA_OV/ov_tcga_raw.h5ad', compression='gzip')

# --- Reload from saved file ---
new_tcga = ov.bulk.pyTCGA(gdc_sample_sheet, gdc_download_files, clinical_cart)
new_tcga.adata_read('data/TCGA_OV/ov_tcga_raw.h5ad')

# --- Metadata and survival initialization ---
aml_tcga.adata_meta_init()   # Gene ID mapping + patient metadata
aml_tcga.survial_init()      # NOTE: "survial" (no 'v') — intentional API spelling

# --- Single-gene survival analysis ---
# Uses Kaplan-Meier with median expression cutoff + log-rank test
# layer='deseq_normalize' recommended (accounts for library size + composition bias)
aml_tcga.survival_analysis('MYC', layer='deseq_normalize', plot=True)

# --- All-gene survival sweep (can take 5-15 minutes) ---
aml_tcga.survial_analysis_all()  # NOTE: "survial" (no 'v') — intentional API spelling

# --- Export enriched dataset ---
aml_tcga.adata.write_h5ad('data/TCGA_OV/ov_tcga_survival.h5ad', compression='gzip')
```

## API Spelling Quick Reference
| Method | Correct Spelling |
|--------|-----------------|
| Initialize survival metadata | `survial_init()` (no 'v') |
| Single-gene survival curve | `survival_analysis()` (correct spelling) |
| All-gene survival sweep | `survial_analysis_all()` (no 'v') |
