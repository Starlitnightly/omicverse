# TCGA preprocessing quick commands

```python
import omicverse as ov
import scanpy as sc

ov.plot_set()

gdc_sample_sheet = 'data/TCGA_OV/gdc_sample_sheet.2024-07-05.tsv'
gdc_download_files = 'data/TCGA_OV/gdc_download_20240705_180129.081531'
clinical_cart = 'data/TCGA_OV/clinical.cart.2024-07-05'

aml_tcga = ov.bulk.pyTCGA(gdc_sample_sheet, gdc_download_files, clinical_cart)
aml_tcga.adata_init()

aml_tcga.adata.write_h5ad('data/TCGA_OV/ov_tcga_raw.h5ad', compression='gzip')

# reload later
new_tcga = ov.bulk.pyTCGA(gdc_sample_sheet, gdc_download_files, clinical_cart)
new_tcga.adata_read('data/TCGA_OV/ov_tcga_raw.h5ad')

aml_tcga.adata_meta_init()
aml_tcga.survial_init()

aml_tcga.survival_analysis('MYC', layer='deseq_normalize', plot=True)

aml_tcga.survial_analysis_all()
aml_tcga.adata.write_h5ad('data/TCGA_OV/ov_tcga_survial_all.h5ad', compression='gzip')
```
