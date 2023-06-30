# User

Import omicverse as:

```python
import omicverse as ov
```

## Bulk

|module|reference|
| ---- | ---- |
|[bulk.pyDEG](bulk/api_deseq)|Different expression analysis|
|[bulk.pyGSEA](bulk/api_enrichment)|Geneset enrichment analysis|
|[bulk.pyTCGA](bulk/api_tcga)|TCGA database preprocess|
|[bulk.pyPPI](bulk/api_network)|Protein-Protein interaction (PPI) analysis by String-db|
|[bulk.pyWGCNA](bulk/api_module)|WGCNA (Weighted gene co-expression network analysis)|

## Single

|module|reference|
| ---- | ---- |
|[single.pySCSA](single/api_scsa)|Celltype annotation with SCSA|
|[single.pyVIA](single/api_via)|Trajectory calculated with VIA|
|[single.COSG](single/api_cosg)|Marker genes filtered with COSG|
|[single.scDrug](single/api_scdrug)|Drug response predict with scDrug|
|[single.scGSEA](single/api_scgsea)|Pathway analysis with AUCell|
|[single.cpdb](single/api_cpdb)|Cell interaction with CellPhoneDB|
|[single.MOFA](single/api_mofa)|Multi omics analysis by MOFA|
|[single.pySIMBA](single/api_simba)|Batch correction with SIMBA|

## Bulk2Single

|module|reference|
| ---- | ---- |
|[bulk2single.BulkTrajBlend](bulk2single/api_bulktrajblend)|bulk RNA-seq generate interrupt cell in scRNA-seq|
|[bulk2single.Bulk2Single](bulk2single/api_bulk2single)|Bulk RNA-seq to Single RNA-seq|
|[bulk2single.Single2Spatial](bulk2single/api_single2spatial)|Single RNA-seq to Spatial RNA-seq|

## Tools
|module|reference|
| ---- | ---- |
|[pp](utils/api_pp)|preprocessing the scRNA-seq|
|[qc](utils/api_qc)|quantity control the scRNA-seq|
|[plot](utils/api_plot)|plot function|