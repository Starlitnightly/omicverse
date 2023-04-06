# Release notes

## v 1.0.0

First public release

## v 1.1.7

### bulk module:

- Add the Deseq2 included `pyDEseq` `deseq2_normalize`, `estimateSizeFactors`, `estimateDispersions`, `Matrix_ID_mapping`
- Add the tcga included `TCGA`
- Add the Enrichment included `geneset_enrichment`, `geneset_plot`

### single module:

- Add the scdrug included `autoResolution`, `writeGEP`, `Drug_Response`
- Add the cpdb included `cpdb_network_cal`, `cpdb_plot_network`, `cpdb_plot_interaction`, `cpdb_interaction_filtered`
- Add the scgsea included `geneset_aucell`, `pathway_aucell`, `pathway_aucell_enrichment`, `pathway_enrichment`, `pathway_enrichment_plot`

## v 1.1.8

### single module:

- Fix the cpdb's error included `import error` and `color error of cpdb_plot_network`
- Add the cpdb's method included `cpdb_submeans_exacted` that researchers can exact the sub network easily.

## v 1.1.9

### bulk2single module:

- Add the `bulk2single` module
- Fix the model load error from bulk2space
- Fix the early stop from bulk2space
- Add more friendly input method and visualisation
- Add the loss history visualisation

### utils module:

- Add the `pyomic_palette` in plot module

## v 1.1.10

Update all code reference

- Fix the parameter non-vaild on `single.mofa.mofa_run` function 
- Add the layer raw count addition on `single.scanpy_lazy` function
- Add `utils.plot_boxplot` to plot the box plot with jittered points.
- Add `bulk.pyDEseq.plot_boxplot` to plot the box plot with jittered points of specific Genes.


## v 1.2.0

### bulk module:

- Fix the `cutoff` parameter non-vaild on `bulk.geneset_enrichment`
- Add `pyPPI`,`pyGSEA`,`pyWGCNA`,`pyTCGA`,`pyDEG` module.

### bulk2single module:

- Add the `bulk2single.save` to save model by manual

## v 1.2.1-4

### single module:

- Add `pySCSA` module included `cell_anno`, `cell_anno_print`, `cell_auto_anno`, `get_model_tissue`
- Add filter the doublets cells of `single.scanpy_lazy`
- Add `single.scanpy_cellanno_from_dict` to annotate easier
- Updated the database of SCSA from [CellMarker2.0](http://bio-bigdata.hrbmu.edu.cn/CellMarker/)
- Fix the error database key `Ensembl_HGNC` and `Ensembl_Mouse` of SCSA 