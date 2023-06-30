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

## v 1.2.5

### single module:

- Add `pyVIA` module include `run`, `plot_piechart_graph`, `plot_stream`, `plot_trajectory_gams`, `plot_lineage_probability`, `plot_gene_trend`, `plot_gene_trend_heatmap`, `plot_clustergraph`
- Fix the error of warning of `utils.pyomic_plot_set` 
- Update the requirements included `pybind11`, `hnswlib`, `termcolor`, `pygam`, `pillow`, `gdown`

## v 1.2.6

### single module

- Add `pyVIA.get_piechart_dict` and `pyVIA.get_pseudotime`

## v 1.2.7

### bulk2single module

- Add `Single2Spatial` module included `load`, `save`, `train`, `spot_assess`
- Fix the error in install the packages in pip

## v 1.2.8

- fix the error of pip in install

### bulk2single module

- Change the `deep-forest` of `Single2Spatial` to `Neuron Network` to perform classification task
- The entire Single2Spatial inference process is accelerated using the GPU, and can be estimated at the batch level by modifying the set `predicted_size`, the original author's function is estimated spot by spot, which is very inefficient
- Update the logical of `Single2Spatial.load` to accelerate model loading

## v 1.2.9

### bulk module

- fix the duplicates_index mapping of `Matrix_ID_mapping`
- fix the hub genes plot of `pyWGCNA.plot_sub_network`
- fix the backupgene of `pyGSEA.geneset_enrichment` to support the rare species
- add the module matrix plot in `pyWGCNA.plot_matrix`

### single module

- add the `rank_genes_groups` check in `pySCSA`

### bulk2single module

- fix the import error of `deepforest`

## v 1.2.10

renamed the package to `omicverse`

### single module

- fix the argument error of `pySCSA`

### bulk2single module

- update the plot argument of `bulk2single`

## v 1.2.11

### bulk module

- fix `wilcoxon` method in `pyDEG.deg_analysis`
- add the parameter setting of treatment and control group's name in `pyDEG.plot_boxplot`
- fix the figure display not entire of `pyWGCNA.plot_matrix`
- fix the category correlation failed by ont-hot in `pyWGCNA.analysis_meta_correlation`
- fix the network display failed in `pyWGCNA.plot_sub_network` and updated the `utils.plot_network` to avoid this error.

## v 1.3.0

### bulk module

- add `DEseq2` method to `pyDEG.deg_analysis`
- add `pyGSEA` module in `bulk`
- change the name of raw `pyGSEA` to `pyGSE` in `bulk`
- add `get_gene_annotation` of `utils` to perform gene_name transformation

## v 1.3.1

### single module

- add `get_celltype_marker` method in `single`
- add `GLUE_pair`, `pyMOFA`, `pyMOFAART` module in `single`
- add tutorial of `Multi omics analysis by MOFA and GLUE`
- update tutorial of `Multi omics analysis by MOFA`

## v 1.4.0

### bulk2single module

- add `BulkTrajBlend` method in `bulk2single`

### single module

- fix the error of `scnocd` model
- add `save`, `load`, and `get_pair_dict` of `scnocd` model

### utils

- add `mde` method in utils
- add `gz` format support for `utils.read`

## v 1.4.1

### preprocess module

- add `pp`(preprocess) module included `qc`(quantity control), `hvg`(high variable feature), `pca` 
- add `data_files` for cell cycle calculate from [Cellula](https://github.com/andrecossa5/Cellula/) and [pegasus](https://github.com/lilab-bcb/pegasus/)

## v 1.4.3

### preprocess module

- fix sparse preprocess error of `pp`
- fix the trajectory import error of `via`
- add the gene correlation analysis of trajectory 

## v 1.4.4

### single module

- add `panglaodb` database to `pySCSA` module
- fix the error of `pySCSA.cell_auto_anno` when some celltype not found in clusters
- fix the error of `pySCSA.cell_anno` when `rank_genes_groups` not consisted with clusters
- add `pySIMBA` module in single to perform batch correction
- add `store_layers` and `retrieve_layers` in `ov.utils`
- add `plot_embedding_celltype` and `plot_cellproportion` in `ov.utils`



