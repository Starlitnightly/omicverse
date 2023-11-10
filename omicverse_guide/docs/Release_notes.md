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

### preprocess module

- add `store_layers` and `retrieve_layers` in `ov.utils`
- add `plot_embedding_celltype` and `plot_cellproportion` in `ov.utils`

## v 1.4.5

### single module

- add `MetaTiME` module in single to perform celltype annotation automatically in TME

## v 1.4.12

update `conda install omicverse -c conda-forge`

### single module

- add `pyTOSICA` module in single to perform celltype migration from reference scRNA-seq in Tranformer model
- add `atac_concat_get_index`,`atac_concat_inner`,`atac_concat_outer` function to merge/concat the scATAC data.
- fix `MetaTime.predicted` when Unknown cell type appear

### preprocess module

- add `plot_embedding` in `ov.utils` to plot umap in special color dict

## v 1.4.13

### bulk module 

- add `mad_filtered` to filtered the robust genes when calculated the network in `ov.bulk.pyWGCNA` module
- fix `string_interaction` in `ov.bulk.pyPPI` for string-db updated.

### preprocess module

- change `mode` arguement of `pp.preprocess`, normalize|HVGs：We use | to control the preprocessing step, | before for the normalisation step, either `shiftlog` or `pearson`, and | after for the highly variable gene calculation step, either `pearson` or `seurat`. Our default is `shiftlog|pearson`.
- add `ov.utils.embedding`,`ov.utils.neighbors`, and `ov.utils.stacking_vol`

## v 1.4.14

### preprocess module

- add `batch_key` in `pp.preprocess` and `pp.qc`

### utils module

- add `plot_ConvexHull` to visualize the boundary of clusters
- add `weighted_knn_trainer` and `weighted_knn_transfer` for multi adata integrate
  
### single module

- fix the error of import of `mofa` 

## v 1.4.17

### bulk module

- fix the compatibility of `pydeseq2` while version is `0.4.0`
- add `bulk.batch_correction` for multi bulk RNA-seq/microarray sample

### single module

- add `single.batch_correction` for multi single cell datasets

### preprocess module

- add parameter `layers_add` in `pp.scale`

## v 1.5.0

### single module

- add `cellfategenie` to calculate the timing-associated genes/genesets
- fix the name error of `atac_concat_outer`
- add more kwargs of `batch_correction`

### utils module

- add `plot_heatmap` to visualize the heatmap of pseudotime
- fix the `embedding` when the version of mpl larger than 3.7.0
- add `geneset_wordcloud` to visualize the genesets heatmap of pseudotime

## v 1.5.1

### single module

- add `scLTNN` to infer the cell trajectory

### bulk2single module

- Update the cell fraction prediction with `TAPE` in bulk2single
- Fix the group and normalization in bulk2single

### utils module

- add `Ro/e` calculated (by:Haihao Zhang)
- add `cal_paga` and `plot_paga` to visualize the state transfer matrix
- fix the `read` function 

## v 1.5.2

### bulk2single module

- Fix the matrix error when the symbol of genes not unique.
- Fix the `interpolation` of BulkTrajBlend when the target cells not exist.
- Fix the `generate` of BulkTrajBlend.
- Fix thhe arguement of `vae_configure` in BulkTrajBlend when cell_target_num is None
- Add `max_single_cells` for BulkTrajBlend input

### single module
- Fix the error of pyVIA when root is None
