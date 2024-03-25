# Release Notes

## v 1.0.0
- First public release.

## v 1.1.7
### bulk module:
- Added Deseq2, including `pyDEseq` functions: `deseq2_normalize`, `estimateSizeFactors`, `estimateDispersions`, `Matrix_ID_mapping`.
- Included TCGA with `TCGA`.
- Introduced Enrichment with functions `geneset_enrichment`, `geneset_plot`.

### single module:
- Integrated scdrug with functions `autoResolution`, `writeGEP`, `Drug_Response`.
- Added cpdb with functions `cpdb_network_cal`, `cpdb_plot_network`, `cpdb_plot_interaction`, `cpdb_interaction_filtered`.
- Included scgsea with functions `geneset_aucell`, `pathway_aucell`, `pathway_aucell_enrichment`, `pathway_enrichment`, `pathway_enrichment_plot`.

## v 1.1.8
### single module:
- Addressed errors in cpdb, including import errors and color issues in `cpdb_plot_network`.
- Introduced `cpdb_submeans_exacted` in cpdb for easy sub-network extraction.

## v 1.1.9
### bulk2single module:
- Added the `bulk2single` module.
- Fixed model load error from bulk2space.
- Resolved early stop issues from bulk2space.
- Included more user-friendly input methods and visualizations.
- Added loss history visualization.

### utils module:
- Introduced `pyomic_palette` in the plot module.

## v 1.1.10
- Updated all code references.

### single module:
- Fixed non-valid parameters in `single.mofa.mofa_run` function.
- Added layer raw count addition in `single.scanpy_lazy` function.
- Introduced `utils.plot_boxplot` for plotting box plots with jittered points.
- Added `bulk.pyDEseq.plot_boxplot` for plotting box plots with jittered points for specific genes.

## v 1.2.0
### bulk module:
- Fixed non-valid `cutoff` parameter in `bulk.geneset_enrichment`.
- Added modules: `pyPPI`, `pyGSEA`, `pyWGCNA`, `pyTCGA`, `pyDEG`.

### bulk2single module:
- Introduced `bulk2single.save` for manual model saving.

## v 1.2.1-4
### single module:
- Added `pySCSA` module with functions: `cell_anno`, `cell_anno_print`, `cell_auto_anno`, `get_model_tissue`.
- Implemented doublet cell filtering in `single.scanpy_lazy`.
- Added `single.scanpy_cellanno_from_dict` for easier annotation.
- Updated SCSA database from [CellMarker2.0](http://bio-bigdata.hrbmu.edu.cn/CellMarker/).
- Fixed errors in SCSA database keys: `Ensembl_HGNC` and `Ensembl_Mouse`.

## v 1.2.5
### single module:
- Added `pyVIA` module with functions: `run`, `plot_piechart_graph`, `plot_stream`, `plot_trajectory_gams`, `plot_lineage_probability`, `plot_gene_trend`, `plot_gene_trend_heatmap`, `plot_clustergraph`.
- Fixed warning error in `utils.pyomic_plot_set`.
- Updated requirements, including `pybind11`, `hnswlib`, `termcolor`, `pygam`, `pillow`, `gdown`.

## v 1.2.6
### single module:
- Added `pyVIA.get_piechart_dict` and `pyVIA.get_pseudotime`.

## v 1.2.7
### bulk2single module:
- Added `Single2Spatial` module with functions: `load`, `save`, `train`, `spot_assess`.
- Fixed installation errors for packages in pip.

## v 1.2.8
- Fixed pip installation errors.

### bulk2single module:
- Replaced `deep-forest` in `Single2Spatial` with `Neuron Network` for classification tasks.
- Accelerated the entire Single2Spatial inference process using GPU and batch-level estimation by modifying the `predicted_size` setting.

## v 1.2.9
### bulk module:
- Fixed duplicates_index mapping in `Matrix_ID_mapping`.
- Resolved hub genes plot issues in `pyWGCNA.plot_sub_network`.
- Fixed backupgene in `pyGSEA.geneset_enrichment` to support rare species.
- Added matrix plot module in `pyWGCNA.plot_matrix`.

### single module:
- Added `rank_genes_groups` check in `pySCSA`.

### bulk2single module:
- Fixed import error of `deepforest`.

## v 1.2.10
- Renamed the package to `omicverse`.

### single module:
- Fixed argument error in `pySCSA`.

### bulk2single module:
- Updated plot arguments in `bulk2single`.

## v 1.2.11
### bulk module:
- Fixed `wilcoxon` method in `pyDEG.deg_analysis`.
- Added parameter setting for treatment and control group names in `pyDEG.plot_boxplot`.
- Fixed figure display issues in `pyWGCNA.plot_matrix`.
- Fixed category correlation failed by one-hot in `pyWGCNA.analysis_meta_correlation`.
- Fixed network display issues in `pyWGCNA.plot_sub_network` and updated `utils.plot_network` to avoid errors.

## v 1.3.0
### bulk module:
- Added `DEseq2` method to `pyDEG.deg_analysis`.
- Introduced `pyGSEA` module in `bulk`.
- Renamed raw `pyGSEA` to `pyGSE` in `bulk`.
- Added `get_gene_annotation` in `utils` for gene name transformation.

## v 1.3.1
### single module:
- Added `get_celltype_marker` method.

### single module:
- Added `GLUE_pair`, `pyMOFA`, `pyMOFAART` module.
- Added tutorials for `Multi omics analysis by MOFA and GLUE`.
- Updated tutorial for `Multi omics analysis by MOFA`.

## v 1.4.0
### bulk2single module:
- Added `BulkTrajBlend` method.

### single module:
- Fixed errors in `scnocd` model.
- Added `save`, `load`, and `get_pair_dict` in `scnocd` model.

### utils module:
- Added `mde` method.
- Added `gz` format support for `utils.read`.

## v 1.4.1
### preprocess module:
- Added `pp` (preprocess) module with `qc` (quantity control), `hvg` (high variable feature), `pca`.
- Added `data_files` for cell cycle calculation from [Cellula](https://github.com/andrecossa5/Cellula/) and [pegasus](https://github.com/lilab-bcb/pegasus/).

## v 1.4.3
###

 preprocess module:
- Fixed sparse preprocess error in `pp`.
- Fixed trajectory import error in `via`.
- Added gene correlation analysis of trajectory.

## v 1.4.4
### single module:
- Added `panglaodb` database to `pySCSA` module.
- Fixed errors in `pySCSA.cell_auto_anno` when some cell types are not found in clusters.
- Fixed errors in `pySCSA.cell_anno` when `rank_genes_groups` are not consistent with clusters.
- Added `pySIMBA` module in single for batch correction.

### preprocess module:
- Added `store_layers` and `retrieve_layers` in `ov.utils`.
- Added `plot_embedding_celltype` and `plot_cellproportion` in `ov.utils`.

## v 1.4.5
### single module:
- Added `MetaTiME` module to perform cell type annotation automatically in TME.

## v 1.4.12
- Updated `conda install omicverse -c conda-forge`.

### single module:
- Added `pyTOSICA` module to perform cell type migration from reference scRNA-seq in Transformer model.
- Added `atac_concat_get_index`, `atac_concat_inner`, `atac_concat_outer` functions to merge/concatenate scATAC data.
- Fixed `MetaTime.predicted` when Unknown cell type appears.

### preprocess module:
- Added `plot_embedding` in `ov.utils` to plot UMAP in a special color dictionary.

## v 1.4.13
### bulk module:
- Added `mad_filtered` to filter robust genes when calculating the network in `ov.bulk.pyWGCNA` module.
- Fixed `string_interaction` in `ov.bulk.pyPPI` for string-db updates.

### preprocess module:
- Changed `mode` argument of `pp.preprocess` to control preprocessing steps.
- Added `ov.utils.embedding`, `ov.utils.neighbors`, and `ov.utils.stacking_vol`.

## v 1.4.14
### preprocess module:
- Added `batch_key` in `pp.preprocess` and `pp.qc`.

### utils module:
- Added `plot_ConvexHull` to visualize the boundary of clusters.
- Added `weighted_knn_trainer` and `weighted_knn_transfer` for multi-adata integration.

### single module:
- Fixed import errors in `mofa`.

## v 1.4.17
### bulk module:
- Fixed compatibility issues with `pydeseq2` version `0.4.0`.
- Added `bulk.batch_correction` for multi-bulk RNA-seq/microarray samples.

### single module:
- Added `single.batch_correction` for multi-single cell datasets.

### preprocess module:
- Added parameter `layers_add` in `pp.scale`.

## v 1.5.0
### single module:
- Added `cellfategenie` to calculate timing-associated genes/genesets.
- Fixed the name error in `atac_concat_outer`.
- Added more kwargs for `batch_correction`.

### utils module:
- Added `plot_heatmap` to visualize the heatmap of pseudotime.
- Fixed `embedding` when the version of `mpl` is larger than `3.7.0`.
- Added `geneset_wordcloud` to visualize geneset heatmaps of pseudotime.

## v 1.5.1
### single module:
- Added `scLTNN` to infer cell trajectory.

### bulk2single module:
- Updated cell fraction prediction with `TAPE` in bulk2single.
- Fixed group and normalization issues in bulk2single.

### utils module:
- Added `Ro/e` calculation (by: Haihao Zhang).
- Added `cal_paga` and `plot_paga` to visualize the state transfer matrix.
- Fixed the `read` function.

## v 1.5.2
### bulk2single Module:
- Resolved a matrix error occurring when gene symbols are not unique.
- Addressed the `interpolation` issue in `BulkTrajBlend` when target cells do not exist.
- Corrected the `generate` function in `BulkTrajBlend`.
- Rectified the argument for `vae_configure` in `BulkTrajBlend` when `cell_target_num` is set to None.
- Introduced the parameter `max_single_cells` for input in `BulkTrajBlend`.
- Defaulted to using `scaden` for deconvolution in Bulk RNA-seq.

### single Module:
- Fixed an error in `pyVIA` when the root is set to None.
- Added the `TrajInfer` module for inferring cell trajectories.
- Integrated `Palantir` and `Diffusion_map` into the `TrajInfer` module.
- Corrected the parameter error in `batch_correction`.

### utils Module:
- Introduced `plot_pca_variance_ratio` for visualizing the ratio of PCA variance.
- Added the `cluster` and `filtered` module for clustering the cells
- Integrated `MiRA` to calculate the LDA topic

## v 1.5.3
### single Module:
- Added `scVI` and `MIRA` to remove batch effect

### space Module:
- Added `STAGATE` to cluster and denoisy the spatial RNA-seq 

### pp Module:
- Added `doublets` argument of `ov.pp.qc` to control doublets('Default'=True)

## v 1.5.4
### bulk Module:
- Fixed an error in `pyDEG.deg_analysis` when `n_cpus` can not be set in `pyDeseq2(v0.4.3)`

### single Module:
- Fixed an argument error in `single.batch_correction` of combat

### utils Module:
- Added `venn4` plot to visualize
- Fixed the label visualization of `plot_network`
- Added `ondisk` argument of `LDA_topic`

### space Module:
- Added `Tangram` to mapping the scRNA-seq to stRNA-seq

## v 1.5.5
### pp Module:
- Added `max_cells_ratio` and `max_genes_ratio` to control the max threshold in qc of scRNA-seq

### single Module:
- Added `SEACells` model to calculate the metacells from scRNA-seq

### space Module:
- Added `STAligner` to integrate multi stRNA-seq

## v 1.5.6
### pp Module
- Added `mt_startswith` argument to control the `qc` in mouse or other species.

### utils Module
- Added `schist` method to cluster the single cell RNA-seq

### single Module
- Fixed the import error of `palantir` in SEACells
- Added `CEFCON` model to identify the driver regulators of cell fate decisions

### bulk2single Module
- Added `use_rep` and `neighbor_rep` argument to configure the nocd 

### space Module
- Added `SpaceFlow` to identify the pseudo-spatial map

## v 1.5.8

### pp Module
- Added `score_genes_cell_cycle` function to calculate the cell cycle

### bulk Module
- Fixed `dds.plot_volcano` text plot error when the version of `adjustText` larger than `0.9`

### single Module
- Optimised `MetaCell.load` model loading logic
- Fixed an error when loading the model usng `MetaCell.load`
- Added tutorials of `Metacells`

### pl Module

Add `pl` as a unified drawing prefix for the next release, to replace the drawing functionality in the original utils, while retaining the drawing in the original utils.

- Added `embedding` to plot the embedding of scRNA-seq using `ov.pl.embedding`
- Added `optim_palette` to provide a spatially constrained approach that generates discriminate color assignments for visualizing single-cell spatial data in various scenarios
- Added `cellproportion` to plot the proportion of stack bar of scRNA-seq
- Added `embedding_celltype` to plot the figures both celltype proportion and embedding
- Added `ConvexHull` to plot the ConvexHull around the target cells
- Added `embedding_adjust` to adjust the text of celltype legend in embedding
- Added `embedding_density` to plot the category density in the cells
- Added `bardotplot` to plot the bardotplot between different groups.
- Added `add_palue` to plot the p-threshold between different groups.
- Added `embedding_multi` to support the `mudata` object
- Added `purple_color` to visualize the purple palette.
- Added `venn` to plot the venn from set 2 to set 4
- Added `boxplot` to visualize the boxdotplot
- Added `volcano` to visualzize the result of differential expressed genes

## v 1.5.9

### single Module

- Added `slingshot` in `single.TrajInfer`
- Fixed some error of `scLTNN`