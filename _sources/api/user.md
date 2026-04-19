# User API

Import OmicVerse as:

```python
import omicverse as ov
```

This page is auto-generated from `@register_function` entries in the OmicVerse registry.

Public registry entries listed here: 301

```{eval-rst}
.. currentmodule:: omicverse
```

## Top-Level API

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   generate_reference_table
```

## Settings

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   settings.cpu_gpu_mixed_init
   settings.gpu_init
```

## Data IO

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   io.load
   io.read
   io.read_10x_h5
   io.read_10x_mtx
   io.read_csv
   io.read_h5ad
   io.read_nanostring
   io.read_visium_hd
   io.read_visium_hd_bin
   io.read_visium_hd_seg
   io.read_xenium
   io.save
   io.spatial.read_visium
   io.write_visium_hd_cellseg
```

## Alignment

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   alignment.bulk_rnaseq_pipeline
   alignment.count
   alignment.fastp
   alignment.featureCount
   alignment.fqdump
   alignment.parallel_fastq_dump
   alignment.prefetch
   alignment.ref
   alignment.STAR
```

## Preprocessing (`pp`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pp.anndata_to_CPU
   pp.anndata_to_GPU
   pp.binary_search
   pp.filter_cells
   pp.filter_genes
   pp.highly_variable_features
   pp.highly_variable_genes
   pp.identify_robust_genes
   pp.leiden
   pp.log1p
   pp.louvain
   pp.mde
   pp.neighbors
   pp.normalize_pearson_residuals
   pp.pca
   pp.preprocess
   pp.qc
   pp.recover_counts
   pp.regress
   pp.regress_and_scale
   pp.remove_cc_genes
   pp.scale
   pp.score_genes_cell_cycle
   pp.scrublet
   pp.scrublet_simulate_doublets
   pp.select_hvf_pegasus
   pp.sude
   pp.tsne
   pp.umap
```

## Single-cell (`single`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   single.Annotation
   single.AnnotationRef
   single.autoResolution
   single.batch_correction
   single.CellOntologyMapper
   single.CellVote
   single.cNMF
   single.convert_human_to_mouse_network
   single.cosg
   single.cytotrace2
   single.DCT
   single.DEG
   single.download_cl
   single.Drug_Response
   single.dynamic_features
   single.factor_correlation
   single.factor_exact
   single.Fate
   single.find_markers
   single.gene_trends
   single.generate_scRNA_report
   single.geneset_aucell
   single.get_celltype_marker
   single.get_cluster_celltype
   single.get_markers
   single.get_obs_value
   single.get_weights
   single.GLUE_pair
   single.gptcelltype
   single.gptcelltype_local
   single.hematopoiesis
   single.lazy
   single.load_human_prior_interaction_network
   single.MetaCell
   single.MetaTiME
   single.Monocle
   single.mouse_hsc_nestorowa16
   single.pathway_aucell
   single.pathway_aucell_enrichment
   single.pathway_enrichment
   single.pathway_enrichment_plot
   single.plot_metacells
   single.pyCEFCON
   single.pyMOFA
   single.pyMOFAART
   single.pySCSA
   single.pySIMBA
   single.pyTOSICA
   single.pyVIA
   single.run_cellphonedb_v5
   single.scanpy_cellanno_from_dict
   single.SCENIC
   single.TrajInfer
   single.Velo
```

## Bulk RNA-seq (`bulk`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   bulk.batch_correction
   bulk.Deconvolution
   bulk.geneset_enrichment
   bulk.geneset_plot
   bulk.geneset_plot_multi
   bulk.Matrix_ID_mapping
   bulk.pyDEG
   bulk.pyGSEA
   bulk.pyPPI
   bulk.pyTCGA
   bulk.pyWGCNA
   bulk.readWGCNA
   bulk.string_interaction
```

## Spatial transcriptomics (`space`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   space.bin2cell
   space.Cal_Spatial_Net
   space.calculate_gene_signature
   space.CAST
   space.cellcharter
   space.CellLoc
   space.CellMap
   space.clusters
   space.create_communication_anndata
   space.crop_space_visium
   space.Deconvolution
   space.GASTON
   space.map_spatial_auto
   space.map_spatial_manual
   space.merge_cluster
   space.moranI
   space.pySpaceFlow
   space.pySTAGATE
   space.pySTAligner
   space.read_visium_10x
   space.rotate_space_visium
   space.salvage_secondary_labels
   space.spatial_autocorr
   space.spatial_neighbors
   space.STT
   space.svg
   space.sync_visium_hd_seg_geometries
   space.Tangram
   space.update_classification_from_database
   space.visium_10x_hd_cellpose_expand
   space.visium_10x_hd_cellpose_gex
   space.visium_10x_hd_cellpose_he
```

## Bulk-to-Single (`bulk2single`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   bulk2single.Bulk2Single
   bulk2single.bulk2single_plot_cellprop
   bulk2single.bulk2single_plot_correlation
   bulk2single.BulkTrajBlend
   bulk2single.Single2Spatial
```

## Plotting (`pl`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.add_density_contour
   pl.add_palue
   pl.add_pie2spatial
   pl.add_streamplot
   pl.bardotplot
   pl.boxplot
   pl.calculate_gene_density
   pl.ccc_heatmap
   pl.ccc_network_plot
   pl.ccc_stat_plot
   pl.cell_cor_heatmap
   pl.CellChatViz
   pl.cellproportion
   pl.complexheatmap
   pl.contour
   pl.ConvexHull
   pl.create_custom_colormap
   pl.dotplot
   pl.dynamic_heatmap
   pl.dynamic_trends
   pl.embedding
   pl.embedding_adjust
   pl.embedding_atlas
   pl.embedding_celltype
   pl.embedding_density
   pl.feature_heatmap
   pl.ForbiddenCity
   pl.gen_mpl_labels
   pl.geneset_wordcloud
   pl.group_heatmap
   pl.marker_heatmap
   pl.markers_dotplot
   pl.palette
   pl.plot_cellproportion
   pl.plot_ConvexHull
   pl.plot_embedding_celltype
   pl.plot_flowsig_network
   pl.plot_grouped_fractions
   pl.plot_pca_variance_ratio
   pl.plot_set
   pl.plot_spatial
   pl.plot_text_set
   pl.pyomic_palette
   pl.rank_genes_groups_dotplot
   pl.single_group_boxplot
   pl.stacking_vol
   pl.tsne
   pl.umap
   pl.venn
   pl.violin
   pl.volcano
```

## Datasets

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   datasets.bhattacherjee
   datasets.blobs
   datasets.bm
   datasets.bone_marrow
   datasets.burczynski06
   datasets.chromaffin
   datasets.cite_seq
   datasets.create_mock_dataset
   datasets.decov_bulk_covid_bulk
   datasets.decov_bulk_covid_single
   datasets.dentate_gyrus
   datasets.dentate_gyrus_scvelo
   datasets.download_data
   datasets.download_data_requests
   datasets.get_adata
   datasets.gillespie
   datasets.haber
   datasets.hematopoiesis
   datasets.hematopoiesis_raw
   datasets.hg_forebrain_glutamatergic
   datasets.hl60
   datasets.human_tfs
   datasets.krumsiek11
   datasets.moignard15
   datasets.multi_brain_5k
   datasets.nascseq
   datasets.pancreas_cellrank
   datasets.pancreatic_endocrinogenesis
   datasets.paul15
   datasets.pbmc3k
   datasets.pbmc8k
   datasets.sc_ref_Lymph_Node
   datasets.sceu_seq_organoid
   datasets.sceu_seq_rpe1
   datasets.scifate
   datasets.scnt_seq_neuron_labeling
   datasets.scnt_seq_neuron_splicing
   datasets.scslamseq
   datasets.seqfish
   datasets.toggleswitch
   datasets.zebrafish
```

## External Integrations (`external`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   external.GraphST
```

## Utilities (`utils`)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   utils.biocontext.call_tool
   utils.biocontext.get_ensembl_id
   utils.biocontext.get_fulltext
   utils.biocontext.get_uniprot_id
   utils.biocontext.list_tools
   utils.biocontext.query_alphafold
   utils.biocontext.query_cell_ontology
   utils.biocontext.query_chebi
   utils.biocontext.query_efo
   utils.biocontext.query_go
   utils.biocontext.query_hpa
   utils.biocontext.query_interpro
   utils.biocontext.query_opentargets
   utils.biocontext.query_panglaodb
   utils.biocontext.query_reactome
   utils.biocontext.query_string
   utils.biocontext.query_uniprot
   utils.biocontext.search_clinical_trials
   utils.biocontext.search_drugs
   utils.biocontext.search_interpro
   utils.biocontext.search_literature
   utils.biocontext.search_preprints
   utils.biocontext.search_pride
   utils.cal_paga
   utils.cluster
   utils.convert2gene_id
   utils.convert2gene_symbol
   utils.convert2symbol
   utils.convert_adata_for_rust
   utils.convert_to_pandas
   utils.download_CaDRReS_model
   utils.download_GDSC_data
   utils.download_geneid_annotation_pair
   utils.download_pathway_database
   utils.download_tosica_gmt
   utils.geneset_prepare
   utils.get_gene_annotation
   utils.gtf_to_pair_tsv
   utils.LDA_topic
   utils.mde
   utils.plot_paga
   utils.refine_label
   utils.retrieve_layers
   utils.roe
   utils.store_layers
   utils.symbol2id
   utils.weighted_knn_trainer
   utils.weighted_knn_transfer
   utils.wrap_dataframe
```
