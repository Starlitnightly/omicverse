# Layer 1: Comprehensive Prerequisite Metadata Review Plan

**Date:** 2025-11-11
**Status:** Planning Phase
**Total Registered Functions:** 110

---

## Executive Summary

Out of 110 registered functions, we need to prioritize **42 functions** for prerequisite metadata based on:
- Workflow dependency complexity
- Common usage patterns by ov.agent
- Potential for user errors (e.g., calling PCA on raw data)
- Data structure requirements

**Already Completed:** 5 functions (pca, scale, preprocess, qc, neighbors)
**Remaining High Priority:** 15 functions
**Remaining Medium Priority:** 22 functions
**Low Priority (No Prerequisites):** 68 functions

---

## Categorization Methodology

### **Tier 1: Critical Workflow Functions (Need Prerequisites)**
Functions that:
- Are commonly chained together in workflows
- Require specific data structures to exist
- Can fail cryptically without proper preprocessing
- Are frequently used by ov.agent

### **Tier 2: Standalone Functions (No Prerequisites)**
Functions that:
- Work on raw data
- Are primarily data loading/saving operations
- Are visualization functions (operate on existing results)
- Are utility functions with no dependencies

---

## Category 1: PREPROCESSING (13 functions)

### ✅ Already Completed (5 functions)

| Function | Prerequisites | Requires | Auto-fix | Status |
|----------|---------------|----------|----------|--------|
| `qc()` | None | None | none | ✅ Done |
| `preprocess()` | qc (optional) | None | none | ✅ Done |
| `scale()` | normalize/qc (optional) | None | none | ✅ Done |
| `pca()` | scale | layers['scaled'] | escalate | ✅ Done |
| `neighbors()` | pca (optional) | obsm['X_pca'] | auto | ✅ Done |

### 🔴 High Priority - Need Prerequisites (4 functions)

#### 1. **`umap()`** - UMAP Embedding
```python
prerequisites={
    'functions': ['neighbors'],
    'optional_functions': ['pca']
},
requires={
    'uns': ['neighbors'],
    'obsp': ['connectivities', 'distances']
},
produces={
    'obsm': ['X_umap']
},
auto_fix='auto'  # Simple: just needs neighbors
```
**Reasoning:** Requires neighbor graph. Common failure when called without neighbors. Simple auto-fix.

#### 2. **`leiden()`** - Leiden Clustering
```python
prerequisites={
    'functions': ['neighbors'],
    'optional_functions': ['pca', 'umap']
},
requires={
    'uns': ['neighbors'],
    'obsp': ['connectivities']
},
produces={
    'obs': ['leiden']
},
auto_fix='auto'  # Simple: just needs neighbors
```
**Reasoning:** Critical clustering function. Requires neighbor graph. Common in workflows. Simple auto-fix.

#### 3. **`score_genes_cell_cycle()`** - Cell Cycle Scoring
```python
prerequisites={
    'optional_functions': ['qc', 'normalize']
},
requires={},  # Works on normalized or raw, but better on normalized
produces={
    'obs': ['S_score', 'G2M_score', 'phase']
},
auto_fix='none'  # Optional preprocessing
```
**Reasoning:** Often used after QC/normalization. No hard requirements but better with preprocessing.

#### 4. **`sude()`** - SUDE Dimensionality Reduction
```python
prerequisites={
    'optional_functions': ['scale', 'preprocess']
},
requires={
    'layers': ['scaled']  # Can use X_pca if available
},
produces={
    'obsm': ['X_sude']
},
auto_fix='escalate'  # Needs preprocessing like PCA
```
**Reasoning:** Alternative to UMAP/tSNE. Similar requirements to PCA.

### 🟢 Low Priority - No Prerequisites (4 functions)

- `anndata_to_GPU()` - Data transfer utility (no prerequisites)
- `anndata_to_CPU()` - Data transfer utility (no prerequisites)
- `recover_counts()` - Reverse operation (no prerequisites)
- `wrapper()` - Internal registry function (no prerequisites)

---

## Category 2: SINGLE CELL ANALYSIS (25 functions)

### 🔴 High Priority - Need Prerequisites (6 functions)

#### 5. **`cell_anno()` (pySCSA)** - Cell Annotation
```python
prerequisites={
    'optional_functions': ['preprocess', 'leiden']
},
requires={
    'var': ['highly_variable_features'],  # Recommended
    'obs': ['leiden']  # Recommended for cluster-based annotation
},
produces={
    'obs': ['celltype']
},
auto_fix='none'  # Optional but recommended
```
**Reasoning:** Works better with preprocessed data and clustering. Common annotation workflow.

#### 6. **`batch_correction()`** - Batch Correction
```python
prerequisites={
    'optional_functions': ['qc', 'preprocess']
},
requires={},  # Can work on raw or preprocessed
produces={
    'obsm': ['X_pca'],  # Depending on method
    'layers': ['corrected']
},
auto_fix='none'  # Flexible input
```
**Reasoning:** Better with QC'd data. Common in multi-batch experiments.

#### 7. **`cosg()`** - Marker Gene Identification
```python
prerequisites={
    'functions': ['leiden'],  # Or any clustering
},
requires={
    'obs': ['leiden']  # Or any cluster column
},
produces={
    'uns': ['cosg']
},
auto_fix='escalate'  # Needs clustering first
```
**Reasoning:** Requires clustering labels. Common failure without clusters.

#### 8. **`cytotrace2()`** - Cell Potency Prediction
```python
prerequisites={
    'optional_functions': ['qc', 'preprocess']
},
requires={},
produces={
    'obs': ['cytotrace2_score', 'cytotrace2_potency']
},
auto_fix='none'  # Works on raw or preprocessed
```
**Reasoning:** Works better with preprocessed data but not required.

#### 9. **`get_celltype_marker()`** - Get Marker Genes
```python
prerequisites={
    'functions': [],  # Needs cluster column
},
requires={
    'obs': []  # Dynamic: any cluster column
},
produces={
    'uns': ['rank_genes_groups']
},
auto_fix='none'  # User specifies cluster column
```
**Reasoning:** Needs clustering but flexible on which column.

#### 10. **`palantir_cal_branch()`** (TrajInfer)** - Trajectory Inference
```python
prerequisites={
    'functions': ['pca', 'neighbors'],
    'optional_functions': ['leiden']
},
requires={
    'obsm': ['X_pca'],
    'uns': ['neighbors']
},
produces={
    'obs': ['palantir_pseudotime'],
    'obsm': ['X_palantir']
},
auto_fix='auto'  # Can auto-insert pca+neighbors
```
**Reasoning:** Requires dimensionality reduction and neighbors. Common trajectory workflow.

### 🟡 Medium Priority - Context-Dependent (7 functions)

#### 11. **`scanpy_cellanno_from_dict()`** - Manual Annotation
```python
prerequisites={
    'optional_functions': ['leiden']
},
requires={
    'obs': []  # Needs cluster column to map
},
produces={
    'obs': ['celltype']
},
auto_fix='none'
```

#### 12. **`gptcelltype()`** / **`gptcelltype_local()`** - AI Annotation
```python
prerequisites={
    'optional_functions': ['leiden', 'cosg']
},
requires={
    'obs': ['leiden']  # Needs clusters
},
produces={
    'obs': ['celltype']
},
auto_fix='none'
```

#### 13. **`get_cluster_celltype()`** (CellVote) - Cluster Annotation
```python
prerequisites={
    'functions': ['leiden']
},
requires={
    'obs': ['leiden']
},
produces={
    'obs': ['celltype']
},
auto_fix='escalate'  # Needs clustering
```

#### 14. **`run()` (DCT)** - Differential Composition Analysis
```python
prerequisites={
    'functions': [],
},
requires={
    'obs': ['celltype', 'condition']  # User-specified columns
},
produces={
    'uns': ['dct_results']
},
auto_fix='none'  # User provides columns
```

#### 15. **`run()` (DEG)** - Differential Expression
```python
prerequisites={
    'optional_functions': ['preprocess']
},
requires={
    'obs': ['celltype', 'condition']  # User-specified
},
produces={
    'uns': ['deg_results']
},
auto_fix='none'  # Works on raw or preprocessed
```

#### 16. **`train()` (MetaCell)** - Metacell Construction
```python
prerequisites={
    'optional_functions': ['qc', 'preprocess']
},
requires={},
produces={
    'obs': ['metacell']
},
auto_fix='none'
```

#### 17. **`run()` (pyVIA)** - VIA Trajectory
```python
prerequisites={
    'functions': ['pca'],
    'optional_functions': ['neighbors']
},
requires={
    'obsm': ['X_pca']
},
produces={
    'obs': ['via_pseudotime']
},
auto_fix='auto'
```

### 🟢 Low Priority - Specialized/Standalone (12 functions)

- `mouse_hsc_nestorowa16()` - Data loading (no prerequisites)
- `load_human_prior_interaction_network()` - Data loading
- `convert_human_to_mouse_network()` - Utility function
- `preprocess()` (CEFCON-specific) - Internal function
- `get_related_peak()` (CellFateGenie) - Specialized analysis
- `setup_llm_expansion()` (CellOntologyMapper) - Configuration
- `download_cl()` - Data download utility
- `plot_metacells()` - Visualization (no prerequisites)
- `get_obs_value()` - Data extraction utility
- `hematopoiesis()` - Data loading
- Other specialized/internal functions

---

## Category 3: SPATIAL TRANSCRIPTOMICS (20 functions)

### 🔴 High Priority - Need Prerequisites (4 functions)

#### 18. **`train()` (pySTAGATE)** - Spatial Clustering
```python
prerequisites={
    'functions': ['Cal_Spatial_Net'],  # Build spatial graph first
},
requires={
    'uns': ['spatial_net']  # Or obsp spatial adjacency
},
produces={
    'obs': ['STAGATE_cluster']
},
auto_fix='auto'  # Can auto-insert spatial network
```
**Reasoning:** Requires spatial neighborhood graph. Common spatial workflow.

#### 19. **`clusters()` (Multi-method Spatial Clustering)** - Spatial Clustering
```python
prerequisites={
    'functions': ['Cal_Spatial_Net'],
    'optional_functions': ['svg']
},
requires={
    'uns': ['spatial_net']
},
produces={
    'obs': ['spatial_cluster']
},
auto_fix='auto'
```

#### 20. **`svg()` - Spatially Variable Genes**
```python
prerequisites={
    'optional_functions': ['qc']
},
requires={
    'obsm': ['spatial']  # Spatial coordinates
},
produces={
    'var': ['spatially_variable']
},
auto_fix='none'  # Just needs spatial coords
```

#### 21. **`train()` (STAligner)** - Spatial Integration
```python
prerequisites={
    'functions': ['Cal_Spatial_Net']
},
requires={
    'uns': ['spatial_net'],
    'obsm': ['spatial']
},
produces={
    'obsm': ['STAligner']
},
auto_fix='auto'
```

### 🟡 Medium Priority - Spatial-Specific (6 functions)

#### 22. **`Cal_Spatial_Net()` - Build Spatial Network**
```python
prerequisites={},  # First step
requires={
    'obsm': ['spatial']
},
produces={
    'uns': ['spatial_net'],
    'obsp': ['spatial_distances']
},
auto_fix='none'  # First step in spatial workflows
```
**Reasoning:** Foundation for spatial analysis. No prerequisites.

#### 23. **`train()` (Tangram)** - Spatial Deconvolution
```python
prerequisites={
    'optional_functions': ['preprocess']  # For reference data
},
requires={},  # Needs both spatial and scRNA-seq data
produces={
    'obsm': ['tangram_pred']
},
auto_fix='none'  # Cross-modality mapping
```

#### 24. **`forward()` (SpaceFlow)** - Spatial Flow
```python
prerequisites={
    'optional_functions': ['Cal_Spatial_Net']
},
requires={
    'obsm': ['spatial']
},
produces={
    'obsm': ['X_spaceflow']
},
auto_fix='none'
```

#### 25. **`train()` (STT)** - Spatial Transition Tensor
```python
prerequisites={
    'functions': ['Cal_Spatial_Net']
},
requires={
    'uns': ['spatial_net']
},
produces={
    'uns': ['stt']
},
auto_fix='auto'
```

#### 26. **`get_top_pearson_residuals()` (GASTON)**
```python
prerequisites={
    'optional_functions': ['qc']
},
requires={},
produces={
    'var': ['highly_variable']
},
auto_fix='none'
```

#### 27. **`merge_cluster()` - Merge Spatial Clusters**
```python
prerequisites={
    'functions': ['clusters']
},
requires={
    'obs': ['spatial_cluster']
},
produces={
    'obs': ['merged_cluster']
},
auto_fix='none'  # Post-clustering refinement
```

### 🟢 Low Priority - Data Loading/Utility (10 functions)

- `crop_space_visium()` - Data manipulation (no prerequisites)
- `rotate_space_visium()` - Data manipulation
- `map_spatial_auto()` - Data alignment utility
- `map_spatial_manual()` - Data alignment utility
- `read_visium_10x()` - Data loading (no prerequisites)
- `visium_10x_hd_cellpose_he()` - Image processing
- `visium_10x_hd_cellpose_expand()` - Image processing
- `visium_10x_hd_cellpose_gex()` - Image processing
- `salvage_secondary_labels()` - Post-processing utility
- `bin2cell()` - Data conversion utility

---

## Category 4: BULK RNA-SEQ (8 functions)

### 🟡 Medium Priority (2 functions)

#### 28. **`batch_correction()` (bulk)** - Combat Batch Correction
```python
prerequisites={
    'optional_functions': ['Matrix_ID_mapping']
},
requires={
    'obs': ['batch']  # Batch column
},
produces={
    'layers': ['corrected']
},
auto_fix='none'
```

#### 29. **`string_interaction()` / **`pyPPI()`** - PPI Network
```python
prerequisites={},  # Works on gene lists
requires={},
produces={
    'uns': ['ppi_network']
},
auto_fix='none'  # Standalone analysis
```

### 🟢 Low Priority (6 functions)

- `Matrix_ID_mapping()` - Data preprocessing utility
- `drop_duplicates_index()` - Data cleaning utility
- `geneset_plot()` - Visualization (no prerequisites)
- `adata_read()` (TCGA) - Data loading
- `readWGCNA()` - Model loading
- `findModules()` (WGCNA) - Standalone analysis

---

## Category 5: VISUALIZATION (18 functions) - All Low Priority

**Reasoning:** Visualization functions operate on existing results. They don't have prerequisites in the traditional sense—they just need the data structures they visualize to exist. The user explicitly specifies what to plot.

**Examples:**
- `volcano()`, `venn()`, `boxplot()` - Plot results
- `embedding()`, `dotplot()`, `violin()` - Plot existing data
- All visualization functions work on whatever data the user provides

**No prerequisite metadata needed.**

---

## Category 6: UTILS (26 functions)

### 🟡 Medium Priority (4 functions)

#### 30. **`cluster()` - Clustering Wrapper**
```python
prerequisites={
    'optional_functions': ['pca', 'neighbors']
},
requires={},  # Flexible
produces={
    'obs': []  # Dynamic cluster column
},
auto_fix='none'
```

#### 31. **`refine_label()` - Label Refinement**
```python
prerequisites={
    'functions': ['cluster']  # Or any clustering
},
requires={
    'obs': [],  # Any cluster column
    'uns': ['neighbors']  # Optionally
},
produces={
    'obs': []  # Refined cluster column
},
auto_fix='none'
```

#### 32. **`mde()` - MDE Dimensionality Reduction**
```python
prerequisites={
    'optional_functions': ['pca']
},
requires={},  # Can work on raw or PCA
produces={
    'obsm': ['X_mde']
},
auto_fix='none'
```

#### 33. **`weighted_knn_transfer()` - KNN Transfer**
```python
prerequisites={
    'functions': ['weighted_knn_trainer']
},
requires={},  # Needs trained model
produces={
    'obs': []  # Transferred labels
},
auto_fix='none'
```

### 🟢 Low Priority (22 functions)

All utility functions:
- `gpu_init()`, `cpu_gpu_mixed_init()` - Configuration
- `read()` - Data loading
- `download_*()` - Data download utilities
- `convert_*()` - Data conversion utilities
- `plot_*()` - Visualization utilities
- `store_layers()`, `retrieve_layers()` - Data management
- Other internal utilities

---

## Summary: Prioritized Implementation Plan

### Phase 1: Critical Workflow Functions (11 functions) - **Week 1**

**Already Done (5):** qc, preprocess, scale, pca, neighbors

**Remaining (6):**
1. ✅ `umap()` - UMAP embedding
2. ✅ `leiden()` - Leiden clustering
3. ✅ `score_genes_cell_cycle()` - Cell cycle scoring
4. ✅ `sude()` - SUDE dimensionality reduction
5. ✅ `cosg()` - Marker gene identification
6. ✅ `palantir_cal_branch()` - Trajectory inference

### Phase 2: Annotation & Analysis Functions (9 functions) - **Week 2**

7. ✅ `cell_anno()` - SCSA annotation
8. ✅ `batch_correction()` (single) - Batch correction
9. ✅ `cytotrace2()` - Cell potency
10. ✅ `get_celltype_marker()` - Marker genes
11. ✅ `scanpy_cellanno_from_dict()` - Manual annotation
12. ✅ `gptcelltype()` - GPT annotation
13. ✅ `get_cluster_celltype()` - Cluster annotation
14. ✅ `run()` (DEG) - Differential expression
15. ✅ `run()` (pyVIA) - VIA trajectory

### Phase 3: Spatial Functions (8 functions) - **Week 3**

16. ✅ `Cal_Spatial_Net()` - Spatial network (foundation)
17. ✅ `train()` (pySTAGATE) - Spatial clustering
18. ✅ `clusters()` - Multi-method spatial clustering
19. ✅ `svg()` - Spatially variable genes
20. ✅ `train()` (STAligner) - Spatial integration
21. ✅ `train()` (Tangram) - Spatial deconvolution
22. ✅ `train()` (STT) - Spatial transition tensor
23. ✅ `merge_cluster()` - Merge clusters

### Phase 4: Specialized Functions (8 functions) - **Week 4**

24. ✅ `forward()` (SpaceFlow) - Spatial flow
25. ✅ `get_top_pearson_residuals()` (GASTON)
26. ✅ `batch_correction()` (bulk) - Combat
27. ✅ `cluster()` - Clustering wrapper
28. ✅ `refine_label()` - Label refinement
29. ✅ `mde()` - MDE embedding
30. ✅ `weighted_knn_transfer()` - KNN transfer
31. ✅ `run()` (DCT) - Differential composition
32. ✅ `train()` (MetaCell) - Metacell construction

---

## Functions That DO NOT Need Prerequisites (68 functions)

### Visualization (18 functions)
- All `pl.*` functions - operate on user-specified data

### Data Loading (12 functions)
- `read()`, `adata_read()`, `read_visium_10x()`
- `hematopoiesis()`, `mouse_hsc_nestorowa16()`
- Other data loading utilities

### Data Processing Utilities (15 functions)
- `anndata_to_GPU()`, `anndata_to_CPU()`
- `convert_*()`, `download_*()`
- `Matrix_ID_mapping()`, `drop_duplicates_index()`
- Image processing functions

### Configuration & Internal (23 functions)
- `gpu_init()`, `cpu_gpu_mixed_init()`
- `wrapper()`, `setup_llm_expansion()`
- Plot utilities, data management functions
- Other internal/configuration functions

---

## Decision Criteria for Prerequisite Metadata

A function **NEEDS** prerequisite metadata if:
1. ✅ It requires specific data structures (layers, obsm, uns, obsp)
2. ✅ It depends on output from other functions
3. ✅ It commonly fails without proper preprocessing
4. ✅ It's part of a standard analysis workflow
5. ✅ The ov.agent frequently generates code using it

A function **DOES NOT NEED** prerequisite metadata if:
1. ❌ It's a data loading function
2. ❌ It's a visualization function (user specifies what to plot)
3. ❌ It's a utility/configuration function
4. ❌ It works independently on any data
5. ❌ It has no structural requirements

---

## Expected Impact

### After Phase 1 (11 functions total)
- ✅ Core preprocessing workflow complete
- ✅ PCA/UMAP/Leiden chain tracked
- ✅ Most common agent failures prevented
- ✅ ~80% of typical single-cell workflows covered

### After Phase 2 (20 functions total)
- ✅ Annotation workflows tracked
- ✅ Trajectory inference prerequisites clear
- ✅ Differential expression guided
- ✅ ~90% of single-cell analysis covered

### After Phase 3 (28 functions total)
- ✅ Spatial transcriptomics workflows complete
- ✅ Cross-modality analysis tracked
- ✅ ~95% of spatial workflows covered

### After Phase 4 (36 functions total)
- ✅ All critical analysis functions covered
- ✅ Edge cases handled
- ✅ ~99% of ov.agent use cases tracked

---

## Testing Strategy

For each phase, test:

1. **Registry Query Tests**
   ```python
   info = _global_registry.get_prerequisites('umap')
   assert info['required_functions'] == ['neighbors']
   ```

2. **Prerequisite Chain Tests**
   ```python
   chain = _global_registry.get_prerequisite_chain('leiden')
   assert chain == ['neighbors', 'leiden']
   ```

3. **Data Validation Tests**
   ```python
   result = _global_registry.check_prerequisites('umap', raw_adata)
   assert result['satisfied'] == False
   assert 'neighbors' in result['missing_structures']
   ```

4. **LLM Prompt Formatting Tests**
   ```python
   prompt = _global_registry.format_prerequisites_for_llm('leiden')
   assert 'neighbors → leiden' in prompt
   ```

---

## Implementation Checklist

### Phase 1 (Week 1) - 6 functions
- [ ] `umap()` - pp/_preprocess.py
- [ ] `leiden()` - pp/_preprocess.py
- [ ] `score_genes_cell_cycle()` - pp/_preprocess.py
- [ ] `sude()` - pp/_sude.py
- [ ] `cosg()` - single/_cosg.py
- [ ] `palantir_cal_branch()` - single/_traj.py

### Phase 2 (Week 2) - 9 functions
- [ ] `cell_anno()` - single/_anno.py
- [ ] `batch_correction()` - single/_batch.py
- [ ] `cytotrace2()` - single/_cytotrace2.py
- [ ] `get_celltype_marker()` - single/_anno.py
- [ ] `scanpy_cellanno_from_dict()` - single/_anno.py
- [ ] `gptcelltype()` - single/_gptcelltype.py
- [ ] `get_cluster_celltype()` - single/_cellvote.py
- [ ] `run()` (DEG) - single/_deg_ct.py
- [ ] `run()` (pyVIA) - single/_via.py

### Phase 3 (Week 3) - 8 functions
- [ ] `Cal_Spatial_Net()` - space/_integrate.py
- [ ] `train()` (pySTAGATE) - space/_cluster.py
- [ ] `clusters()` - space/_cluster.py
- [ ] `svg()` - space/_svg.py
- [ ] `train()` (STAligner) - space/_integrate.py
- [ ] `train()` (Tangram) - space/_tangram.py
- [ ] `train()` (STT) - space/_stt.py
- [ ] `merge_cluster()` - space/_cluster.py

### Phase 4 (Week 4) - 8 functions
- [ ] `forward()` (SpaceFlow) - space/_spaceflow.py
- [ ] `get_top_pearson_residuals()` (GASTON) - space/_gaston.py
- [ ] `batch_correction()` (bulk) - bulk/_combat.py
- [ ] `cluster()` - utils/_cluster.py
- [ ] `refine_label()` - utils/_cluster.py
- [ ] `mde()` - utils/_mde.py
- [ ] `weighted_knn_transfer()` - utils/_knn.py
- [ ] `run()` (DCT) - single/_deg_ct.py

---

## Conclusion

This plan covers **36 out of 110 functions** (32.7%) with prerequisite metadata. The remaining 68 functions are:
- **18 visualization functions** - user-driven, no prerequisites
- **12 data loading functions** - entry points, no prerequisites
- **15 utility functions** - standalone operations
- **23 configuration/internal functions** - infrastructure

By focusing on the 36 workflow-critical functions, we achieve:
- ✅ Maximum impact on agent reliability
- ✅ Coverage of all major analysis workflows
- ✅ Prevention of >95% of common prerequisite errors
- ✅ Clear guidance for users on workflow steps

**Next Action:** Begin Phase 1 implementation with the 6 remaining high-priority preprocessing functions.
