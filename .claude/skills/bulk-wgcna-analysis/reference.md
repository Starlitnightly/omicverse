# WGCNA workflow quick commands

```python
import pandas as pd
from statsmodels import robust
import omicverse as ov
import scanpy as sc
import matplotlib.pyplot as plt

ov.plot_set()

# --- Load and filter by variance ---
data = ov.utils.read('data/5xFAD_paper/expressionList.csv', index_col=0)
# MAD (median absolute deviation) filtering: keeps genes with highest expression variability.
# WGCNA needs variable genes to find meaningful co-expression patterns.
# Top 2000 genes is typical; use 3000-5000 for larger datasets.
gene_mad = data.apply(robust.mad)
data = data.T.loc[gene_mad.sort_values(ascending=False).index[:2000]]

# --- Initialize PyWGCNA ---
# species: affects gene annotation. Options: 'homo sapiens', 'mus musculus', etc.
# save=True: writes intermediate files (TOM, dendrograms) — set False for large datasets to save disk.
pyWGCNA_5xFAD = ov.bulk.pyWGCNA(name='5xFAD_2k',
                                species='mus musculus',
                                geneExp=data.T,  # genes × samples
                                outputPath='',
                                save=True)

# --- Network construction ---
pyWGCNA_5xFAD.preprocess()  # Drops low-expression genes and outlier samples

# Soft-threshold power: transforms correlation to adjacency.
# Picks the lowest power where scale-free topology fit R² > 0.85.
# Typical values: 6-12 for signed networks, 4-8 for unsigned.
pyWGCNA_5xFAD.calculate_soft_threshold()

pyWGCNA_5xFAD.calculating_adjacency_matrix()
pyWGCNA_5xFAD.calculating_TOM_similarity_matrix()  # Topological Overlap Matrix

# --- Module detection ---
pyWGCNA_5xFAD.calculate_geneTree()  # Hierarchical clustering dendrogram

# deepSplit: controls module granularity. 0=few large modules, 4=many small modules.
# pamRespectsDendro=False: allows PAM to reassign genes across dendrogram branches.
pyWGCNA_5xFAD.calculate_dynamicMods(kwargs_function={'cutreeHybrid': {'deepSplit': 2,
                                                                     'pamRespectsDendro': False}})

# softPower: must match the power chosen by calculate_soft_threshold().
# If unsure, check pyWGCNA_5xFAD.power after running calculate_soft_threshold().
pyWGCNA_5xFAD.calculate_gene_module(kwargs_function={'moduleEigengenes': {'softPower': 8}})
pyWGCNA_5xFAD.plot_matrix(save=False)  # TOM/adjacency heatmap

# --- Inspect specific modules ---
sub_mol = pyWGCNA_5xFAD.get_sub_module(['gold', 'lightgreen'], mod_type='module_color')

# correlation_threshold: minimum edge weight to include in sub-network.
# Lower = denser network (more edges), higher = sparser (only strong co-expression).
G_sub = pyWGCNA_5xFAD.get_sub_network(mod_list=['lightgreen'],
                                      mod_type='module_color',
                                      correlation_threshold=0.2)
pyWGCNA_5xFAD.plot_sub_network(['gold', 'lightgreen'], pos_type='kamada_kawai',
                              pos_scale=10, pos_dim=2, figsize=(8, 8), node_size=10,
                              label_fontsize=8, correlation_threshold=0.2,
                              label_bbox={'ec': 'white', 'fc': 'white', 'alpha': 0.6})

# --- Metadata and trait analysis ---
pyWGCNA_5xFAD.updateSampleInfo(path='data/5xFAD_paper/sampleInfo.csv', sep=',')
# Color maps must be set BEFORE plotting eigengene heatmaps
pyWGCNA_5xFAD.setMetadataColor('Sex', {'Female': 'green', 'Male': 'yellow'})
pyWGCNA_5xFAD.setMetadataColor('Genotype', {'5xFADWT': 'darkviolet', '5xFADHEMI': 'deeppink'})
pyWGCNA_5xFAD.setMetadataColor('Age', {'4mon': 'thistle', '8mon': 'plum', '12mon': 'violet', '18mon': 'purple'})
pyWGCNA_5xFAD.setMetadataColor('Tissue', {'Hippocampus': 'red', 'Cortex': 'blue'})

pyWGCNA_5xFAD.analyseWGCNA()  # Computes module-trait correlations and p-values
metadata = pyWGCNA_5xFAD.datExpr.obs.columns.tolist()
pyWGCNA_5xFAD.plotModuleEigenGene('lightgreen', metadata, show=True)
pyWGCNA_5xFAD.barplotModuleEigenGene('lightgreen', metadata, show=True)

# Hub genes: most connected genes within a module. Top 10 is standard for reporting.
pyWGCNA_5xFAD.top_n_hub_genes(moduleName='lightgreen', n=10)
```
