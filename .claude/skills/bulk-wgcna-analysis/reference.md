# WGCNA workflow quick commands

```python
import pandas as pd
from statsmodels import robust
import omicverse as ov
import scanpy as sc
import matplotlib.pyplot as plt

ov.plot_set()

data = ov.utils.read('data/5xFAD_paper/expressionList.csv', index_col=0)
gene_mad = data.apply(robust.mad)
data = data.T.loc[gene_mad.sort_values(ascending=False).index[:2000]]

pyWGCNA_5xFAD = ov.bulk.pyWGCNA(name='5xFAD_2k',
                                species='mus musculus',
                                geneExp=data.T,
                                outputPath='',
                                save=True)

pyWGCNA_5xFAD.preprocess()
pyWGCNA_5xFAD.calculate_soft_threshold()
pyWGCNA_5xFAD.calculating_adjacency_matrix()
pyWGCNA_5xFAD.calculating_TOM_similarity_matrix()

pyWGCNA_5xFAD.calculate_geneTree()
pyWGCNA_5xFAD.calculate_dynamicMods(kwargs_function={'cutreeHybrid': {'deepSplit': 2,
                                                                     'pamRespectsDendro': False}})
pyWGCNA_5xFAD.calculate_gene_module(kwargs_function={'moduleEigengenes': {'softPower': 8}})
pyWGCNA_5xFAD.plot_matrix(save=False)

sub_mol = pyWGCNA_5xFAD.get_sub_module(['gold', 'lightgreen'], mod_type='module_color')
G_sub = pyWGCNA_5xFAD.get_sub_network(mod_list=['lightgreen'],
                                      mod_type='module_color',
                                      correlation_threshold=0.2)
pyWGCNA_5xFAD.plot_sub_network(['gold', 'lightgreen'], pos_type='kamada_kawai',
                              pos_scale=10, pos_dim=2, figsize=(8, 8), node_size=10,
                              label_fontsize=8, correlation_threshold=0.2,
                              label_bbox={'ec': 'white', 'fc': 'white', 'alpha': 0.6})

pyWGCNA_5xFAD.updateSampleInfo(path='data/5xFAD_paper/sampleInfo.csv', sep=',')
pyWGCNA_5xFAD.setMetadataColor('Sex', {'Female': 'green', 'Male': 'yellow'})
pyWGCNA_5xFAD.setMetadataColor('Genotype', {'5xFADWT': 'darkviolet', '5xFADHEMI': 'deeppink'})
pyWGCNA_5xFAD.setMetadataColor('Age', {'4mon': 'thistle', '8mon': 'plum', '12mon': 'violet', '18mon': 'purple'})
pyWGCNA_5xFAD.setMetadataColor('Tissue', {'Hippocampus': 'red', 'Cortex': 'blue'})

pyWGCNA_5xFAD.analyseWGCNA()
metadata = pyWGCNA_5xFAD.datExpr.obs.columns.tolist()
pyWGCNA_5xFAD.plotModuleEigenGene('lightgreen', metadata, show=True)
pyWGCNA_5xFAD.barplotModuleEigenGene('lightgreen', metadata, show=True)

pyWGCNA_5xFAD.top_n_hub_genes(moduleName='lightgreen', n=10)
```
