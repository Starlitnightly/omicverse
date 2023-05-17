r"""
bulk (A omic framework for bulk omic analysis)
"""

#from Pyomic.bulk.DeGene import find_DEG,Density_norm,Plot_gene_expression,ID_mapping
#from Pyomic.bulk.Gene_module import pywgcna

from ._Gene_module import pyWGCNA
from ._Enrichment import pyGSEA,pyGSE,geneset_enrichment,geneset_plot,geneset_enrichment_GSEA
from ._network import pyPPI,string_interaction,string_map,generate_G
from ._chm13 import get_chm13_gene,find_chm13_gene
from ._Deseq2 import pyDEG,deseq2_normalize,estimateSizeFactors,estimateDispersions,Matrix_ID_mapping,data_drop_duplicates_index
from ._tcga import pyTCGA