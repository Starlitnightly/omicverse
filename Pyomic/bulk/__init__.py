r"""
bulk (A omic framework for bulk omic analysis)
"""

#from Pyomic.bulk.DeGene import find_DEG,Density_norm,Plot_gene_expression,ID_mapping
#from Pyomic.bulk.Gene_module import pywgcna

from ._Gene_module import pywgcna
from ._Enrichment import enrichment_KEGG,enrichment_GO,enrichment_GSEA,Plot_GSEA,geneset_enrichment,geneset_plot
from ._DeGene import find_DEG,ID_mapping,Drop_dupligene
from ._network import string_interaction,string_map,generate_G
from ._chm13 import get_chm13_gene,find_chm13_gene
from ._Deseq2 import pyDEseq,deseq2_normalize,estimateSizeFactors,estimateDispersions,Matrix_ID_mapping
from ._tcga import TCGA