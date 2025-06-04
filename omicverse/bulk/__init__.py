"""Bulk omics analysis utilities."""

# Heavy functionality lives in submodules and is imported lazily.
from ._Gene_module import pyWGCNA,readWGCNA
from ._Enrichment import pyGSEA,pyGSE,geneset_enrichment,geneset_plot,geneset_enrichment_GSEA,geneset_plot_multi,enrichment_multi_concat
from ._network import pyPPI,string_interaction,string_map,generate_G
from ._chm13 import get_chm13_gene,find_chm13_gene
from ._Deseq2 import pyDEG,deseq2_normalize,estimateSizeFactors,estimateDispersions,Matrix_ID_mapping,data_drop_duplicates_index
from ._tcga import pyTCGA
from ._combat import batch_correction

__all__ = [
    'pyWGCNA',
    'readWGCNA',
    'pyGSEA',
    'pyGSE',
    'geneset_enrichment',
    'geneset_plot',
    'geneset_enrichment_GSEA',
    'geneset_plot_multi',
    'enrichment_multi_concat',
    'pyPPI',
    'string_interaction',
    'string_map',
    'generate_G',
    'get_chm13_gene',
    'find_chm13_gene',
    'pyDEG',
    'deseq2_normalize',
    'estimateSizeFactors',
    'estimateDispersions',
    'Matrix_ID_mapping',
    'data_drop_duplicates_index',
    'pyTCGA',
    'batch_correction',
    'pyDEG',
    'deseq2_normalize',
    'estimateSizeFactors',
    'estimateDispersions',
    'Matrix_ID_mapping',
    'data_drop_duplicates_index',
    'pyTCGA',
    'batch_correction',
]
