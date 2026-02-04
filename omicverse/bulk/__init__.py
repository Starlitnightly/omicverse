r"""
Bulk omics analysis utilities.

This module provides comprehensive tools for bulk RNA-seq analysis including:
- Differential expression analysis with DESeq2
- Gene co-expression network analysis (WGCNA)
- Pathway enrichment analysis (GSEA)
- Protein-protein interaction networks
- Batch effect correction
- Multi-omics integration

Classes:
    pyDEG: Differential expression analysis using DESeq2
    pyWGCNA: Weighted Gene Co-expression Network Analysis
    pyGSEA: Gene Set Enrichment Analysis
    pyPPI: Protein-Protein Interaction analysis
    pyTCGA: TCGA data analysis utilities

Functions:
    geneset_enrichment: Perform pathway enrichment analysis
    string_interaction: Retrieve STRING database interactions
    batch_correction: Correct for batch effects
    get_chm13_gene: Map genes to CHM13 reference

Examples:
    >>> import omicverse as ov
    >>> # Differential expression analysis
    >>> dds = ov.bulk.pyDEG(count_matrix, design_matrix)
    >>> dds.drop_duplicates_index()
    >>> dds.deseq2()
    >>> deg_results = dds.get_results()
    >>> 
    >>> # WGCNA analysis
    >>> wgcna = ov.bulk.pyWGCNA(adata)
    >>> wgcna.calculate_correlation()
    >>> wgcna.module_detection()
    """

# Heavy functionality lives in submodules and is imported lazily.
from ._Gene_module import pyWGCNA,readWGCNA
from ._Enrichment import pyGSEA,pyGSE,geneset_enrichment,geneset_plot,geneset_enrichment_GSEA,geneset_plot_multi,enrichment_multi_concat
from ._network import pyPPI,string_interaction,string_map,generate_G
from ._chm13 import get_chm13_gene,find_chm13_gene
from ._Deseq2 import pyDEG,deseq2_normalize,estimateSizeFactors,estimateDispersions,Matrix_ID_mapping,data_drop_duplicates_index
from ._tcga import pyTCGA
from ._combat import batch_correction
from ._decov import Deconvolution

__all__ = [

    # Gene co-expression analysis
    'pyWGCNA',
    'readWGCNA',
    
    # Pathway enrichment analysis
    'pyGSEA',
    'pyGSE',
    'geneset_enrichment',
    'geneset_plot',
    'geneset_enrichment_GSEA',
    'geneset_plot_multi',
    'enrichment_multi_concat',
    
    # Protein-protein interaction networks
    'pyPPI',
    'string_interaction',
    'string_map',
    'generate_G',
    
    # Genome reference utilities
    'get_chm13_gene',
    'find_chm13_gene',
    
    # Differential expression analysis
    'pyDEG',
    'deseq2_normalize',
    'estimateSizeFactors',
    'estimateDispersions',
    'Matrix_ID_mapping',
    'data_drop_duplicates_index',
    
    # TCGA analysis
    'pyTCGA',
    
    # Batch correction
    'batch_correction',

    # Deconvolution
    'Deconvolution',
]
