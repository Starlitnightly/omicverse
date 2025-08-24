"""
DataCollect module for OmicVerse - Comprehensive bioinformatics data collection.

This module provides access to 29+ biological databases with seamless OmicVerse integration,
including proteins, genomics, expression, and pathway data.

Key Features:
- 29 API clients for major biological databases
- Unified data format conversion to AnnData, pandas, MuData
- Comprehensive error handling and rate limiting
- 597+ passing tests ensuring reliability
- Full backward compatibility

Main API clients:
    Proteins: UniProt, PDB, AlphaFold, InterPro, STRING, EMDB
    Genomics: Ensembl, ClinVar, dbSNP, gnomAD, GWAS Catalog, UCSC  
    Expression: GEO, OpenTargets, ReMap, CCRE
    Pathways: KEGG, Reactome, GtoPdb
    Specialized: BLAST, JASPAR, MPD, IUCN, PRIDE, cBioPortal, RegulomeDB
"""

# Import API clients if all dependencies are available; otherwise defer to lazy imports in wrappers
try:
    from .api import *  # noqa: F401,F403
except Exception as _e:
    # Defer failures to function-level lazy imports so the module remains usable
    _API_IMPORT_ERROR = _e  # type: ignore
else:
    _API_IMPORT_ERROR = None  # type: ignore

# Import collectors
try:
    from .collectors import *
except ImportError:
    pass

# Import utilities
try:
    from .utils.omicverse_adapters import to_pandas, to_anndata, to_mudata
except ImportError:
    pass

# Utility functions for OmicVerse integration
def collect_protein_data(identifier, source='uniprot', to_format='pandas', **kwargs):
    """
    Collect protein data from specified source and convert to desired format.
    
    Parameters:
    - identifier: Protein identifier (UniProt ID, gene name, etc.)
    - source: Data source ('uniprot', 'pdb', 'alphafold', 'interpro', 'string')
    - to_format: Output format ('pandas', 'anndata', 'mudata', 'dict')
    - **kwargs: Additional parameters for the specific client
    
    Returns:
    - Data in specified format
    """
    # Lazy import to avoid failing on optional dependencies at module import time
    try:
        from .api.proteins import (
            UniProtClient,
            PDBClient,
            AlphaFoldClient,
            InterProClient,
            STRINGClient,
        )
    except Exception as e:
        raise ImportError(
            f"Protein API clients unavailable. Ensure dependencies are installed (e.g., httpx). Root cause: {e}"
        )

    source_map = {
        'uniprot': UniProtClient,
        'pdb': PDBClient,
        'alphafold': AlphaFoldClient,
        'interpro': InterProClient,
        'string': STRINGClient,
    }
    
    if source not in source_map:
        raise ValueError(f"Unknown source: {source}. Available: {list(source_map.keys())}")
    
    client = source_map[source]()
    data = client.get_data(identifier, **kwargs)
    
    # Convert to desired format
    if to_format == 'anndata':
        return to_anndata(data)
    elif to_format == 'mudata':
        return to_mudata(data)
    elif to_format == 'pandas':
        return to_pandas(data, "protein")
    else:
        return data

def collect_expression_data(identifier, source='geo', to_format='anndata', **kwargs):
    """
    Collect gene expression data and convert to AnnData format (OmicVerse standard).
    
    Parameters:
    - identifier: Dataset identifier (GEO accession, etc.)
    - source: Data source ('geo', 'ccre')
    - to_format: Output format ('anndata', 'pandas', 'mudata', 'dict')
    - **kwargs: Additional parameters
    
    Returns:
    - Expression data in AnnData format (default) or specified format
    """
    # Lazy import
    try:
        from .api.expression import GEOClient, CCREClient
    except Exception as e:
        raise ImportError(
            f"Expression API clients unavailable. Ensure dependencies are installed. Root cause: {e}"
        )

    source_map = {
        'geo': GEOClient,
        'ccre': CCREClient,
    }
    
    if source not in source_map:
        raise ValueError(f"Unknown source: {source}. Available: {list(source_map.keys())}")
    
    client = source_map[source]()
    data = client.get_data(identifier, **kwargs)
    
    # Convert to desired format (default: anndata for OmicVerse compatibility)
    if to_format == 'anndata':
        from .utils.transformers import to_anndata
        return to_anndata(data)
    elif to_format == 'mudata':
        from .utils.transformers import to_mudata
        return to_mudata(data)
    elif to_format == 'pandas':
        from .utils.transformers import to_pandas
        return to_pandas(data)
    else:
        return data

def collect_pathway_data(identifier, source='kegg', to_format='pandas', **kwargs):
    """
    Collect pathway data from specified source.
    
    Parameters:
    - identifier: Pathway identifier
    - source: Data source ('kegg', 'reactome', 'gtopdb')
    - to_format: Output format ('pandas', 'anndata', 'mudata', 'dict')
    - **kwargs: Additional parameters
    
    Returns:
    - Pathway data in specified format
    """
    # Lazy import
    try:
        from .api.pathways import KEGGClient, ReactomeClient, GtoPdbClient
    except Exception as e:
        raise ImportError(
            f"Pathway API clients unavailable. Ensure dependencies are installed. Root cause: {e}"
        )

    source_map = {
        'kegg': KEGGClient,
        'reactome': ReactomeClient,
        'gtopdb': GtoPdbClient,
    }
    
    if source not in source_map:
        raise ValueError(f"Unknown source: {source}. Available: {list(source_map.keys())}")
    
    client = source_map[source]()
    data = client.get_data(identifier, **kwargs)
    
    # Convert to desired format
    if to_format == 'anndata':
        return to_anndata(data)
    elif to_format == 'mudata':
        return to_mudata(data)
    elif to_format == 'pandas':
        return to_pandas(data, "pathway")
    else:
        return data

__version__ = "1.0.0"
__author__ = "DataCollect Team"

__all__ = [
    # Main collection functions
    'collect_protein_data',
    'collect_expression_data', 
    'collect_pathway_data',
    
    # Protein clients
    'UniProtClient',
    'PDBClient', 
    'AlphaFoldClient',
    'InterProClient',
    'STRINGClient',
    'EMDBClient',
    
    # Genomics clients
    'EnsemblClient',
    'ClinVarClient',
    'dbSNPClient',
    'gnomADClient',
    'GWASCatalogClient',
    'UCSCClient',
    
    # Expression clients
    'GEOClient',
    'CCREClient',
    
    # Pathway clients
    'KEGGClient',
    'ReactomeClient', 
    'GtoPdbClient',
    
    # Specialized clients
    'BLASTClient',
    'JASPARClient',
    'PRIDEClient',
    'cBioPortalClient',
    'RegulomeDBClient',
]
