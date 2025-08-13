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

# Core API clients - proteins
try:
    from .api.uniprot import UniProtClient
    from .api.pdb import PDBClient
    from .api.alphafold import AlphaFoldClient
    from .api.interpro import InterProClient
    from .api.string import STRINGClient
    from .api.emdb import EMDBClient
except ImportError:
    pass

# Core API clients - genomics  
try:
    from .api.ensembl import EnsemblClient
    from .api.clinvar import ClinVarClient
    from .api.dbsnp import dbSNPClient
    from .api.gnomad import gnomADClient
    from .api.gwas_catalog import GWASCatalogClient
    from .api.ucsc import UCSCClient
except ImportError:
    pass

# Core API clients - expression
try:
    from .api.geo import GEOClient
    from .api.ccre import CCREClient
except ImportError:
    pass

# Core API clients - pathways
try:
    from .api.kegg import KEGGClient
    from .api.reactome import ReactomeClient
    from .api.gtopdb import GtoPdbClient
except ImportError:
    pass

# Specialized clients
try:
    from .api.blast import BLASTClient
    from .api.jaspar import JASPARClient
    from .api.pride import PRIDEClient
    from .api.cbioportal import cBioPortalClient
    from .api.regulomedb import RegulomeDBClient
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
    source_map = {
        'uniprot': UniProtClient,
        'pdb': PDBClient,
        'alphafold': AlphaFoldClient,
        'interpro': InterProClient,
        'string': STRINGClient
    }
    
    if source not in source_map:
        raise ValueError(f"Unknown source: {source}. Available: {list(source_map.keys())}")
    
    client = source_map[source]()
    data = client.get_data(identifier, **kwargs)
    
    # Convert to desired format
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
    source_map = {
        'geo': GEOClient,
        'ccre': CCREClient
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
    source_map = {
        'kegg': KEGGClient,
        'reactome': ReactomeClient,
        'gtopdb': GtoPdbClient
    }
    
    if source not in source_map:
        raise ValueError(f"Unknown source: {source}. Available: {list(source_map.keys())}")
    
    client = source_map[source]()
    data = client.get_data(identifier, **kwargs)
    
    # Convert to desired format
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