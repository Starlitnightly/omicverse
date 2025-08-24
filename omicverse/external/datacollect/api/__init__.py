"""
API clients module for DataCollect.

This module provides organized access to 29+ biological database APIs,
categorized by data type for easy discovery and use.

Categories:
- proteins: Protein sequences, structures, and interactions
- genomics: Gene annotations, variants, and genomics data  
- expression: Gene expression and regulatory data
- pathways: Metabolic and signaling pathways
- specialized: Domain-specific databases
"""

# Import base classes
from .base import BaseAPIClient, RateLimiter

# Import organized API clients
from .proteins import *
from .genomics import *
from .expression import * 
from .pathways import *
from .specialized import *

__all__ = [
    'BaseAPIClient', 'RateLimiter',
    # Protein APIs
    'UniProtClient', 'PDBClient', 'AlphaFoldClient', 'InterProClient', 'STRINGClient', 'EMDBClient',
    # Genomics APIs  
    'EnsemblClient', 'ClinVarClient', 'dbSNPClient', 'GnomADClient', 'GWASCatalogClient', 'UCSCClient', 'RegulomeDBClient',
    # Expression APIs
    'GEOClient', 'OpenTargetsClient', 'OpenTargetsGeneticsClient', 'ReMapClient', 'CCREClient',
    # Pathway APIs
    'KEGGClient', 'ReactomeClient', 'GtoPdbClient',
    # Specialized APIs
    'BLASTClient', 'JASPARClient', 'MPDClient', 'IUCNClient', 'PRIDEClient', 'cBioPortalClient', 'WoRMSClient', 'PaleobiologyClient'
]
