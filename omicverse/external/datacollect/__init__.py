"""
DataCollect module for OmicVerse - Comprehensive bioinformatics data collection.

This module provides access to 29+ biological databases with seamless OmicVerse integration,
including proteins, genomics, expression, and pathway data.
"""

# API clients
from .api.proteins import *
from .api.genomics import *
from .api.expression import *
from .api.pathways import *
from .api.specialized import *

# Collectors
from .collectors import *

# Utils and format converters
from .utils.omicverse_adapters import to_anndata, to_pandas, to_mudata
from .utils.validation import SequenceValidator, IdentifierValidator
from .utils.transformers import SequenceTransformer, DataNormalizer

# Configuration
from .config.config import settings

__version__ = "1.0.0"
__author__ = "DataCollect2BioNMI Team"
__description__ = "Comprehensive bioinformatics data collection for OmicVerse"

# Main convenience functions
def collect_protein_data(identifier, **kwargs):
    """Collect protein data from multiple sources."""
    from .collectors.uniprot_collector import UniProtCollector
    collector = UniProtCollector()
    return collector.collect_single(identifier, **kwargs)

def collect_expression_data(accession, **kwargs):
    """Collect gene expression data."""
    from .collectors.geo_collector import GEOCollector
    collector = GEOCollector()
    return collector.collect_single(accession, **kwargs)

def collect_pathway_data(pathway_id, **kwargs):
    """Collect pathway data."""
    from .collectors.kegg_collector import KEGGCollector
    collector = KEGGCollector()
    return collector.collect_single(pathway_id, **kwargs)

__all__ = [
    # Main functions
    "collect_protein_data",
    "collect_expression_data", 
    "collect_pathway_data",
    # Format converters
    "to_anndata",
    "to_pandas",
    "to_mudata",
    # Validators
    "SequenceValidator",
    "IdentifierValidator",
    # Transformers
    "SequenceTransformer",
    "DataNormalizer",
    # Settings
    "settings",
]
