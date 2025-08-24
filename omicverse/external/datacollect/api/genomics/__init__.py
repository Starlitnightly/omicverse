"""
Genomics-related API clients for DataCollect.

This module provides API clients for genomics databases including:
- Ensembl: Gene annotations, sequences, variants
- ClinVar: Clinical significance of variants
- dbSNP: Single nucleotide polymorphisms
- gnomAD: Population genetics data
- GWAS Catalog: Genome-wide association studies
- UCSC: Genome browser data
- RegulomeDB: Regulatory variants
"""

from .ensembl import EnsemblClient
from .clinvar import ClinVarClient
from .dbsnp import dbSNPClient
# Export both canonical and legacy-cased names for compatibility
from .gnomad import GnomADClient as gnomADClient  # legacy alias
from .gnomad import GnomADClient
from .gwas_catalog import GWASCatalogClient
from .ucsc import UCSCClient
from .regulomedb import RegulomeDBClient

__all__ = [
    'EnsemblClient',
    'ClinVarClient',
    'dbSNPClient',
    'GnomADClient',
    'gnomADClient',  # legacy export
    'GWASCatalogClient',
    'UCSCClient',
    'RegulomeDBClient',
]
