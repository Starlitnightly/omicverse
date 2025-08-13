"""
Expression and regulation-related API clients for DataCollect.

This module provides API clients for gene expression databases including:
- GEO: Gene expression datasets
- OpenTargets: Drug target information  
- OpenTargets Genetics: Genetics evidence
- ReMap: Transcriptional regulators
- CCRE: Candidate cis-regulatory elements
"""

from .geo import GEOClient
from .opentargets import OpenTargetsClient
from .opentargets_genetics import OpenTargetsGeneticsClient
from .remap import ReMapClient
from .ccre import CCREClient

__all__ = [
    'GEOClient',
    'OpenTargetsClient',
    'OpenTargetsGeneticsClient',
    'ReMapClient',
    'CCREClient'
]