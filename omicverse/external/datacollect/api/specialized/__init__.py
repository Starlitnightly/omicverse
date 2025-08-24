"""
Specialized API clients for DataCollect.

This module provides API clients for specialized databases including:
- BLAST: Sequence similarity searches
- JASPAR: Transcription factor binding profiles  
- MPD: Mouse Phenome Database
- IUCN: Species conservation status
- PRIDE: Proteomics data repository
- cBioPortal: Cancer genomics data
- WORMS: World Register of Marine Species
- Paleobiology: Fossil and geological data
"""

from .blast import BLASTClient
from .jaspar import JASPARClient
from .mpd import MPDClient
from .iucn import IUCNClient
from .pride import PRIDEClient
from .cbioportal import cBioPortalClient
from .worms import WoRMSClient as WORMSClient  # maintain expected name
from .worms import WoRMSClient  # also expose canonical name
from .paleobiology import PaleobiologyClient

__all__ = [
    'BLASTClient',
    'JASPARClient',
    'MPDClient',
    'IUCNClient',
    'PRIDEClient',
    'cBioPortalClient',
    'WORMSClient',
    'WoRMSClient',
    'PaleobiologyClient'
]
