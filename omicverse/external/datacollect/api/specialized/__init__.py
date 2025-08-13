"""
Specialized API clients for datacollect module.
"""

from .iucn import IUCNClient
from .regulomedb import RegulomeDBClient
from .mpd import MPDClient
from .paleobiology import PaleobiologyClient
from .cbioportal import cBioPortalClient
from .worms import WoRMSClient
from .jaspar import JASPARClient
from .blast import BLASTClient
from .pride import PRIDEClient

__all__ = ["IUCNClient", "RegulomeDBClient", "MPDClient", "PaleobiologyClient", "cBioPortalClient", "WoRMSClient", "JASPARClient", "BLASTClient", "PRIDEClient"]
