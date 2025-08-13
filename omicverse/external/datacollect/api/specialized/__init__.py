"""
Specialized API clients for datacollect module.
"""

from .blast import BlastClient
from .jaspar import JasparClient
from .mpd import MpdClient
from .iucn import IucnClient
from .pride import PrideClient
from .cbioportal import CbioportalClient
from .regulomedb import RegulomedbClient
from .worms import WormsClient
from .paleobiology import PaleobiologyClient

__all__ = ["BlastClient", "JasparClient", "MpdClient", "IucnClient", "PrideClient", "CbioportalClient", "RegulomedbClient", "WormsClient", "PaleobiologyClient"]
