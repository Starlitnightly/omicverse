"""
Proteins API clients for datacollect module.
"""

from .uniprot import UniProtClient
from .pdb import PDBClient
from .pdb_simple import PDBSimpleClient  
from .alphafold import AlphaFoldClient
from .interpro import InterProClient
from .string import STRINGClient
from .emdb import EMDBClient

__all__ = ["UniProtClient", "PDBClient", "PDBSimpleClient", "AlphaFoldClient", "InterProClient", "STRINGClient", "EMDBClient"]
