"""
Proteins API clients for datacollect module.
"""

from .uniprot import UniProtClient
from .alphafold import AlphaFoldClient
from .emdb import EMDBClient
from .pdb import PDBClient
from .interpro import InterProClient
from .string import STRINGClient
from .pdb_simple import SimplePDBClient

__all__ = ["UniProtClient", "AlphaFoldClient", "EMDBClient", "PDBClient", "InterProClient", "STRINGClient", "SimplePDBClient"]
