"""
Protein-related API clients for DataCollect.

This module provides API clients for protein databases including:
- UniProt: Protein sequences, annotations, features
- PDB: 3D protein structures  
- AlphaFold: AI-predicted protein structures
- InterPro: Protein domains and families
- STRING: Protein-protein interactions
- EMDB: Electron Microscopy Data Bank
"""

from .uniprot import UniProtClient
from .pdb import PDBClient  
from .alphafold import AlphaFoldClient
from .interpro import InterProClient
from .string import STRINGClient
from .emdb import EMDBClient

__all__ = [
    'UniProtClient',
    'PDBClient', 
    'AlphaFoldClient',
    'InterProClient',
    'STRINGClient',
    'EMDBClient'
]