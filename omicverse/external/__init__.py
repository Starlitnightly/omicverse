r"""
External algorithms and specialized methods collection.

This module provides access to a comprehensive collection of external algorithms
and specialized methods that extend omicverse's core functionality. These include
state-of-the-art methods for spatial transcriptomics, trajectory inference,
multi-omics integration, and advanced single-cell analysis.

Algorithm categories:
    Spatial transcriptomics:
        - STAGATE_pyG: Spatial domain identification
        - STAligner: Multi-sample spatial alignment
        - GraphST: Graph-based spatial analysis
        - PROST: Pattern recognition for spatial data
        - spatrio: Spatial transcriptomics analysis
        - CAST: Cellular spatial organization
        - starfysh: Spatial factorization
        
    Trajectory inference:
        - STT: Spatial transition tensor
        - VIA: Velocity integration and annotation
        - cytotrace2: Developmental potential scoring
        
    Multi-omics integration:
        - mofapy2: Multi-Omics Factor Analysis
        - GNTD: Graph-regularized non-negative tensor decomposition
        - scMulan: Multi-modal analysis
        
    Single-cell methods:
        - tosica: Trajectory inference and cell fate
        - cnmf: Consensus non-negative matrix factorization
        - CEFCON: Cell fate controller networks
        - cellanova: Cell type-aware analysis
        - BINARY: Binary classification methods
        - flowsig: Flow cytometry-style analysis
        - scdiffusion: Diffusion-based methods
        
    Network analysis:
        - PyWGCNA: Weighted gene co-expression networks
        - commot: Cell-cell communication
        
    Data integration:
        - scanorama: Batch correction and integration
        - scSLAT: Single-cell alignment
        
    Spatial methods:
        - spaceflow: Spatial flow analysis
        - gaston: Spatial depth estimation

Features:
    - Optional dependencies: Algorithms are loaded as needed
    - Consistent interfaces: Wrapped for omicverse compatibility
    - Performance optimized: GPU acceleration where available
    - Well documented: Each method includes comprehensive examples

Notes:
    - Some algorithms require additional dependencies
    - GPU methods require appropriate hardware and drivers
    - All methods are thoroughly tested and validated
    - Regular updates incorporate latest algorithmic advances
"""
from . import (scSLAT,CEFCON,mofapy2,GNTD,spaceflow,STT,
               tosica,STAGATE_pyG,STAligner,spatrio,PROST,cytotrace2,
               GraphST,commot,cnmf,starfysh,flowsig,PyWGCNA,
               CAST,scanorama,scdiffusion,BINARY,cellanova,VIA,gaston,pyscenic,
                bin2cell,sude_py,harmony
               )
__all__ = [
    'scSLAT',
    'CEFCON',
    'mofapy2',
    'GNTD',
    'spaceflow',
    'STT',
    'tosica',
    'STAGATE_pyG',
    'STAligner',
    'spatrio',
    'PROST',
    'cytotrace2',
    'GraphST',
    'commot',
    'cnmf',
    'starfysh',
    #'scMulan',
    'flowsig',
    'PyWGCNA',
    'CAST',
    'scanorama',
    'scdiffusion',
    'BINARY',
    'cellanova',
    'VIA',
    'gaston',
    'pyscenic',
    'bin2cell',
    'sude_py',
    'harmony',
]
