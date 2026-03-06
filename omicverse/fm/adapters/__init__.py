"""
Model Adapters for Single-Cell Foundation Models
==================================================

Each adapter handles:
- Loading checkpoints and tokenizers
- Preprocessing data according to model requirements
- Running inference
- Writing results to AnnData with standardized keys

Only adapters whose upstream libraries are installed will be importable.
Use :func:`omicverse.fm.registry.get_registry().get_adapter_class` for
safe, lazy-loaded access.
"""

from .base import BaseAdapter

# Bridge adapters that wrap existing omicverse.llm implementations.
# Import errors are silently ignored — the registry's lazy-load mechanism
# will report missing adapters at runtime.
try:
    from ._scgpt import ScGPTAdapter
except ImportError:
    ScGPTAdapter = None

try:
    from ._geneformer import GeneformerAdapter
except ImportError:
    GeneformerAdapter = None

try:
    from ._uce import UCEAdapter
except ImportError:
    UCEAdapter = None

try:
    from ._scfoundation import ScFoundationAdapter
except ImportError:
    ScFoundationAdapter = None

try:
    from ._cellplm import CellPLMAdapter
except ImportError:
    CellPLMAdapter = None

try:
    from ._experimental import (
        ScBERTAdapter,
        GeneCompassAdapter,
        NicheformerAdapter,
        ScMulanAdapter,
        TGPTAdapter,
        CellFMAdapter,
        ScCelloAdapter,
        ScPrintAdapter,
        AiDocellAdapter,
        PulsarAdapter,
        AtacformerAdapter,
        ScPlantLLMAdapter,
        LangCellAdapter,
        Cell2SentenceAdapter,
        GenePTAdapter,
        ChatCellAdapter,
        TabulaAdapter,
    )
except ImportError:
    ScBERTAdapter = None
    GeneCompassAdapter = None
    NicheformerAdapter = None
    ScMulanAdapter = None
    TGPTAdapter = None
    CellFMAdapter = None
    ScCelloAdapter = None
    ScPrintAdapter = None
    AiDocellAdapter = None
    PulsarAdapter = None
    AtacformerAdapter = None
    ScPlantLLMAdapter = None
    LangCellAdapter = None
    Cell2SentenceAdapter = None
    GenePTAdapter = None
    ChatCellAdapter = None
    TabulaAdapter = None

__all__ = [
    "BaseAdapter",
    "ScGPTAdapter",
    "GeneformerAdapter",
    "UCEAdapter",
    "ScFoundationAdapter",
    "CellPLMAdapter",
    "ScBERTAdapter",
    "GeneCompassAdapter",
    "NicheformerAdapter",
    "ScMulanAdapter",
    "TGPTAdapter",
    "CellFMAdapter",
    "ScCelloAdapter",
    "ScPrintAdapter",
    "AiDocellAdapter",
    "PulsarAdapter",
    "AtacformerAdapter",
    "ScPlantLLMAdapter",
    "LangCellAdapter",
    "Cell2SentenceAdapter",
    "GenePTAdapter",
    "ChatCellAdapter",
    "TabulaAdapter",
]
