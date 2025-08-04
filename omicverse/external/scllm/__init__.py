"""
scLLM: Single-cell Language Models
==================================

A unified interface for single-cell language models including scGPT, scFoundation and others.

Main classes:
- SCLLMManager: High-level interface for model operations
- ModelFactory: Factory for creating different model types
- ScGPTModel: scGPT model implementation
- ScFoundationModel: scFoundation model implementation

Quick start with scGPT:
```python
import omicverse as ov

# Load and use scGPT for cell annotation
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/scgpt/model"
)

# Annotate cells
results = manager.annotate_cells(adata)

# Get embeddings
embeddings = manager.get_embeddings(adata)
```

Quick start with scFoundation:
```python
import omicverse as ov

# Load and use scFoundation for embedding extraction
manager = ov.external.scllm.SCLLMManager(
    model_type="scfoundation",
    model_path="/path/to/scfoundation/model.ckpt"
)

# Get cell embeddings
embeddings = manager.get_embeddings(adata)
```
"""

# Import only essential components to avoid dependency issues
from .base import SCLLMBase, ModelConfig, TaskConfig
from .scgpt_model import ScGPTModel
from .scfoundation_model import ScFoundationModel
from .geneformer_model import GeneformerModel
from .model_factory import (
    ModelFactory, SCLLMManager, load_scgpt, annotate_with_scgpt,
    fine_tune_scgpt, predict_celltypes_workflow, end_to_end_scgpt_annotation,
    train_integration_scgpt, integrate_batches_workflow, end_to_end_scgpt_integration,
    integrate_with_scgpt, load_scfoundation, get_embeddings_with_scfoundation,
    end_to_end_scfoundation_embedding, fine_tune_scfoundation, 
    predict_celltypes_with_scfoundation, end_to_end_scfoundation_annotation,
    integrate_with_scfoundation, integrate_batches_with_scfoundation, 
    end_to_end_scfoundation_integration
)

# Optional import of scgpt module - only if dependencies are available
try:
    from . import scgpt
    _scgpt_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"scGPT submodule not fully available due to missing dependencies: {e}")
    scgpt = None
    _scgpt_available = False

__all__ = [
    "SCLLMBase",
    "ModelConfig", 
    "TaskConfig",
    "ScGPTModel",
    "ScFoundationModel",
    "GeneformerModel",
    "ModelFactory",
    "SCLLMManager",
    "load_scgpt",
    "annotate_with_scgpt",
    "fine_tune_scgpt",
    "predict_celltypes_workflow", 
    "end_to_end_scgpt_annotation",
    "train_integration_scgpt",
    "integrate_batches_workflow",
    "end_to_end_scgpt_integration",
    "integrate_with_scgpt",
    "load_scfoundation",
    "get_embeddings_with_scfoundation",
    "end_to_end_scfoundation_embedding",
    "fine_tune_scfoundation",
    "predict_celltypes_with_scfoundation", 
    "end_to_end_scfoundation_annotation",
    "integrate_with_scfoundation",
    "integrate_batches_with_scfoundation",
    "end_to_end_scfoundation_integration",
]

# Add scgpt to __all__ only if available
if _scgpt_available:
    __all__.append("scgpt")

__version__ = "0.1.0"