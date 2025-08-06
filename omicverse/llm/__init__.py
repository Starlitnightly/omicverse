"""
scLLM: Single-cell Language Models
==================================

A unified interface for single-cell language models including scGPT, scFoundation and others.

Main classes:
- SCLLMManager: High-level interface for model operations (RECOMMENDED)
- ModelFactory: Factory for creating different model types
- ScGPTModel: scGPT model implementation
- ScFoundationModel: scFoundation model implementation
- CellPLMModel: CellPLM model implementation
- UCEModel: UCE (Universal Cell Embeddings) model implementation

Quick start with SCLLMManager (RECOMMENDED):
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

# Fine-tune for annotation
results = manager.fine_tune(train_adata, task="annotation")

# Integrate batches
integration_results = manager.integrate(adata, batch_key="batch")
```

Quick start with other models:
```python
import omicverse as ov

# scFoundation
manager = ov.external.scllm.SCLLMManager(
    model_type="scfoundation",
    model_path="/path/to/scfoundation/model.ckpt"
)

# CellPLM
manager = ov.external.scllm.SCLLMManager(
    model_type="cellplm",
    model_path="/path/to/cellplm/checkpoint",
    pretrain_version="20231027_85M"
)

# UCE
manager = ov.external.scllm.SCLLMManager(
    model_type="uce",
    model_path="/path/to/uce/model.torch",
    species="human"
)
```

For legacy compatibility, specific model functions are still available
but accessed on-demand via SCLLMManager or from model_factory module directly.
"""

# Core imports - always available
from .base import SCLLMBase, ModelConfig, TaskConfig

from .model_factory import SCLLMManager, ModelFactory


