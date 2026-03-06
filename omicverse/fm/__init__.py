"""
``ov.fm`` — Foundation Model Module
====================================

Unified API for discovering, selecting, executing, and interpreting
single-cell foundation models.

Quick start::

    import omicverse as ov

    # List available models
    ov.fm.list_models(task="embed")

    # Profile your data
    profile = ov.fm.profile_data("my_data.h5ad")

    # Select best model
    result = ov.fm.select_model("my_data.h5ad", task="embed")

    # Run inference
    output = ov.fm.run(task="embed", model_name="scgpt", adata_path="my_data.h5ad")

    # Interpret results
    ov.fm.interpret_results("output.h5ad", task="embed")
"""

from .registry import (
    GeneIDScheme,
    ModelRegistry,
    ModelSpec,
    Modality,
    SkillReadyStatus,
    TaskType,
    get_registry,
)
from .api import (
    list_models,
    describe_model,
    profile_data,
    select_model,
    preprocess_validate,
    run,
    interpret_results,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelSpec",
    "TaskType",
    "Modality",
    "GeneIDScheme",
    "SkillReadyStatus",
    "get_registry",
    # API functions
    "list_models",
    "describe_model",
    "profile_data",
    "select_model",
    "preprocess_validate",
    "run",
    "interpret_results",
]