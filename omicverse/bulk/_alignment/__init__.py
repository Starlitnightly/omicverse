"""Compatibility layer for bulk alignment API."""

from ._old import (
    Alignment,
    AlignmentConfig,
    EnhancedAlignmentConfig,
    ISeqHandler,
    geo_data_preprocess,
    fq_data_preprocess,
    load_config,
    get_example_configs,
    check_all_tools,
    check_tool_availability,
)

__all__ = [
    "Alignment",
    "AlignmentConfig",
    "EnhancedAlignmentConfig",
    "ISeqHandler",
    "geo_data_preprocess",
    "fq_data_preprocess",
    "load_config",
    "get_example_configs",
    "check_all_tools",
    "check_tool_availability",
]
