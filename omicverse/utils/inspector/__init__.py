"""
DataStateInspector - Runtime validation for OmicVerse prerequisite tracking.

This module provides runtime validation of prerequisite chains by inspecting
actual AnnData object state. It verifies that required data structures exist
and prerequisite functions have been executed before allowing operations.

Main Components:
    DataStateInspector: Main orchestrator for validation
    DataValidators: Validates data structure requirements
    PrerequisiteChecker: Detects executed functions (coming soon)
    SuggestionEngine: Generates fix suggestions (coming soon)
    LLMFormatter: Formats results for LLM integration (coming soon)

Usage:
    >>> from omicverse.utils.inspector import DataStateInspector
    >>> from omicverse.utils.registry import get_registry
    >>>
    >>> inspector = DataStateInspector(adata, get_registry())
    >>> result = inspector.validate_prerequisites('leiden')
    >>>
    >>> if not result.is_valid:
    ...     print(result.message)
    ...     for suggestion in result.suggestions:
    ...         print(suggestion.code)
"""

from .inspector import DataStateInspector
from .validators import DataValidators
from .data_structures import (
    ValidationResult,
    DataCheckResult,
    ObsCheckResult,
    ObsmCheckResult,
    ObspCheckResult,
    UnsCheckResult,
    LayersCheckResult,
)

__all__ = [
    'DataStateInspector',
    'DataValidators',
    'ValidationResult',
    'DataCheckResult',
    'ObsCheckResult',
    'ObsmCheckResult',
    'ObspCheckResult',
    'UnsCheckResult',
    'LayersCheckResult',
]

__version__ = '0.1.0'
