"""
DataStateInspector - Runtime validation for OmicVerse prerequisite tracking.

This module provides runtime validation of prerequisite chains by inspecting
actual AnnData object state. It verifies that required data structures exist
and prerequisite functions have been executed before allowing operations.

Main Components:
    DataStateInspector: Main orchestrator for validation
    DataValidators: Validates data structure requirements
    PrerequisiteChecker: Detects executed functions (Phase 2)
    SuggestionEngine: Generates fix suggestions (Phase 3)
    LLMFormatter: Formats results for LLM integration (Phase 4)

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
from .prerequisite_checker import PrerequisiteChecker, DetectionResult
from .suggestion_engine import SuggestionEngine, WorkflowPlan, WorkflowStep, WorkflowStrategy
from .llm_formatter import LLMFormatter, LLMPrompt, OutputFormat
from .data_structures import (
    ValidationResult,
    DataCheckResult,
    ObsCheckResult,
    ObsmCheckResult,
    ObspCheckResult,
    UnsCheckResult,
    LayersCheckResult,
    ExecutionEvidence,
    Suggestion,
)

__all__ = [
    'DataStateInspector',
    'DataValidators',
    'PrerequisiteChecker',
    'SuggestionEngine',
    'LLMFormatter',
    'ValidationResult',
    'DataCheckResult',
    'DetectionResult',
    'WorkflowPlan',
    'WorkflowStep',
    'WorkflowStrategy',
    'LLMPrompt',
    'OutputFormat',
    'ObsCheckResult',
    'ObsmCheckResult',
    'ObspCheckResult',
    'UnsCheckResult',
    'LayersCheckResult',
    'ExecutionEvidence',
    'Suggestion',
]

__version__ = '0.4.0'
