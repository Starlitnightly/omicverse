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
    Production API: Convenient factory functions and helpers (Phase 5)
    AgentContextInjector: Injects prerequisite state into LLM prompts (Layer 3 Phase 1)
    DataStateValidator: Pre-execution validation with auto-correction (Layer 3 Phase 2)
    WorkflowEscalator: Intelligent workflow complexity analysis and escalation (Layer 3 Phase 3)
    AutoPrerequisiteInserter: Automatic prerequisite insertion into code (Layer 3 Phase 4)

Usage (Quick Start):
    >>> from omicverse.utils.inspector import create_inspector, validate_function
    >>>
    >>> # Quick validation
    >>> result = validate_function(adata, 'leiden')
    >>> if not result.is_valid:
    ...     print(result.message)
    >>>
    >>> # Or create inspector for multiple validations
    >>> inspector = create_inspector(adata)
    >>> result = inspector.validate_prerequisites('leiden')
    >>> if not result.is_valid:
    ...     print(result.message)
    ...     for suggestion in result.suggestions:
    ...         print(suggestion.code)

Advanced Usage:
    >>> # Get workflow suggestions
    >>> workflow = get_workflow_suggestions(adata, 'leiden', strategy='comprehensive')
    >>>
    >>> # Batch validate multiple functions
    >>> results = batch_validate(adata, ['pca', 'neighbors', 'leiden'])
    >>>
    >>> # Generate validation report
    >>> report = get_validation_report(adata, format='markdown')
    >>>
    >>> # Use as decorator
    >>> @check_prerequisites('leiden')
    ... def my_function(adata):
    ...     # Your code here
    ...     pass
    >>>
    >>> # Use as context manager
    >>> with ValidationContext(adata, 'leiden') as ctx:
    ...     if ctx.is_valid:
    ...         # Run your analysis
    ...         pass
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
# Phase 5: Production API
from .production_api import (
    create_inspector,
    clear_inspector_cache,
    validate_function,
    explain_requirements,
    check_prerequisites,
    get_workflow_suggestions,
    batch_validate,
    get_validation_report,
    ValidationContext,
)
# Layer 3 Phase 1: Agent Context Injection
from .agent_context_injector import AgentContextInjector, ConversationState
# Layer 3 Phase 2: Data State Validator
from .data_state_validator import DataStateValidator, ValidationFeedback, validate_code
# Layer 3 Phase 3: Workflow Escalator
from .workflow_escalator import WorkflowEscalator, EscalationResult, EscalationStrategy, analyze_and_escalate
# Layer 3 Phase 4: Auto Prerequisite Inserter
from .auto_prerequisite_inserter import AutoPrerequisiteInserter, InsertionResult, InsertionPolicy, auto_insert_prerequisites
# Shared data structures
from .data_structures import ComplexityLevel

__all__ = [
    # Core components
    'DataStateInspector',
    'DataValidators',
    'PrerequisiteChecker',
    'SuggestionEngine',
    'LLMFormatter',
    # Data structures
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
    # Production API (Phase 5)
    'create_inspector',
    'clear_inspector_cache',
    'validate_function',
    'explain_requirements',
    'check_prerequisites',
    'get_workflow_suggestions',
    'batch_validate',
    'get_validation_report',
    'ValidationContext',
    # Layer 3 Phase 1: Agent Context Injection
    'AgentContextInjector',
    'ConversationState',
    # Layer 3 Phase 2: Data State Validator
    'DataStateValidator',
    'ValidationFeedback',
    'validate_code',
    # Layer 3 Phase 3: Workflow Escalator
    'WorkflowEscalator',
    'EscalationResult',
    'EscalationStrategy',
    'analyze_and_escalate',
    # Layer 3 Phase 4: Auto Prerequisite Inserter
    'AutoPrerequisiteInserter',
    'InsertionResult',
    'InsertionPolicy',
    'auto_insert_prerequisites',
    # Shared data structures
    'ComplexityLevel',
]

__version__ = '1.0.0'  # Layer 3 COMPLETE - All 4 phases implemented
