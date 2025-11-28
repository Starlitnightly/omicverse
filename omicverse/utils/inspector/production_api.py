"""
Production API for OmicVerse DataStateInspector.

This module provides convenient factory functions and integration helpers
for using the DataStateInspector in production workflows.
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
from anndata import AnnData
import warnings

from .inspector import DataStateInspector
from .data_structures import ValidationResult
from .llm_formatter import OutputFormat


# Global cache for inspector instances
_INSPECTOR_CACHE: Dict[int, DataStateInspector] = {}


def create_inspector(
    adata: AnnData,
    registry: Optional[Any] = None,
    cache: bool = True,
) -> DataStateInspector:
    """Create a DataStateInspector instance with smart defaults.

    This is the recommended factory function for creating inspectors.
    It handles registry loading and provides optional caching for performance.

    Args:
        adata: AnnData object to inspect.
        registry: Function registry. If None, attempts to load from omicverse.utils.registry.
        cache: If True, caches inspector instances by adata object id.

    Returns:
        DataStateInspector instance ready for validation.

    Example:
        >>> inspector = create_inspector(adata)
        >>> result = inspector.validate_prerequisites('leiden')

    Raises:
        ValueError: If registry cannot be loaded and none is provided.
    """
    # Try to load registry if not provided
    if registry is None:
        try:
            from omicverse.utils.registry import get_registry
            registry = get_registry()
        except Exception as e:
            raise ValueError(
                "Could not load function registry. Please provide a registry "
                "or ensure omicverse.utils.registry.get_registry() is available."
            ) from e

    # Check cache if enabled
    if cache:
        adata_id = id(adata)
        if adata_id in _INSPECTOR_CACHE:
            cached = _INSPECTOR_CACHE[adata_id]
            # Verify the cached inspector still references the same adata
            if cached.adata is adata and cached.registry is registry:
                return cached

        # Create new inspector and cache it
        inspector = DataStateInspector(adata, registry)
        _INSPECTOR_CACHE[adata_id] = inspector
        return inspector

    # No caching, create new instance
    return DataStateInspector(adata, registry)


def clear_inspector_cache():
    """Clear the global inspector cache.

    Call this if you need to free memory or force re-creation of inspectors.

    Example:
        >>> clear_inspector_cache()
    """
    global _INSPECTOR_CACHE
    _INSPECTOR_CACHE.clear()


def validate_function(
    adata: AnnData,
    function_name: str,
    registry: Optional[Any] = None,
    raise_on_invalid: bool = False,
) -> ValidationResult:
    """Quick validation of prerequisites for a function.

    Convenience function for one-off validations without creating an inspector.

    Args:
        adata: AnnData object to validate.
        function_name: Name of the function to validate prerequisites for.
        registry: Function registry (auto-loaded if None).
        raise_on_invalid: If True, raises ValueError when validation fails.

    Returns:
        ValidationResult with validation details.

    Example:
        >>> result = validate_function(adata, 'leiden')
        >>> if not result.is_valid:
        ...     print(result.message)

    Raises:
        ValueError: If raise_on_invalid=True and validation fails.
    """
    inspector = create_inspector(adata, registry, cache=True)
    result = inspector.validate_prerequisites(function_name)

    if raise_on_invalid and not result.is_valid:
        raise ValueError(
            f"Validation failed for {function_name}: {result.message}\n"
            f"Missing prerequisites: {result.missing_prerequisites}\n"
            f"Missing data: {result.missing_data_structures}"
        )

    return result


def explain_requirements(
    adata: AnnData,
    function_name: str,
    registry: Optional[Any] = None,
    format: str = "markdown",
) -> str:
    """Get a human-readable explanation of what's needed to run a function.

    Args:
        adata: AnnData object to inspect.
        function_name: Name of the function to explain.
        registry: Function registry (auto-loaded if None).
        format: Output format ('markdown', 'plain_text', 'natural').

    Returns:
        Formatted explanation string.

    Example:
        >>> explanation = explain_requirements(adata, 'leiden', format='natural')
        >>> print(explanation)
    """
    inspector = create_inspector(adata, registry, cache=True)
    result = inspector.validate_prerequisites(function_name)

    if format == "natural":
        return inspector.get_natural_language_explanation(function_name)
    elif format == "markdown":
        return inspector.format_for_llm(result, OutputFormat.MARKDOWN)
    elif format == "plain_text":
        return inspector.format_for_llm(result, OutputFormat.PLAIN_TEXT)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'markdown', 'plain_text', or 'natural'.")


def check_prerequisites(
    function_name: str,
    raise_on_invalid: bool = True,
    registry: Optional[Any] = None,
) -> Callable:
    """Decorator to automatically validate prerequisites before function execution.

    This decorator checks prerequisites and either raises an error or warns
    if requirements are not met.

    Args:
        function_name: Name of the function in the registry.
        raise_on_invalid: If True, raises ValueError on validation failure.
                         If False, issues a warning and continues.
        registry: Function registry (auto-loaded if None).

    Returns:
        Decorator function.

    Example:
        >>> @check_prerequisites('leiden', raise_on_invalid=True)
        ... def my_leiden(adata):
        ...     # Your implementation
        ...     pass

    Raises:
        ValueError: If raise_on_invalid=True and validation fails.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(adata: AnnData, *args, **kwargs):
            # Validate prerequisites
            result = validate_function(
                adata,
                function_name,
                registry=registry,
                raise_on_invalid=False,
            )

            if not result.is_valid:
                error_msg = (
                    f"Prerequisites not satisfied for {function_name}:\n"
                    f"  Missing prerequisites: {result.missing_prerequisites}\n"
                    f"  Missing data: {result.missing_data_structures}\n"
                    f"  Suggestions: {len(result.suggestions)} available"
                )

                if raise_on_invalid:
                    raise ValueError(error_msg)
                else:
                    warnings.warn(
                        f"Warning: {error_msg}",
                        UserWarning,
                        stacklevel=2,
                    )

            # Execute the function
            return func(adata, *args, **kwargs)

        return wrapper
    return decorator


def get_workflow_suggestions(
    adata: AnnData,
    function_name: str,
    registry: Optional[Any] = None,
    strategy: str = "minimal",
) -> Dict[str, Any]:
    """Get workflow suggestions to satisfy prerequisites.

    Args:
        adata: AnnData object to inspect.
        function_name: Target function name.
        registry: Function registry (auto-loaded if None).
        strategy: Workflow strategy ('minimal', 'comprehensive', 'alternative').

    Returns:
        Dict containing workflow plan with steps and code.

    Example:
        >>> workflow = get_workflow_suggestions(adata, 'leiden', strategy='comprehensive')
        >>> for step in workflow['steps']:
        ...     print(step['code'])
    """
    from .suggestion_engine import WorkflowStrategy

    inspector = create_inspector(adata, registry, cache=True)
    result = inspector.validate_prerequisites(function_name)

    if result.is_valid:
        return {
            "is_valid": True,
            "message": "All prerequisites satisfied",
            "steps": [],
        }

    # Map string to enum
    strategy_map = {
        "minimal": WorkflowStrategy.MINIMAL,
        "comprehensive": WorkflowStrategy.COMPREHENSIVE,
        "alternative": WorkflowStrategy.ALTERNATIVE,
    }
    workflow_strategy = strategy_map.get(strategy, WorkflowStrategy.COMPREHENSIVE)

    # Generate workflow plan
    workflow_plan = inspector.suggestion_engine.create_workflow_plan(
        function_name=function_name,
        missing_prerequisites=result.missing_prerequisites,
        strategy=workflow_strategy,
    )

    # Format as dict
    return {
        "is_valid": False,
        "message": "Prerequisites missing - workflow plan generated",
        "strategy": strategy,
        "total_steps": len(workflow_plan.steps),
        "estimated_time": f"{workflow_plan.total_time_seconds} seconds",
        "steps": [
            {
                "order": step.order,
                "function": step.function_name,
                "description": step.description,
                "code": step.code_example,
                "time": step.estimated_time,
                "dependencies": step.dependencies,
            }
            for step in workflow_plan.steps
        ],
    }


def batch_validate(
    adata: AnnData,
    function_names: list,
    registry: Optional[Any] = None,
) -> Dict[str, ValidationResult]:
    """Validate prerequisites for multiple functions at once.

    Args:
        adata: AnnData object to validate.
        function_names: List of function names to validate.
        registry: Function registry (auto-loaded if None).

    Returns:
        Dict mapping function names to ValidationResult objects.

    Example:
        >>> results = batch_validate(adata, ['leiden', 'umap', 'rank_genes_groups'])
        >>> for func, result in results.items():
        ...     print(f"{func}: {'✓' if result.is_valid else '✗'}")
    """
    inspector = create_inspector(adata, registry, cache=True)
    results = {}

    for func_name in function_names:
        try:
            results[func_name] = inspector.validate_prerequisites(func_name)
        except Exception as e:
            # Create an error result
            results[func_name] = ValidationResult(
                function_name=func_name,
                is_valid=False,
                message=f"Validation error: {str(e)}",
            )

    return results


def get_validation_report(
    adata: AnnData,
    function_names: Optional[list] = None,
    registry: Optional[Any] = None,
    format: str = "summary",
) -> str:
    """Generate a comprehensive validation report.

    Args:
        adata: AnnData object to validate.
        function_names: List of functions to validate. If None, validates common functions.
        registry: Function registry (auto-loaded if None).
        format: Report format ('summary', 'detailed', 'markdown').

    Returns:
        Formatted validation report string.

    Example:
        >>> report = get_validation_report(adata, format='markdown')
        >>> print(report)
    """
    # Default to common analysis functions
    if function_names is None:
        function_names = [
            'qc', 'preprocess', 'scale', 'pca',
            'neighbors', 'umap', 'leiden', 'rank_genes_groups'
        ]

    # Batch validate
    results = batch_validate(adata, function_names, registry)

    # Format report
    if format == "summary":
        lines = ["=== Validation Report ===\n"]
        valid_count = sum(1 for r in results.values() if r.is_valid)
        lines.append(f"Valid: {valid_count}/{len(results)}\n")
        for func, result in results.items():
            status = "✓" if result.is_valid else "✗"
            lines.append(f"  {status} {func}")
        return "\n".join(lines)

    elif format == "detailed":
        lines = ["=== Detailed Validation Report ===\n"]
        for func, result in results.items():
            status = "VALID" if result.is_valid else "INVALID"
            lines.append(f"\n{func}: {status}")
            if not result.is_valid:
                if result.missing_prerequisites:
                    lines.append(f"  Missing prerequisites: {', '.join(result.missing_prerequisites)}")
                if result.missing_data_structures:
                    lines.append(f"  Missing data: {result.missing_data_structures}")
                if result.suggestions:
                    lines.append(f"  Suggestions: {len(result.suggestions)} available")
        return "\n".join(lines)

    elif format == "markdown":
        lines = ["# Validation Report\n"]
        valid_count = sum(1 for r in results.values() if r.is_valid)
        lines.append(f"**Status**: {valid_count}/{len(results)} functions valid\n")
        lines.append("## Results\n")
        for func, result in results.items():
            emoji = "✅" if result.is_valid else "❌"
            lines.append(f"### {emoji} {func}")
            if not result.is_valid:
                if result.missing_prerequisites:
                    lines.append(f"\n**Missing prerequisites**: {', '.join(result.missing_prerequisites)}")
                if result.missing_data_structures:
                    lines.append(f"\n**Missing data**: `{result.missing_data_structures}`")
            lines.append("")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {format}")


# Context manager for validation
class ValidationContext:
    """Context manager for validating prerequisites before code execution.

    Example:
        >>> with ValidationContext(adata, 'leiden') as ctx:
        ...     if ctx.is_valid:
        ...         # Run leiden clustering
        ...         ov.pp.leiden(adata)
        ...     else:
        ...         print(ctx.result.message)
    """

    def __init__(
        self,
        adata: AnnData,
        function_name: str,
        registry: Optional[Any] = None,
        raise_on_invalid: bool = False,
    ):
        """Initialize validation context.

        Args:
            adata: AnnData object to validate.
            function_name: Function to validate.
            registry: Function registry (auto-loaded if None).
            raise_on_invalid: If True, raises ValueError on validation failure.
        """
        self.adata = adata
        self.function_name = function_name
        self.registry = registry
        self.raise_on_invalid = raise_on_invalid
        self.result: Optional[ValidationResult] = None
        self.is_valid: bool = False

    def __enter__(self):
        """Enter context and validate prerequisites."""
        self.result = validate_function(
            self.adata,
            self.function_name,
            registry=self.registry,
            raise_on_invalid=self.raise_on_invalid,
        )
        self.is_valid = self.result.is_valid
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # No cleanup needed
        return False
