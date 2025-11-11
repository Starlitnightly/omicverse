"""
DataStateValidator - Pre-execution validation with auto-correction.

This module provides validation of generated code before execution, with
automatic correction for simple prerequisite issues.

Main Components:
    DataStateValidator: Pre-execution validator with auto-correction
    ValidationFeedback: Structured feedback for validation failures
    ComplexityLevel: Workflow complexity classification

Usage:
    >>> from omicverse.utils.inspector import DataStateValidator, create_inspector
    >>>
    >>> inspector = create_inspector(adata)
    >>> validator = DataStateValidator(adata, inspector.registry)
    >>>
    >>> # Validate code before execution
    >>> code = "ov.pp.leiden(adata, resolution=1.0)"
    >>> result = validator.validate_before_execution(code)
    >>>
    >>> if not result.is_valid:
    ...     # Try auto-correction
    ...     corrected = validator.auto_correct(code, result)
    ...     if corrected != code:
    ...         print(f"Auto-corrected: {corrected}")
    ...     else:
    ...         # Get feedback for manual correction
    ...         feedback = validator.get_validation_feedback(code, result)
    ...         print(feedback)
"""

import re
import ast
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from anndata import AnnData

from .inspector import DataStateInspector
from .data_structures import ValidationResult, ComplexityLevel


@dataclass
class ValidationFeedback:
    """Structured feedback for validation failures."""
    is_valid: bool
    generated_code: str
    issues: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    complexity: Optional[ComplexityLevel] = None
    can_auto_correct: bool = False
    explanation: Optional[str] = None

    def format_message(self) -> str:
        """Format feedback as user-friendly message."""
        if self.is_valid:
            return "✅ Code validation passed"

        lines = ["⚠️ Code Validation Failed", ""]
        lines.append("Generated code:")
        lines.append(f"  {self.generated_code}")
        lines.append("")

        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  ✗ {issue}")
            lines.append("")

        if self.can_auto_correct:
            lines.append("✅ This can be auto-corrected")
        elif self.suggested_fix:
            lines.append("Suggested Fix:")
            for line in self.suggested_fix.split('\n'):
                if line.strip():
                    lines.append(f"  {line}")
            lines.append("")

        if self.explanation:
            lines.append("Explanation:")
            lines.append(f"  {self.explanation}")

        return '\n'.join(lines)


class DataStateValidator:
    """
    Pre-execution validator with auto-correction.

    Validates generated code before execution and auto-corrects simple
    prerequisite issues.
    """

    # Simple prerequisites that can be auto-inserted
    SIMPLE_PREREQUISITES = {
        'scale': {
            'code_template': 'ov.pp.scale(adata)',
            'time_cost': 5,  # seconds
            'requires': [],
        },
        'pca': {
            'code_template': 'ov.pp.pca(adata, n_pcs=50)',
            'time_cost': 10,
            'requires': ['scale'],
        },
        'neighbors': {
            'code_template': 'ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
            'time_cost': 15,
            'requires': ['pca'],
        },
        'umap': {
            'code_template': 'ov.pp.umap(adata)',
            'time_cost': 20,
            'requires': ['neighbors'],
        },
    }

    # Complex prerequisites that should NOT be auto-inserted
    COMPLEX_PREREQUISITES = {
        'qc', 'preprocess', 'batch_correct', 'highly_variable_genes'
    }

    def __init__(self, adata: AnnData, registry: Any):
        """
        Initialize validator.

        Parameters
        ----------
        adata : AnnData
            AnnData object to validate against
        registry : PrerequisiteRegistry
            Registry with function metadata
        """
        self.adata = adata
        self.registry = registry
        self.inspector = DataStateInspector(adata, registry)

    def validate_before_execution(self, code: str) -> ValidationResult:
        """
        Validate code before execution.

        Parameters
        ----------
        code : str
            Generated code to validate

        Returns
        -------
        ValidationResult
            Aggregated validation result
        """
        # Extract function calls from code
        functions = self._extract_function_calls(code)

        if not functions:
            # No functions to validate - consider valid
            return ValidationResult(
                function_name="",
                is_valid=True,
                message="No OmicVerse functions to validate",
                missing_prerequisites=[],
                missing_data_structures={},
                executed_functions=[],
                suggestions=[]
            )

        # Validate each function
        results = []
        for func in functions:
            result = self.inspector.validate_prerequisites(func)
            results.append((func, result))

        # Aggregate results
        return self._aggregate_results(results)

    def auto_correct(self, code: str, validation_result: ValidationResult) -> str:
        """
        Auto-correct code by inserting prerequisites.

        Only corrects if ALL missing prerequisites are simple (auto-insertable).

        Parameters
        ----------
        code : str
            Original code
        validation_result : ValidationResult
            Validation result indicating missing prerequisites

        Returns
        -------
        str
            Corrected code with prerequisites inserted, or original code if
            correction not possible
        """
        if validation_result.is_valid:
            return code  # No correction needed

        missing = validation_result.missing_prerequisites

        # Check if all missing are simple (auto-insertable)
        if not self._all_simple_prerequisites(missing):
            return code  # Can't auto-correct complex prerequisites

        # Generate corrected code with prerequisites
        try:
            corrected = self._insert_prerequisites(code, missing)
            return corrected
        except Exception:
            # If insertion fails, return original code
            return code

    def get_validation_feedback(
        self,
        code: str,
        validation_result: ValidationResult
    ) -> ValidationFeedback:
        """
        Get structured validation feedback.

        Parameters
        ----------
        code : str
            Generated code
        validation_result : ValidationResult
            Validation result

        Returns
        -------
        ValidationFeedback
            Structured feedback with issues and suggestions
        """
        if validation_result.is_valid:
            return ValidationFeedback(
                is_valid=True,
                generated_code=code
            )

        # Extract issues
        issues = []
        for prereq in validation_result.missing_prerequisites:
            issues.append(f"{prereq} is required but not executed")

        for data_type, keys in validation_result.missing_data_structures.items():
            for key in keys:
                issues.append(f"Missing data: adata.{data_type}['{key}']")

        # Analyze complexity
        complexity = self._analyze_complexity(validation_result.missing_prerequisites)

        # Check if can auto-correct
        can_auto_correct = self._all_simple_prerequisites(
            validation_result.missing_prerequisites
        )

        # Generate suggested fix
        suggested_fix = None
        explanation = None

        if can_auto_correct:
            # Simple case - show auto-corrected code
            suggested_fix = self.auto_correct(code, validation_result)
            explanation = "These prerequisites can be automatically inserted."
        else:
            # Complex case - use suggestions from validation result
            if validation_result.suggestions:
                # Use first suggestion (highest priority)
                suggestion = validation_result.suggestions[0]
                suggested_fix = suggestion.code
                explanation = suggestion.explanation

        return ValidationFeedback(
            is_valid=False,
            generated_code=code,
            issues=issues,
            suggested_fix=suggested_fix,
            complexity=complexity,
            can_auto_correct=can_auto_correct,
            explanation=explanation
        )

    def _extract_function_calls(self, code: str) -> List[str]:
        """
        Extract OmicVerse function calls from code.

        Looks for patterns like:
        - ov.pp.function_name(...)
        - ov.single.function_name(...)
        - omicverse.pp.function_name(...)

        Parameters
        ----------
        code : str
            Python code

        Returns
        -------
        List[str]
            List of function names
        """
        functions = []

        # Pattern: ov.pp.function_name or ov.single.function_name
        pattern = r'ov\.(pp|single|pl|bulk|space)\.(\w+)\('
        matches = re.finditer(pattern, code)
        for match in matches:
            func_name = match.group(2)
            functions.append(func_name)

        # Pattern: omicverse.pp.function_name
        pattern2 = r'omicverse\.(pp|single|pl|bulk|space)\.(\w+)\('
        matches2 = re.finditer(pattern2, code)
        for match in matches2:
            func_name = match.group(2)
            if func_name not in functions:
                functions.append(func_name)

        return functions

    def _all_simple_prerequisites(self, missing: List[str]) -> bool:
        """
        Check if all missing prerequisites are simple (auto-insertable).

        Parameters
        ----------
        missing : List[str]
            List of missing prerequisite function names

        Returns
        -------
        bool
            True if all missing are simple, False otherwise
        """
        if not missing:
            return True

        for func in missing:
            # Check if it's a complex prerequisite
            if func in self.COMPLEX_PREREQUISITES:
                return False

            # Check if it's NOT in simple prerequisites
            if func not in self.SIMPLE_PREREQUISITES:
                # Unknown function - consider complex to be safe
                return False

        return True

    def _insert_prerequisites(self, code: str, missing: List[str]) -> str:
        """
        Insert prerequisites into code in correct order.

        Parameters
        ----------
        code : str
            Original code
        missing : List[str]
            Missing prerequisites to insert

        Returns
        -------
        str
            Code with prerequisites inserted before original code
        """
        # Resolve dependency order
        ordered_prereqs = self._resolve_insertion_order(missing)

        # Generate prerequisite code
        prereq_lines = []
        for func in ordered_prereqs:
            if func in self.SIMPLE_PREREQUISITES:
                template = self.SIMPLE_PREREQUISITES[func]['code_template']
                prereq_lines.append(template)

        # Combine prerequisite code with original code
        if prereq_lines:
            prereq_code = '\n'.join(prereq_lines)
            return f"{prereq_code}\n{code}"
        else:
            return code

    def _resolve_insertion_order(self, missing: List[str]) -> List[str]:
        """
        Resolve order for inserting prerequisites (topological sort).

        Parameters
        ----------
        missing : List[str]
            Missing prerequisites

        Returns
        -------
        List[str]
            Prerequisites in correct execution order
        """
        # Build dependency graph
        graph = {}
        in_degree = {}

        for func in missing:
            if func in self.SIMPLE_PREREQUISITES:
                requires = self.SIMPLE_PREREQUISITES[func]['requires']
                graph[func] = [r for r in requires if r in missing]
                in_degree[func] = 0

        # Calculate in-degrees
        for func in graph:
            for dep in graph[func]:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Topological sort (Kahn's algorithm)
        queue = [func for func, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            func = queue.pop(0)
            result.append(func)

            for neighbor in graph.get(func, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Reverse to get correct order (dependencies first)
        return list(reversed(result))

    def _aggregate_results(
        self,
        results: List[Tuple[str, ValidationResult]]
    ) -> ValidationResult:
        """
        Aggregate multiple validation results.

        Parameters
        ----------
        results : List[Tuple[str, ValidationResult]]
            List of (function_name, validation_result) tuples

        Returns
        -------
        ValidationResult
            Aggregated result
        """
        all_valid = all(result.is_valid for _, result in results)

        # Get all function names
        func_names = [name for name, _ in results]

        if all_valid:
            return ValidationResult(
                function_name=", ".join(func_names),
                is_valid=True,
                message="All functions validated successfully",
                missing_prerequisites=[],
                missing_data_structures={},
                executed_functions=[],
                suggestions=[]
            )

        # Aggregate missing prerequisites and data
        all_missing_prereqs = []
        all_missing_data = {}
        all_executed = []
        all_suggestions = []

        for func_name, result in results:
            if not result.is_valid:
                # Collect missing prerequisites
                for prereq in result.missing_prerequisites:
                    if prereq not in all_missing_prereqs:
                        all_missing_prereqs.append(prereq)

                # Collect missing data
                for data_type, keys in result.missing_data_structures.items():
                    if data_type not in all_missing_data:
                        all_missing_data[data_type] = []
                    for key in keys:
                        if key not in all_missing_data[data_type]:
                            all_missing_data[data_type].append(key)

                # Collect suggestions
                all_suggestions.extend(result.suggestions)

            # Collect executed functions (it's a list)
            for func in result.executed_functions:
                if func not in all_executed:
                    all_executed.append(func)

        # Create aggregated message
        message = f"Validation failed for: {', '.join(func_names)}"

        return ValidationResult(
            function_name=", ".join(func_names),
            is_valid=False,
            message=message,
            missing_prerequisites=all_missing_prereqs,
            missing_data_structures=all_missing_data,
            executed_functions=all_executed,
            suggestions=all_suggestions[:5]  # Limit to top 5 suggestions
        )

    def _analyze_complexity(self, missing_prerequisites: List[str]) -> ComplexityLevel:
        """
        Analyze workflow complexity.

        Parameters
        ----------
        missing_prerequisites : List[str]
            Missing prerequisites

        Returns
        -------
        ComplexityLevel
            Complexity classification
        """
        num_missing = len(missing_prerequisites)

        # Check for complex prerequisites
        has_complex = any(
            prereq in self.COMPLEX_PREREQUISITES
            for prereq in missing_prerequisites
        )

        if num_missing >= 4 or has_complex:
            return ComplexityLevel.HIGH
        elif num_missing >= 2:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW


# Convenience function for quick validation
def validate_code(
    adata: AnnData,
    code: str,
    registry: Optional[Any] = None,
    auto_correct: bool = False
) -> Tuple[bool, str, Optional[ValidationFeedback]]:
    """
    Quick validation of code.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    code : str
        Code to validate
    registry : Optional[Any]
        Registry (auto-loaded if None)
    auto_correct : bool
        If True, attempt auto-correction

    Returns
    -------
    Tuple[bool, str, Optional[ValidationFeedback]]
        (is_valid, code_or_corrected, feedback)
    """
    if registry is None:
        from .prerequisite_registry import PrerequisiteRegistry
        registry = PrerequisiteRegistry()

    validator = DataStateValidator(adata, registry)
    result = validator.validate_before_execution(code)

    if result.is_valid:
        return True, code, None

    # Get feedback
    feedback = validator.get_validation_feedback(code, result)

    # Try auto-correction if requested
    if auto_correct:
        corrected = validator.auto_correct(code, result)
        if corrected != code:
            return False, corrected, feedback

    return False, code, feedback
