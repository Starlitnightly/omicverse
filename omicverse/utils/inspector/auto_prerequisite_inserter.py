"""
AutoPrerequisiteInserter - Automatic prerequisite insertion into generated code.

This module provides automatic insertion of simple prerequisites into generated
code, creating complete executable workflows with proper dependency ordering.

Main Components:
    AutoPrerequisiteInserter: Automatic prerequisite inserter
    InsertionResult: Result of insertion analysis and code generation

Usage:
    >>> from omicverse.utils.inspector import AutoPrerequisiteInserter
    >>>
    >>> inserter = AutoPrerequisiteInserter(registry)
    >>> result = inserter.insert_prerequisites(
    ...     code="ov.pp.leiden(adata, resolution=1.0)",
    ...     missing_prerequisites=['pca', 'neighbors']
    ... )
    >>>
    >>> if result.inserted:
    ...     print(result.modified_code)
    ...     # Output:
    ...     # # Auto-inserted prerequisites
    ...     # ov.pp.pca(adata, n_pcs=50)
    ...     # ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    ...     #
    ...     # # Original code
    ...     # ov.pp.leiden(adata, resolution=1.0)
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .data_structures import ComplexityLevel


class InsertionPolicy(Enum):
    """Policy for prerequisite insertion."""
    AUTO_INSERT = "auto_insert"  # Simple prerequisites, safe to auto-insert
    ESCALATE = "escalate"  # Complex prerequisites, escalate to high-level function
    MANUAL = "manual"  # User configuration needed, provide guidance


@dataclass
class InsertionResult:
    """Result of prerequisite insertion analysis."""
    inserted: bool
    original_code: str
    modified_code: str

    # Insertion details
    inserted_prerequisites: List[str] = field(default_factory=list)
    insertion_policy: Optional[InsertionPolicy] = None

    # Metadata
    estimated_time_seconds: int = 0
    explanation: str = ""

    # Alternative suggestions (if not inserted)
    alternative_suggestion: Optional[str] = None


class AutoPrerequisiteInserter:
    """
    Automatic prerequisite inserter.

    Automatically inserts simple prerequisites into generated code to create
    complete executable workflows.
    """

    # Simple prerequisites that can be auto-inserted
    SIMPLE_PREREQUISITES = {
        'scale': {
            'code_template': 'ov.pp.scale(adata)',
            'time_cost': 5,  # seconds
            'requires': [],
            'description': 'Scale data to unit variance',
        },
        'pca': {
            'code_template': 'ov.pp.pca(adata, n_pcs=50)',
            'time_cost': 10,
            'requires': ['scale'],
            'description': 'Principal component analysis',
        },
        'neighbors': {
            'code_template': 'ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
            'time_cost': 15,
            'requires': ['pca'],
            'description': 'Compute neighborhood graph',
        },
        'umap': {
            'code_template': 'ov.pp.umap(adata)',
            'time_cost': 20,
            'requires': ['neighbors'],
            'description': 'UMAP dimensionality reduction',
        },
        'tsne': {
            'code_template': 'ov.pp.tsne(adata)',
            'time_cost': 30,
            'requires': ['pca'],
            'description': 't-SNE dimensionality reduction',
        },
    }

    # Complex prerequisites that should NOT be auto-inserted
    COMPLEX_PREREQUISITES = {
        'qc': 'Quality control requires parameter tuning',
        'preprocess': 'Preprocessing has multiple configuration options',
        'batch_correct': 'Batch correction requires batch key information',
        'highly_variable_genes': 'HVG selection requires parameter choices',
        'normalize': 'Normalization method needs to be specified',
    }

    def __init__(self, registry: Any):
        """
        Initialize inserter.

        Parameters
        ----------
        registry : PrerequisiteRegistry
            Registry with function metadata
        """
        self.registry = registry

    def insert_prerequisites(
        self,
        code: str,
        missing_prerequisites: List[str],
        complexity: Optional[ComplexityLevel] = None
    ) -> InsertionResult:
        """
        Insert missing prerequisites into code.

        Parameters
        ----------
        code : str
            Original generated code
        missing_prerequisites : List[str]
            Missing prerequisite functions
        complexity : Optional[ComplexityLevel]
            Pre-analyzed complexity (optional)

        Returns
        -------
        InsertionResult
            Result with modified code or explanation
        """
        if not missing_prerequisites:
            return InsertionResult(
                inserted=False,
                original_code=code,
                modified_code=code,
                insertion_policy=InsertionPolicy.AUTO_INSERT,
                explanation="No missing prerequisites"
            )

        # Determine insertion policy
        policy = self._determine_policy(missing_prerequisites, complexity)

        if policy == InsertionPolicy.AUTO_INSERT:
            # Auto-insert simple prerequisites
            return self._auto_insert(code, missing_prerequisites)

        elif policy == InsertionPolicy.ESCALATE:
            # Complex workflow, suggest escalation
            return self._suggest_escalation(code, missing_prerequisites)

        else:  # MANUAL
            # Provide manual guidance
            return self._suggest_manual(code, missing_prerequisites)

    def can_auto_insert(self, missing_prerequisites: List[str]) -> bool:
        """
        Check if all missing prerequisites can be auto-inserted.

        Parameters
        ----------
        missing_prerequisites : List[str]
            Missing prerequisite functions

        Returns
        -------
        bool
            True if all can be auto-inserted
        """
        if not missing_prerequisites:
            return True

        for prereq in missing_prerequisites:
            if prereq in self.COMPLEX_PREREQUISITES:
                return False
            if prereq not in self.SIMPLE_PREREQUISITES:
                return False

        return True

    def _determine_policy(
        self,
        missing_prerequisites: List[str],
        complexity: Optional[ComplexityLevel]
    ) -> InsertionPolicy:
        """
        Determine insertion policy based on missing prerequisites.

        Parameters
        ----------
        missing_prerequisites : List[str]
            Missing prerequisites
        complexity : Optional[ComplexityLevel]
            Pre-analyzed complexity

        Returns
        -------
        InsertionPolicy
            Policy to use
        """
        # Check for complex prerequisites
        has_complex = any(
            prereq in self.COMPLEX_PREREQUISITES
            for prereq in missing_prerequisites
        )

        if has_complex:
            # Has complex prerequisites - need manual config or escalation
            if complexity == ComplexityLevel.HIGH or len(missing_prerequisites) >= 4:
                return InsertionPolicy.ESCALATE
            else:
                return InsertionPolicy.MANUAL

        # Check if all are simple
        all_simple = all(
            prereq in self.SIMPLE_PREREQUISITES
            for prereq in missing_prerequisites
        )

        if not all_simple:
            # Has unknown prerequisites
            return InsertionPolicy.MANUAL

        # All simple - can auto-insert
        return InsertionPolicy.AUTO_INSERT

    def _auto_insert(
        self,
        code: str,
        missing_prerequisites: List[str]
    ) -> InsertionResult:
        """
        Auto-insert simple prerequisites.

        Parameters
        ----------
        code : str
            Original code
        missing_prerequisites : List[str]
            Missing prerequisites (all must be simple)

        Returns
        -------
        InsertionResult
            Result with modified code
        """
        # Resolve dependency order
        ordered_prereqs = self._resolve_dependencies(missing_prerequisites)

        # Calculate total time
        total_time = sum(
            self.SIMPLE_PREREQUISITES[p]['time_cost']
            for p in ordered_prereqs
        )

        # Generate prerequisite code
        prereq_lines = []
        for prereq in ordered_prereqs:
            meta = self.SIMPLE_PREREQUISITES[prereq]
            prereq_lines.append(meta['code_template'])

        # Build modified code with comments
        modified_lines = []

        if prereq_lines:
            modified_lines.append("# Auto-inserted prerequisites")
            modified_lines.extend(prereq_lines)
            modified_lines.append("")
            modified_lines.append("# Original code")

        modified_lines.append(code.strip())

        modified_code = '\n'.join(modified_lines)

        # Build explanation
        prereq_names = ', '.join(ordered_prereqs)
        explanation = (
            f"Auto-inserted {len(ordered_prereqs)} prerequisite(s): {prereq_names}. "
            f"Estimated time: {total_time} seconds."
        )

        return InsertionResult(
            inserted=True,
            original_code=code,
            modified_code=modified_code,
            inserted_prerequisites=ordered_prereqs,
            insertion_policy=InsertionPolicy.AUTO_INSERT,
            estimated_time_seconds=total_time,
            explanation=explanation
        )

    def _suggest_escalation(
        self,
        code: str,
        missing_prerequisites: List[str]
    ) -> InsertionResult:
        """
        Suggest escalation to high-level function.

        Parameters
        ----------
        code : str
            Original code
        missing_prerequisites : List[str]
            Missing prerequisites

        Returns
        -------
        InsertionResult
            Result with escalation suggestion
        """
        # Build suggestion
        complex_items = [
            prereq for prereq in missing_prerequisites
            if prereq in self.COMPLEX_PREREQUISITES
        ]

        suggestion = (
            f"This workflow requires {len(missing_prerequisites)} prerequisite(s), "
            f"including complex steps: {', '.join(complex_items)}.\n\n"
            f"Consider using a high-level preprocessing function:\n"
            f"ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)\n\n"
            f"This handles qc, normalization, feature selection, scaling, and PCA in one step."
        )

        explanation = (
            f"Complex prerequisites detected ({', '.join(complex_items)}). "
            f"Escalation to high-level function recommended."
        )

        return InsertionResult(
            inserted=False,
            original_code=code,
            modified_code=code,
            insertion_policy=InsertionPolicy.ESCALATE,
            explanation=explanation,
            alternative_suggestion=suggestion
        )

    def _suggest_manual(
        self,
        code: str,
        missing_prerequisites: List[str]
    ) -> InsertionResult:
        """
        Suggest manual prerequisite handling.

        Parameters
        ----------
        code : str
            Original code
        missing_prerequisites : List[str]
            Missing prerequisites

        Returns
        -------
        InsertionResult
            Result with manual suggestion
        """
        # Separate simple and complex
        simple = [p for p in missing_prerequisites if p in self.SIMPLE_PREREQUISITES]
        complex_items = [p for p in missing_prerequisites if p in self.COMPLEX_PREREQUISITES]
        unknown = [
            p for p in missing_prerequisites
            if p not in self.SIMPLE_PREREQUISITES and p not in self.COMPLEX_PREREQUISITES
        ]

        # Build suggestion
        suggestion_lines = ["Missing prerequisites require configuration:"]
        suggestion_lines.append("")

        for prereq in complex_items:
            reason = self.COMPLEX_PREREQUISITES[prereq]
            suggestion_lines.append(f"- {prereq}: {reason}")

        if unknown:
            suggestion_lines.append("")
            suggestion_lines.append(f"Unknown prerequisites: {', '.join(unknown)}")

        if simple:
            suggestion_lines.append("")
            suggestion_lines.append("Simple prerequisites that can be added:")
            for prereq in simple:
                meta = self.SIMPLE_PREREQUISITES[prereq]
                suggestion_lines.append(f"- {meta['code_template']}")

        suggestion = '\n'.join(suggestion_lines)

        explanation = (
            f"Manual configuration needed for: {', '.join(complex_items or unknown)}."
        )

        return InsertionResult(
            inserted=False,
            original_code=code,
            modified_code=code,
            insertion_policy=InsertionPolicy.MANUAL,
            explanation=explanation,
            alternative_suggestion=suggestion
        )

    def _resolve_dependencies(self, prerequisites: List[str]) -> List[str]:
        """
        Resolve dependency order (topological sort).

        Parameters
        ----------
        prerequisites : List[str]
            Prerequisites to order

        Returns
        -------
        List[str]
            Prerequisites in correct execution order
        """
        if not prerequisites:
            return []

        # Build dependency graph
        graph = {}
        in_degree = {}

        for prereq in prerequisites:
            if prereq in self.SIMPLE_PREREQUISITES:
                requires = self.SIMPLE_PREREQUISITES[prereq]['requires']
                graph[prereq] = [r for r in requires if r in prerequisites]
            else:
                graph[prereq] = []
            in_degree[prereq] = 0

        # Calculate in-degrees
        for prereq in graph:
            for dep in graph[prereq]:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Kahn's algorithm
        queue = [prereq for prereq, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            prereq = queue.pop(0)
            result.append(prereq)

            for neighbor in graph.get(prereq, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Reverse to get correct order (dependencies first)
        return list(reversed(result))


# Convenience function
def auto_insert_prerequisites(
    registry: Any,
    code: str,
    missing_prerequisites: List[str],
    complexity: Optional[ComplexityLevel] = None
) -> InsertionResult:
    """
    Convenience function for automatic prerequisite insertion.

    Parameters
    ----------
    registry : PrerequisiteRegistry
        Registry with function metadata
    code : str
        Original generated code
    missing_prerequisites : List[str]
        Missing prerequisite functions
    complexity : Optional[ComplexityLevel]
        Pre-analyzed complexity

    Returns
    -------
    InsertionResult
        Result with modified code or suggestions
    """
    inserter = AutoPrerequisiteInserter(registry)
    return inserter.insert_prerequisites(code, missing_prerequisites, complexity)
