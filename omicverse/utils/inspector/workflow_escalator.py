"""
WorkflowEscalator - Intelligent workflow complexity analysis and escalation.

This module provides complexity analysis for prerequisite workflows and
intelligently escalates complex workflows to high-level functions instead
of suggesting many small steps.

Main Components:
    WorkflowEscalator: Complexity analyzer and escalation engine
    EscalationResult: Result of escalation analysis
    EscalationStrategy: Strategy for handling complex workflows

Usage:
    >>> from omicverse.utils.inspector import WorkflowEscalator
    >>>
    >>> escalator = WorkflowEscalator(registry)
    >>> result = escalator.should_escalate(
    ...     target_function='leiden',
    ...     missing_prerequisites=['qc', 'preprocess', 'scale', 'pca', 'neighbors'],
    ...     missing_data={'obsm': ['X_pca'], 'obsp': ['connectivities']}
    ... )
    >>>
    >>> if result.should_escalate:
    ...     print(result.escalated_suggestion.code)
    ...     # Output: ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .data_structures import Suggestion, ComplexityLevel


class EscalationStrategy(Enum):
    """Strategy for handling complex workflows."""
    NO_ESCALATION = "no_escalation"  # Simple case, no escalation needed
    WORKFLOW_CHAIN = "workflow_chain"  # Medium complexity, generate ordered chain
    HIGH_LEVEL_FUNCTION = "high_level_function"  # High complexity, use preprocess/batch_correct


@dataclass
class EscalationResult:
    """Result of escalation analysis."""
    should_escalate: bool
    complexity: ComplexityLevel
    strategy: EscalationStrategy

    # Escalation suggestion (if should_escalate=True)
    escalated_suggestion: Optional[Suggestion] = None

    # Original workflow (if no escalation)
    original_workflow: List[str] = field(default_factory=list)

    # Analysis details
    dependency_depth: int = 0
    num_missing: int = 0
    has_complex_prerequisites: bool = False

    # Explanation
    explanation: str = ""


class WorkflowEscalator:
    """
    Intelligent workflow escalator.

    Analyzes workflow complexity and escalates complex workflows to high-level
    functions instead of suggesting many small steps.
    """

    # High-level functions that can replace multiple steps
    HIGH_LEVEL_FUNCTIONS = {
        'preprocess': {
            'replaces': ['qc', 'normalize', 'highly_variable_genes', 'scale', 'pca'],
            'code_template': 'ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000, n_pcs=50)',
            'description': 'Complete preprocessing pipeline',
            'estimated_time': '1-2 minutes',
            'estimated_time_seconds': 90,
        },
        'batch_correct': {
            'replaces': ['combat', 'harmony', 'scanorama'],
            'code_template': 'ov.pp.batch_correct(adata, batch_key="batch", method="combat")',
            'description': 'Batch correction workflow',
            'estimated_time': '30-60 seconds',
            'estimated_time_seconds': 45,
        },
        'bulk_rnaseq_pipeline': {
            'replaces': ['prefetch', 'fqdump', 'fastp', 'STAR', 'featureCount'],
            'code_template': (
                "# Bulk RNA-seq pipeline: download -> QC -> align -> quantify\n"
                "fq = ov.alignment.fqdump(sra_ids, output_dir='fastq')\n"
                "clean = ov.alignment.fastp(samples, output_dir='fastp')\n"
                "bams = ov.alignment.STAR(samples, genome_dir=genome_dir, output_dir='star')\n"
                "counts = ov.alignment.featureCount(bam_items, gtf=gtf, output_dir='counts')"
            ),
            'description': 'Complete bulk RNA-seq alignment pipeline',
            'estimated_time': '10-30 minutes',
            'estimated_time_seconds': 1200,
        },
    }

    # Complex prerequisites that trigger escalation
    COMPLEX_TRIGGERS = {
        'qc', 'preprocess', 'batch_correct', 'highly_variable_genes',
        'normalize', 'combat', 'harmony', 'scanorama',
        # alignment / FASTQ analysis triggers
        'prefetch', 'fqdump', 'fastp', 'STAR', 'featureCount',
        'align', 'alignment', 'fastq', 'mapping',
    }

    def __init__(self, registry: Any):
        """
        Initialize escalator.

        Parameters
        ----------
        registry : PrerequisiteRegistry
            Registry with function metadata
        """
        self.registry = registry

    def should_escalate(
        self,
        target_function: str,
        missing_prerequisites: List[str],
        missing_data: Optional[Dict[str, List[str]]] = None
    ) -> EscalationResult:
        """
        Determine if workflow should be escalated.

        Parameters
        ----------
        target_function : str
            Target function to execute
        missing_prerequisites : List[str]
            Missing prerequisite functions
        missing_data : Optional[Dict[str, List[str]]]
            Missing data structures

        Returns
        -------
        EscalationResult
            Result with escalation decision and suggestion
        """
        if missing_data is None:
            missing_data = {}

        # Analyze complexity
        complexity = self.analyze_complexity(missing_prerequisites)
        dependency_depth = self._calculate_dependency_depth(missing_prerequisites)
        num_missing = len(missing_prerequisites)
        has_complex = any(
            prereq in self.COMPLEX_TRIGGERS
            for prereq in missing_prerequisites
        )

        # Determine strategy
        if complexity == ComplexityLevel.HIGH:
            strategy = EscalationStrategy.HIGH_LEVEL_FUNCTION
        elif complexity == ComplexityLevel.MEDIUM:
            strategy = EscalationStrategy.WORKFLOW_CHAIN
        else:
            strategy = EscalationStrategy.NO_ESCALATION

        # Generate escalation suggestion
        should_escalate = strategy != EscalationStrategy.NO_ESCALATION
        escalated_suggestion = None
        explanation = ""

        if strategy == EscalationStrategy.HIGH_LEVEL_FUNCTION:
            escalated_suggestion = self._escalate_to_high_level(
                target_function,
                missing_prerequisites,
                missing_data
            )
            explanation = (
                f"Workflow is highly complex ({num_missing} missing prerequisites). "
                f"Escalating to high-level function for efficiency."
            )
        elif strategy == EscalationStrategy.WORKFLOW_CHAIN:
            escalated_suggestion = self._generate_workflow_chain(
                target_function,
                missing_prerequisites,
                missing_data
            )
            explanation = (
                f"Workflow has medium complexity ({num_missing} missing prerequisites). "
                f"Generating ordered workflow chain."
            )
        else:
            explanation = (
                f"Workflow is simple ({num_missing} missing prerequisites). "
                f"No escalation needed - prerequisites can be auto-inserted."
            )

        return EscalationResult(
            should_escalate=should_escalate,
            complexity=complexity,
            strategy=strategy,
            escalated_suggestion=escalated_suggestion,
            original_workflow=missing_prerequisites,
            dependency_depth=dependency_depth,
            num_missing=num_missing,
            has_complex_prerequisites=has_complex,
            explanation=explanation
        )

    def analyze_complexity(self, missing_prerequisites: List[str]) -> ComplexityLevel:
        """
        Analyze workflow complexity.

        Parameters
        ----------
        missing_prerequisites : List[str]
            Missing prerequisite functions

        Returns
        -------
        ComplexityLevel
            Complexity classification (LOW/MEDIUM/HIGH)
        """
        num_missing = len(missing_prerequisites)

        # Check for complex triggers
        has_complex = any(
            prereq in self.COMPLEX_TRIGGERS
            for prereq in missing_prerequisites
        )

        # Calculate dependency depth
        depth = self._calculate_dependency_depth(missing_prerequisites)

        # Classify complexity
        if num_missing >= 4 or has_complex:
            return ComplexityLevel.HIGH
        elif num_missing >= 2 or depth >= 3:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW

    def _calculate_dependency_depth(self, missing_prerequisites: List[str]) -> int:
        """
        Calculate maximum dependency depth.

        Parameters
        ----------
        missing_prerequisites : List[str]
            Missing prerequisite functions

        Returns
        -------
        int
            Maximum dependency chain depth
        """
        if not missing_prerequisites:
            return 0

        max_depth = 0

        for func in missing_prerequisites:
            depth = self._get_function_depth(func, visited=set())
            max_depth = max(max_depth, depth)

        return max_depth

    def _get_function_depth(self, func_name: str, visited: Set[str]) -> int:
        """
        Get dependency depth for a function (recursive).

        Parameters
        ----------
        func_name : str
            Function name
        visited : Set[str]
            Set of visited functions (to avoid cycles)

        Returns
        -------
        int
            Depth of dependency chain
        """
        if func_name in visited:
            return 0

        visited.add(func_name)

        func_meta = self.registry.get_function(func_name)
        if not func_meta:
            return 1

        # Get required prerequisites
        prereqs = func_meta.get('prerequisites', {}).get('required', [])
        if not prereqs:
            return 1

        # Calculate depth recursively
        max_child_depth = 0
        for prereq in prereqs:
            child_depth = self._get_function_depth(prereq, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    def _escalate_to_high_level(
        self,
        target_function: str,
        missing_prerequisites: List[str],
        missing_data: Dict[str, List[str]]
    ) -> Suggestion:
        """
        Escalate to high-level function.

        Parameters
        ----------
        target_function : str
            Target function
        missing_prerequisites : List[str]
            Missing prerequisites
        missing_data : Dict[str, List[str]]
            Missing data structures

        Returns
        -------
        Suggestion
            Suggestion with high-level function
        """
        # Check if preprocess can replace missing prerequisites
        if self._can_use_preprocess(missing_prerequisites):
            preprocess_meta = self.HIGH_LEVEL_FUNCTIONS['preprocess']

            # Build complete workflow
            code_lines = [preprocess_meta['code_template']]

            # Add any remaining prerequisites not covered by preprocess
            remaining = self._get_remaining_after_preprocess(missing_prerequisites)
            for prereq in remaining:
                code_lines.append(self._get_default_code(prereq))

            # Add target function
            code_lines.append(self._get_default_code(target_function))

            code = '\n'.join(code_lines)

            return Suggestion(
                priority='HIGH',
                suggestion_type='workflow_escalation',
                description=f'Complete preprocessing pipeline for {target_function}',
                code=code,
                explanation=(
                    f'{target_function} requires extensive preprocessing. '
                    f'Use ov.pp.preprocess() to handle qc, normalization, '
                    f'feature selection, scaling, and PCA in one step.'
                ),
                estimated_time=preprocess_meta['estimated_time'],
                estimated_time_seconds=preprocess_meta['estimated_time_seconds'],
                prerequisites=[],
                auto_executable=True
            )

        # Check if batch correction can be used
        elif self._can_use_batch_correct(missing_prerequisites):
            batch_meta = self.HIGH_LEVEL_FUNCTIONS['batch_correct']

            code_lines = [batch_meta['code_template']]
            code_lines.append(self._get_default_code(target_function))

            code = '\n'.join(code_lines)

            return Suggestion(
                priority='HIGH',
                suggestion_type='workflow_escalation',
                description=f'Batch correction workflow for {target_function}',
                code=code,
                explanation=(
                    f'{target_function} requires batch correction. '
                    f'Use ov.pp.batch_correct() to handle batch effects.'
                ),
                estimated_time=batch_meta['estimated_time'],
                estimated_time_seconds=batch_meta['estimated_time_seconds'],
                prerequisites=[],
                auto_executable=True
            )

        # Check if bulk RNA-seq pipeline can be used
        elif self._can_use_bulk_rnaseq_pipeline(missing_prerequisites):
            pipeline_meta = self.HIGH_LEVEL_FUNCTIONS['bulk_rnaseq_pipeline']

            code_lines = [pipeline_meta['code_template']]

            # Add any remaining prerequisites not covered by the pipeline
            pipeline_replaces = set(pipeline_meta['replaces'])
            remaining = [p for p in missing_prerequisites if p not in pipeline_replaces]
            for prereq in remaining:
                code_lines.append(self._get_default_code(prereq))

            code_lines.append(self._get_default_code(target_function))
            code = '\n'.join(code_lines)

            return Suggestion(
                priority='HIGH',
                suggestion_type='workflow_escalation',
                description=f'Bulk RNA-seq alignment pipeline for {target_function}',
                code=code,
                explanation=(
                    f'{target_function} requires a multi-step alignment workflow. '
                    f'Use the ov.alignment pipeline (fqdump -> fastp -> STAR -> featureCount) '
                    f'to process raw sequencing data end-to-end.'
                ),
                estimated_time=pipeline_meta['estimated_time'],
                estimated_time_seconds=pipeline_meta['estimated_time_seconds'],
                prerequisites=[],
                auto_executable=True
            )

        # Fallback: generate workflow chain
        else:
            return self._generate_workflow_chain(
                target_function,
                missing_prerequisites,
                missing_data
            )

    def _generate_workflow_chain(
        self,
        target_function: str,
        missing_prerequisites: List[str],
        missing_data: Dict[str, List[str]]
    ) -> Suggestion:
        """
        Generate ordered workflow chain for medium complexity.

        Parameters
        ----------
        target_function : str
            Target function
        missing_prerequisites : List[str]
            Missing prerequisites
        missing_data : Dict[str, List[str]]
            Missing data structures

        Returns
        -------
        Suggestion
            Suggestion with ordered workflow chain
        """
        # Order prerequisites by dependency
        ordered_prereqs = self._topological_sort(missing_prerequisites)

        # Generate code for each prerequisite
        code_lines = []
        total_time = 0

        for prereq in ordered_prereqs:
            code_lines.append(self._get_default_code(prereq))
            # Estimate 10 seconds per function
            total_time += 10

        # Add target function
        code_lines.append(self._get_default_code(target_function))
        total_time += 10

        code = '\n'.join(code_lines)

        return Suggestion(
            priority='MEDIUM',
            suggestion_type='workflow_guidance',
            description=f'Ordered workflow chain for {target_function}',
            code=code,
            explanation=(
                f'{target_function} requires {len(ordered_prereqs)} prerequisite(s). '
                f'Execute them in order: {" → ".join(ordered_prereqs)} → {target_function}.'
            ),
            estimated_time=f'{total_time} seconds',
            estimated_time_seconds=total_time,
            prerequisites=ordered_prereqs,
            auto_executable=True
        )

    def _can_use_preprocess(self, missing_prerequisites: List[str]) -> bool:
        """Check if preprocess() can replace missing prerequisites."""
        preprocess_replaces = set(self.HIGH_LEVEL_FUNCTIONS['preprocess']['replaces'])
        missing_set = set(missing_prerequisites)

        # Check if at least 2 prerequisites can be replaced by preprocess
        overlap = missing_set & preprocess_replaces
        return len(overlap) >= 2

    def _can_use_batch_correct(self, missing_prerequisites: List[str]) -> bool:
        """Check if batch_correct() can replace missing prerequisites."""
        batch_replaces = set(self.HIGH_LEVEL_FUNCTIONS['batch_correct']['replaces'])
        missing_set = set(missing_prerequisites)

        # Check if any batch correction method is missing
        overlap = missing_set & batch_replaces
        return len(overlap) >= 1

    def _can_use_bulk_rnaseq_pipeline(self, missing_prerequisites: List[str]) -> bool:
        """Check if the bulk RNA-seq pipeline can replace missing prerequisites."""
        pipeline_replaces = set(self.HIGH_LEVEL_FUNCTIONS['bulk_rnaseq_pipeline']['replaces'])
        missing_set = set(missing_prerequisites)

        # Trigger when at least 2 alignment steps are missing
        overlap = missing_set & pipeline_replaces
        return len(overlap) >= 2

    def _get_remaining_after_preprocess(self, missing_prerequisites: List[str]) -> List[str]:
        """Get prerequisites not covered by preprocess()."""
        preprocess_replaces = set(self.HIGH_LEVEL_FUNCTIONS['preprocess']['replaces'])
        remaining = [p for p in missing_prerequisites if p not in preprocess_replaces]
        return self._topological_sort(remaining)

    def _topological_sort(self, functions: List[str]) -> List[str]:
        """
        Sort functions in dependency order (topological sort).

        Parameters
        ----------
        functions : List[str]
            Function names to sort

        Returns
        -------
        List[str]
            Functions in execution order
        """
        if not functions:
            return []

        # Build dependency graph
        graph = {}
        in_degree = {}

        for func in functions:
            func_meta = self.registry.get_function(func)
            if func_meta:
                prereqs = func_meta.get('prerequisites', {}).get('required', [])
                graph[func] = [p for p in prereqs if p in functions]
            else:
                graph[func] = []
            in_degree[func] = 0

        # Calculate in-degrees
        for func in graph:
            for dep in graph[func]:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Kahn's algorithm
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

    def _get_default_code(self, function_name: str) -> str:
        """
        Get default code template for a function.

        Parameters
        ----------
        function_name : str
            Function name

        Returns
        -------
        str
            Default code string
        """
        # Common default parameters
        defaults = {
            'qc': 'ov.pp.qc(adata)',
            'normalize': 'ov.pp.normalize_total(adata)',
            'highly_variable_genes': 'ov.pp.highly_variable_genes(adata, n_top_genes=2000)',
            'scale': 'ov.pp.scale(adata)',
            'pca': 'ov.pp.pca(adata, n_pcs=50)',
            'neighbors': 'ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
            'umap': 'ov.pp.umap(adata)',
            'leiden': 'ov.pp.leiden(adata, resolution=1.0)',
            'louvain': 'ov.pp.louvain(adata, resolution=1.0)',
            'tsne': 'ov.pp.tsne(adata)',
            # alignment / FASTQ analysis defaults
            'prefetch': "ov.alignment.prefetch(sra_ids, output_dir='prefetch')",
            'fqdump': "ov.alignment.fqdump(sra_ids, output_dir='fastq')",
            'fastp': "ov.alignment.fastp(samples, output_dir='fastp')",
            'STAR': "ov.alignment.STAR(samples, genome_dir='index', output_dir='star')",
            'featureCount': "ov.alignment.featureCount(bam_items, gtf='genes.gtf', output_dir='counts')",
        }

        return defaults.get(function_name, f'ov.pp.{function_name}(adata)')


# Convenience function
def analyze_and_escalate(
    registry: Any,
    target_function: str,
    missing_prerequisites: List[str],
    missing_data: Optional[Dict[str, List[str]]] = None
) -> EscalationResult:
    """
    Convenience function for workflow analysis and escalation.

    Parameters
    ----------
    registry : PrerequisiteRegistry
        Registry with function metadata
    target_function : str
        Target function
    missing_prerequisites : List[str]
        Missing prerequisites
    missing_data : Optional[Dict[str, List[str]]]
        Missing data structures

    Returns
    -------
    EscalationResult
        Escalation analysis result
    """
    escalator = WorkflowEscalator(registry)
    return escalator.should_escalate(target_function, missing_prerequisites, missing_data)
