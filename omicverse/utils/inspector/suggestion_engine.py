"""
SuggestionEngine - Enhanced suggestion generation with workflow planning.

This module provides the SuggestionEngine class which generates comprehensive,
actionable suggestions to fix missing prerequisites and data requirements.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .data_structures import Suggestion, DataCheckResult


class WorkflowStrategy(Enum):
    """Workflow strategy for completing prerequisites."""
    MINIMAL = "minimal"  # Shortest path to completion
    COMPREHENSIVE = "comprehensive"  # Complete recommended workflow
    ALTERNATIVE = "alternative"  # Alternative approach


@dataclass
class WorkflowStep:
    """A single step in a workflow plan."""

    function_name: str
    description: str
    code: str
    estimated_time_seconds: int

    # Dependencies
    requires_functions: List[str] = field(default_factory=list)
    requires_data: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    is_optional: bool = False
    alternatives: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        optional = " (optional)" if self.is_optional else ""
        return f"{self.function_name}{optional}: {self.description}"


@dataclass
class WorkflowPlan:
    """A complete workflow plan to satisfy requirements."""

    name: str
    description: str
    strategy: WorkflowStrategy
    steps: List[WorkflowStep]

    total_time_seconds: int = 0
    complexity: str = "MEDIUM"  # LOW, MEDIUM, HIGH

    def __post_init__(self):
        """Calculate total time and complexity."""
        self.total_time_seconds = sum(step.estimated_time_seconds for step in self.steps)

        # Determine complexity based on number of steps
        if len(self.steps) <= 2:
            self.complexity = "LOW"
        elif len(self.steps) <= 5:
            self.complexity = "MEDIUM"
        else:
            self.complexity = "HIGH"

    def get_summary(self) -> str:
        """Get a summary of the workflow plan."""
        steps_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(self.steps))
        time_str = self._format_time(self.total_time_seconds)

        return f"""
Workflow: {self.name}
Strategy: {self.strategy.value}
Complexity: {self.complexity}
Estimated Time: {time_str}

Steps:
{steps_str}
"""

    def _format_time(self, seconds: int) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"


class SuggestionEngine:
    """Enhanced suggestion generator with workflow planning.

    This class generates comprehensive, prioritized suggestions to fix
    missing prerequisites and data requirements. It provides:
    - Multi-step workflow plans
    - Alternative approaches
    - Cost-benefit analysis
    - Dependency resolution

    Attributes:
        registry: Function registry with metadata.
        function_graph: Dependency graph of functions.

    Example:
        >>> engine = SuggestionEngine(registry)
        >>> suggestions = engine.generate_suggestions(
        ...     function_name='leiden',
        ...     missing_prerequisites=['neighbors', 'pca'],
        ...     missing_data={'obsm': ['X_pca']}
        ... )
        >>> for suggestion in suggestions:
        ...     print(suggestion.code)
    """

    def __init__(self, registry: Any):
        """Initialize suggestion engine with registry.

        Args:
            registry: Function registry with Layer 1 metadata.
        """
        self.registry = registry
        self.function_graph = self._build_function_graph()

        # Common function code templates
        self.function_templates = {
            'qc': 'ov.pp.qc(adata, tresh={{"mito_perc": 20, "n_genes_by_counts": 2500}})',
            'preprocess': 'ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)',
            'scale': 'ov.pp.scale(adata)',
            'pca': 'ov.pp.pca(adata, n_pcs=50)',
            'neighbors': 'ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
            'umap': 'ov.pp.umap(adata)',
            'leiden': 'ov.pp.leiden(adata, resolution=1.0)',
            'louvain': 'ov.pp.louvain(adata, resolution=1.0)',
        }

    def generate_suggestions(
        self,
        function_name: str,
        missing_prerequisites: List[str] = None,
        missing_data: Dict[str, List[str]] = None,
        data_check_result: Optional[DataCheckResult] = None,
    ) -> List[Suggestion]:
        """Generate comprehensive suggestions to fix requirements.

        Args:
            function_name: Target function name.
            missing_prerequisites: List of missing prerequisite functions.
            missing_data: Dict of missing data structures.
            data_check_result: Complete data check result.

        Returns:
            List of Suggestion objects, sorted by priority.
        """
        suggestions = []

        missing_prerequisites = missing_prerequisites or []
        missing_data = missing_data or {}

        # Generate workflow plans
        if missing_prerequisites:
            workflow_suggestions = self._generate_workflow_suggestions(
                function_name,
                missing_prerequisites
            )
            suggestions.extend(workflow_suggestions)

        # Generate data-specific suggestions
        if missing_data:
            data_suggestions = self._generate_data_suggestions(
                function_name,
                missing_data,
                data_check_result
            )
            suggestions.extend(data_suggestions)

        # Generate alternative approaches
        alternative_suggestions = self._generate_alternatives(
            function_name,
            missing_prerequisites,
            missing_data
        )
        suggestions.extend(alternative_suggestions)

        # Sort by priority and estimated time
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        suggestions.sort(key=lambda s: (
            priority_order.get(s.priority, 999),
            s.estimated_time_seconds
        ))

        return suggestions

    def create_workflow_plan(
        self,
        function_name: str,
        missing_prerequisites: List[str],
        strategy: WorkflowStrategy = WorkflowStrategy.MINIMAL,
    ) -> WorkflowPlan:
        """Create a workflow plan to satisfy all prerequisites.

        Args:
            function_name: Target function name.
            missing_prerequisites: List of missing prerequisite functions.
            strategy: Workflow strategy (minimal, comprehensive, alternative).

        Returns:
            WorkflowPlan with ordered steps.
        """
        # Resolve dependencies and order steps
        ordered_funcs = self._resolve_dependencies(missing_prerequisites)

        # Create workflow steps
        steps = []
        for func in ordered_funcs:
            step = self._create_workflow_step(func)
            steps.append(step)

        # Add optional steps for comprehensive strategy
        if strategy == WorkflowStrategy.COMPREHENSIVE:
            optional_steps = self._get_optional_steps(function_name)
            steps.extend(optional_steps)

        # Create plan
        plan = WorkflowPlan(
            name=f"Complete prerequisites for {function_name}",
            description=f"Execute required functions to prepare data for {function_name}",
            strategy=strategy,
            steps=steps,
        )

        return plan

    def _generate_workflow_suggestions(
        self,
        function_name: str,
        missing_prerequisites: List[str],
    ) -> List[Suggestion]:
        """Generate workflow-based suggestions.

        Args:
            function_name: Target function name.
            missing_prerequisites: Missing prerequisite functions.

        Returns:
            List of workflow suggestions.
        """
        suggestions = []

        # Create minimal workflow plan
        minimal_plan = self.create_workflow_plan(
            function_name,
            missing_prerequisites,
            WorkflowStrategy.MINIMAL
        )

        # Generate code for minimal workflow
        code_lines = ["# Minimal workflow to satisfy prerequisites"]
        for i, step in enumerate(minimal_plan.steps, 1):
            code_lines.append(f"\n# Step {i}: {step.description}")
            code_lines.append(step.code)

        code = "\n".join(code_lines)

        suggestions.append(Suggestion(
            priority='HIGH',
            suggestion_type='workflow',
            description=f'Complete prerequisite workflow ({len(minimal_plan.steps)} steps)',
            code=code,
            explanation=(
                f'{function_name} requires {len(missing_prerequisites)} prerequisite function(s). '
                f'Execute them in the order shown below. '
                f'Total estimated time: {minimal_plan._format_time(minimal_plan.total_time_seconds)}.'
            ),
            estimated_time=minimal_plan._format_time(minimal_plan.total_time_seconds),
            estimated_time_seconds=minimal_plan.total_time_seconds,
            prerequisites=[],
            impact=f'Satisfies all {len(missing_prerequisites)} missing prerequisites',
            auto_executable=False,
        ))

        # Generate individual step suggestions for each missing prerequisite
        for prereq in missing_prerequisites:
            step = self._create_workflow_step(prereq)

            # Determine priority based on position in chain
            priority = 'CRITICAL' if prereq in ['qc', 'preprocess'] else 'HIGH'

            suggestions.append(Suggestion(
                priority=priority,
                suggestion_type='prerequisite',
                description=f'Run prerequisite: {prereq}',
                code=step.code,
                explanation=(
                    f'{function_name} requires {prereq} to be executed first. '
                    f'{step.description}.'
                ),
                estimated_time=f'{step.estimated_time_seconds} seconds',
                estimated_time_seconds=step.estimated_time_seconds,
                prerequisites=step.requires_functions,
                impact=f'Satisfies prerequisite: {prereq}',
                auto_executable=False,
            ))

        return suggestions

    def _generate_data_suggestions(
        self,
        function_name: str,
        missing_data: Dict[str, List[str]],
        data_check_result: Optional[DataCheckResult],
    ) -> List[Suggestion]:
        """Generate suggestions for missing data structures.

        Args:
            function_name: Target function name.
            missing_data: Missing data structures by type.
            data_check_result: Complete data check result.

        Returns:
            List of data-related suggestions.
        """
        suggestions = []

        # Check for missing obs columns
        if 'obs' in missing_data:
            obs_suggestions = self._suggest_obs_fixes(
                function_name,
                missing_data['obs']
            )
            suggestions.extend(obs_suggestions)

        # Check for missing obsm keys (embeddings)
        if 'obsm' in missing_data:
            obsm_suggestions = self._suggest_obsm_fixes(
                function_name,
                missing_data['obsm']
            )
            suggestions.extend(obsm_suggestions)

        # Check for missing obsp keys (graphs)
        if 'obsp' in missing_data:
            obsp_suggestions = self._suggest_obsp_fixes(
                function_name,
                missing_data['obsp']
            )
            suggestions.extend(obsp_suggestions)

        return suggestions

    def _suggest_obs_fixes(
        self,
        function_name: str,
        missing_columns: List[str],
    ) -> List[Suggestion]:
        """Suggest fixes for missing obs columns."""
        suggestions = []

        for col in missing_columns:
            # Detect what type of column is missing
            if 'leiden' in col.lower() or 'louvain' in col.lower():
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description=f'Run clustering to generate {col}',
                    code=f'ov.pp.leiden(adata, resolution=1.0)  # Generates leiden column',
                    explanation=(
                        f'{function_name} requires cluster labels in adata.obs["{col}"]. '
                        'Run Leiden or Louvain clustering to generate cluster assignments.'
                    ),
                    estimated_time='5-30 seconds',
                    estimated_time_seconds=15,
                    prerequisites=['neighbors'],
                    impact=f'Generates cluster labels in adata.obs["{col}"]',
                    auto_executable=False,
                ))
            else:
                # Generic obs column suggestion
                suggestions.append(Suggestion(
                    priority='MEDIUM',
                    suggestion_type='manual_fix',
                    description=f'Add missing column: {col}',
                    code=f'# adata.obs["{col}"] = ...  # Add {col} column manually',
                    explanation=(
                        f'{function_name} requires adata.obs["{col}"]. '
                        'This column may need to be created manually based on your data.'
                    ),
                    estimated_time='Variable',
                    estimated_time_seconds=60,
                    prerequisites=[],
                    impact=f'Adds required column: {col}',
                    auto_executable=False,
                ))

        return suggestions

    def _suggest_obsm_fixes(
        self,
        function_name: str,
        missing_keys: List[str],
    ) -> List[Suggestion]:
        """Suggest fixes for missing obsm embeddings."""
        suggestions = []

        for key in missing_keys:
            if 'pca' in key.lower():
                suggestions.append(Suggestion(
                    priority='CRITICAL',
                    suggestion_type='direct_fix',
                    description='Run PCA to generate embeddings',
                    code='ov.pp.pca(adata, n_pcs=50)',
                    explanation=(
                        f'{function_name} requires PCA embeddings in adata.obsm["{key}"]. '
                        'Run PCA on preprocessed data.'
                    ),
                    estimated_time='10-60 seconds',
                    estimated_time_seconds=30,
                    prerequisites=['preprocess'],
                    impact=f'Generates PCA embeddings in adata.obsm["{key}"]',
                    auto_executable=False,
                ))
            elif 'umap' in key.lower():
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Run UMAP to generate embeddings',
                    code='ov.pp.umap(adata)',
                    explanation=(
                        f'{function_name} requires UMAP embeddings in adata.obsm["{key}"]. '
                        'Run UMAP on neighbor graph.'
                    ),
                    estimated_time='10-120 seconds',
                    estimated_time_seconds=45,
                    prerequisites=['neighbors'],
                    impact=f'Generates UMAP embeddings in adata.obsm["{key}"]',
                    auto_executable=False,
                ))

        return suggestions

    def _suggest_obsp_fixes(
        self,
        function_name: str,
        missing_keys: List[str],
    ) -> List[Suggestion]:
        """Suggest fixes for missing obsp graphs."""
        suggestions = []

        # Usually obsp keys are from neighbors
        if any('connect' in k.lower() or 'distance' in k.lower() for k in missing_keys):
            suggestions.append(Suggestion(
                priority='CRITICAL',
                suggestion_type='direct_fix',
                description='Run neighbors to generate graph',
                code='ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
                explanation=(
                    f'{function_name} requires neighbor graph in adata.obsp. '
                    'Run neighbors to compute connectivity and distance graphs.'
                ),
                estimated_time='10-120 seconds',
                estimated_time_seconds=45,
                prerequisites=['pca'],
                impact='Generates connectivities and distances in adata.obsp',
                auto_executable=False,
            ))

        return suggestions

    def _generate_alternatives(
        self,
        function_name: str,
        missing_prerequisites: List[str],
        missing_data: Dict[str, List[str]],
    ) -> List[Suggestion]:
        """Generate alternative approaches."""
        suggestions = []

        # Alternative clustering methods
        if function_name == 'leiden':
            suggestions.append(Suggestion(
                priority='LOW',
                suggestion_type='alternative',
                description='Alternative: Use Louvain clustering instead',
                code='ov.pp.louvain(adata, resolution=1.0)',
                explanation=(
                    'Louvain is an alternative clustering algorithm. '
                    'It produces similar results to Leiden but may be faster.'
                ),
                estimated_time='5-30 seconds',
                estimated_time_seconds=15,
                prerequisites=['neighbors'],
                impact='Generates cluster labels using Louvain algorithm',
                auto_executable=False,
            ))
        elif function_name == 'louvain':
            suggestions.append(Suggestion(
                priority='LOW',
                suggestion_type='alternative',
                description='Alternative: Use Leiden clustering instead',
                code='ov.pp.leiden(adata, resolution=1.0)',
                explanation=(
                    'Leiden is an improved version of Louvain clustering. '
                    'It may produce better cluster quality.'
                ),
                estimated_time='5-30 seconds',
                estimated_time_seconds=15,
                prerequisites=['neighbors'],
                impact='Generates cluster labels using Leiden algorithm',
                auto_executable=False,
            ))

        return suggestions

    def _create_workflow_step(self, function_name: str) -> WorkflowStep:
        """Create a workflow step for a function."""
        # Get function metadata
        func_meta = self._get_function_metadata(function_name)

        # Get code template
        code = self.function_templates.get(
            function_name,
            f'# Run {function_name}\n# See documentation for details'
        )

        # Get prerequisites
        prereqs = []
        if func_meta:
            prereq_info = func_meta.get('prerequisites', {})
            prereqs = prereq_info.get('functions', [])

        # Estimate time
        time_estimates = {
            'qc': 5,
            'preprocess': 30,
            'scale': 10,
            'pca': 30,
            'neighbors': 45,
            'umap': 45,
            'leiden': 15,
            'louvain': 15,
        }
        estimated_time = time_estimates.get(function_name, 30)

        # Get description
        descriptions = {
            'qc': 'Quality control filtering',
            'preprocess': 'Normalize and identify highly variable genes',
            'scale': 'Scale data to unit variance',
            'pca': 'Dimensionality reduction via PCA',
            'neighbors': 'Compute neighbor graph',
            'umap': 'Compute UMAP embedding',
            'leiden': 'Cluster cells with Leiden algorithm',
            'louvain': 'Cluster cells with Louvain algorithm',
        }
        description = descriptions.get(function_name, f'Execute {function_name}')

        return WorkflowStep(
            function_name=function_name,
            description=description,
            code=code,
            estimated_time_seconds=estimated_time,
            requires_functions=prereqs,
        )

    def _resolve_dependencies(self, functions: List[str]) -> List[str]:
        """Resolve dependencies and return ordered list of functions.

        Args:
            functions: List of function names to order.

        Returns:
            Ordered list with dependencies resolved.
        """
        # Build dependency graph for these functions
        graph = {}
        for func in functions:
            func_meta = self._get_function_metadata(func)
            if func_meta:
                prereq_info = func_meta.get('prerequisites', {})
                prereqs = prereq_info.get('functions', [])
                # Only include prerequisites that are in our list
                graph[func] = [p for p in prereqs if p in functions]
            else:
                graph[func] = []

        # Topological sort
        ordered = []
        visited = set()

        def visit(func):
            if func in visited:
                return
            visited.add(func)
            for prereq in graph.get(func, []):
                visit(prereq)
            ordered.append(func)

        for func in functions:
            visit(func)

        return ordered

    def _get_optional_steps(self, function_name: str) -> List[WorkflowStep]:
        """Get optional steps that might improve results."""
        optional_steps = []

        # For clustering, suggest UMAP for visualization
        if function_name in ['leiden', 'louvain']:
            func_meta = self._get_function_metadata('umap')
            if func_meta:
                step = self._create_workflow_step('umap')
                step.is_optional = True
                step.description = 'Generate UMAP for visualization (optional)'
                optional_steps.append(step)

        return optional_steps

    def _build_function_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph of all functions in registry."""
        graph = {}

        try:
            all_funcs = self._get_all_functions()
            for func in all_funcs:
                func_meta = self._get_function_metadata(func)
                if func_meta:
                    prereq_info = func_meta.get('prerequisites', {})
                    prereqs = prereq_info.get('functions', [])
                    graph[func] = prereqs
                else:
                    graph[func] = []
        except Exception:
            # If registry doesn't support this, use empty graph
            pass

        return graph

    def _get_function_metadata(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a function from registry."""
        try:
            if hasattr(self.registry, 'get_function'):
                return self.registry.get_function(function_name)
            elif hasattr(self.registry, 'functions'):
                return self.registry.functions.get(function_name)
            else:
                return self.registry.get(function_name)
        except Exception:
            return None

    def _get_all_functions(self) -> List[str]:
        """Get list of all functions in registry."""
        try:
            if hasattr(self.registry, 'get_all_functions'):
                return self.registry.get_all_functions()
            elif hasattr(self.registry, 'functions'):
                return list(self.registry.functions.keys())
            else:
                return list(self.registry.keys())
        except Exception:
            return []
