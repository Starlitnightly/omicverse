"""
Main DataStateInspector class for runtime prerequisite validation.

This module provides the primary interface for validating prerequisites
before function execution in OmicVerse workflows.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from anndata import AnnData

from .validators import DataValidators
from .data_structures import (
    ValidationResult,
    DataCheckResult,
    Suggestion,
)


class DataStateInspector:
    """Runtime validator for OmicVerse prerequisite tracking.

    This class inspects AnnData objects to validate that all prerequisites
    for a given function are satisfied before execution. It checks both
    data structure requirements and prerequisite function execution.

    Attributes:
        adata: The AnnData object to inspect.
        registry: Function registry with Layer 1 metadata.
        validators: DataValidators instance for checking data structures.

    Example:
        >>> from omicverse.utils.registry import get_registry
        >>> inspector = DataStateInspector(adata, get_registry())
        >>> result = inspector.validate_prerequisites('leiden')
        >>> if not result.is_valid:
        ...     print(result.message)
        ...     for suggestion in result.suggestions:
        ...         print(suggestion.code)
    """

    def __init__(self, adata: AnnData, registry: Any):
        """Initialize inspector with AnnData and function registry.

        Args:
            adata: The AnnData object to inspect.
            registry: Function registry with Layer 1 metadata.
        """
        self.adata = adata
        self.registry = registry
        self.validators = DataValidators(adata)

    def validate_prerequisites(self, function_name: str) -> ValidationResult:
        """Validate all prerequisites for a given function.

        This is the main entry point for validation. It checks:
        1. Required data structures exist
        2. Required prerequisite functions have been executed (Phase 2)
        3. Generates suggestions to fix any issues

        Args:
            function_name: Name of the function to validate.

        Returns:
            ValidationResult with comprehensive validation details.

        Example:
            >>> result = inspector.validate_prerequisites('leiden')
            >>> print(result.is_valid)
            >>> print(result.missing_data_structures)
        """
        # Get function metadata from registry
        func_meta = self._get_function_metadata(function_name)

        if not func_meta:
            return ValidationResult(
                function_name=function_name,
                is_valid=False,
                message=f"Function '{function_name}' not found in registry",
            )

        # Check data requirements
        data_check = self.check_data_requirements(function_name)

        # For Phase 1, we only validate data structures
        # Prerequisite function checking will be added in Phase 2

        if data_check.is_valid:
            return ValidationResult(
                function_name=function_name,
                is_valid=True,
                message=f"All requirements satisfied for {function_name}",
                data_check_result=data_check,
            )

        # Generate suggestions for missing requirements
        suggestions = self._generate_data_suggestions(
            function_name,
            data_check,
            func_meta
        )

        return ValidationResult(
            function_name=function_name,
            is_valid=False,
            message=f"Missing requirements for {function_name}",
            missing_data_structures=data_check.all_missing_structures,
            suggestions=suggestions,
            data_check_result=data_check,
        )

    def check_data_requirements(self, function_name: str) -> DataCheckResult:
        """Check if required data structures exist for a function.

        Args:
            function_name: Name of the function to check.

        Returns:
            DataCheckResult with details about missing/present structures.

        Example:
            >>> result = inspector.check_data_requirements('leiden')
            >>> print(result.all_missing_structures)
        """
        func_meta = self._get_function_metadata(function_name)

        if not func_meta:
            # Return invalid result if function not found
            return DataCheckResult(is_valid=False)

        requires = func_meta.get('requires', {})
        return self.validators.check_all_requirements(requires)

    def _get_function_metadata(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a function from the registry.

        Args:
            function_name: Name of the function.

        Returns:
            Metadata dict or None if not found.
        """
        try:
            # Access registry to get function metadata
            # The exact method depends on registry implementation
            if hasattr(self.registry, 'get_function'):
                return self.registry.get_function(function_name)
            elif hasattr(self.registry, 'functions'):
                return self.registry.functions.get(function_name)
            else:
                # Try direct dictionary access
                return self.registry.get(function_name)
        except Exception:
            return None

    def _generate_data_suggestions(
        self,
        function_name: str,
        data_check: DataCheckResult,
        func_meta: Dict[str, Any],
    ) -> List[Suggestion]:
        """Generate suggestions to fix missing data requirements.

        Args:
            function_name: Name of the target function.
            data_check: Result of data requirements check.
            func_meta: Function metadata from registry.

        Returns:
            List of Suggestion objects in priority order.
        """
        suggestions = []
        missing = data_check.all_missing_structures

        # Get prerequisite information
        prerequisites = func_meta.get('prerequisites', {})
        required_funcs = prerequisites.get('functions', [])
        auto_fix = func_meta.get('auto_fix', 'none')

        # Generate suggestions based on missing structures
        if 'obsp' in missing:
            # Missing graph structures typically means neighbors not run
            if 'connectivities' in missing['obsp'] or 'distances' in missing['obsp']:
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Compute neighbor graph (required for clustering/UMAP)',
                    code='sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
                    explanation=(
                        'The function requires a neighbor graph (connectivities and distances). '
                        'Run sc.pp.neighbors() to compute the k-nearest neighbors graph.'
                    ),
                    estimated_time='10-30 seconds',
                    estimated_time_seconds=20,
                    prerequisites=['pca'] if 'X_pca' not in self.adata.obsm else [],
                    impact='Enables clustering and graph-based analyses',
                    auto_executable=(auto_fix == 'auto'),
                ))

        if 'obsm' in missing:
            # Missing embeddings
            if 'X_pca' in missing['obsm']:
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Run PCA for dimensionality reduction',
                    code='ov.pp.scale(adata)\nov.pp.pca(adata, n_pcs=50)',
                    explanation=(
                        'PCA is required but not found. Scale the data and run PCA.'
                    ),
                    estimated_time='5-20 seconds',
                    estimated_time_seconds=10,
                    prerequisites=['preprocess', 'scale'],
                    impact='Required for downstream analyses',
                    auto_executable=False,
                ))

            if 'X_umap' in missing['obsm']:
                suggestions.append(Suggestion(
                    priority='MEDIUM',
                    suggestion_type='direct_fix',
                    description='Compute UMAP embedding',
                    code='ov.pp.umap(adata)',
                    explanation=(
                        'UMAP embedding is required. Ensure neighbors graph exists first.'
                    ),
                    estimated_time='10-60 seconds',
                    estimated_time_seconds=30,
                    prerequisites=['neighbors'],
                    impact='Required for visualization',
                    auto_executable=False,
                ))

        if 'obs' in missing:
            # Missing observation columns
            obs_missing = missing['obs']
            if any('leiden' in col or 'louvain' in col for col in obs_missing):
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Run clustering to generate cluster labels',
                    code='ov.pp.leiden(adata, resolution=1.0)',
                    explanation=(
                        'Cluster labels are required. Run Leiden or Louvain clustering.'
                    ),
                    estimated_time='5-30 seconds',
                    estimated_time_seconds=15,
                    prerequisites=['neighbors'],
                    impact='Generates cell cluster assignments',
                    auto_executable=False,
                ))

        # If we have required functions listed, suggest running them
        if required_funcs and auto_fix == 'escalate':
            func_list = ', '.join(required_funcs)
            suggestions.append(Suggestion(
                priority='HIGH',
                suggestion_type='workflow_guidance',
                description=f'Complete prerequisite workflow: {func_list}',
                code=f'# Run prerequisite functions: {func_list}\n# See function documentation for details',
                explanation=(
                    f'{function_name} requires a complex workflow. '
                    f'Ensure these functions have been run: {func_list}'
                ),
                estimated_time='Varies',
                prerequisites=required_funcs,
                impact='Satisfies all prerequisites',
                auto_executable=False,
            ))

        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 999))

        return suggestions

    def get_validation_summary(self, function_name: str) -> Dict[str, Any]:
        """Get a dictionary summary of validation suitable for LLM consumption.

        Args:
            function_name: Name of the function to validate.

        Returns:
            Dict with validation details in LLM-friendly format.

        Example:
            >>> summary = inspector.get_validation_summary('leiden')
            >>> print(summary['missing_data_structures'])
        """
        result = self.validate_prerequisites(function_name)
        return result.get_summary()
