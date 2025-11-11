"""
PrerequisiteChecker - Detects executed functions from AnnData state.

This module provides the PrerequisiteChecker class which examines AnnData
objects to detect which preprocessing/analysis functions have been executed.
It uses multiple detection strategies with confidence scoring.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from anndata import AnnData
import numpy as np

from .data_structures import ExecutionEvidence


@dataclass
class DetectionResult:
    """Result of detecting a single function's execution.

    Attributes:
        function_name: Name of the function.
        executed: Whether the function appears to have been executed.
        confidence: Confidence score (0.0 to 1.0).
        evidence: List of ExecutionEvidence objects.
        detection_method: Primary method used for detection.
    """
    function_name: str
    executed: bool
    confidence: float
    evidence: List[ExecutionEvidence]
    detection_method: str

    def __str__(self) -> str:
        status = "✓" if self.executed else "✗"
        return f"{status} {self.function_name} (confidence: {self.confidence:.2f}, method: {self.detection_method})"


class PrerequisiteChecker:
    """Detects which prerequisite functions have been executed.

    This class examines an AnnData object to determine which preprocessing
    and analysis functions have been executed, based on the data structures
    and metadata present in the object.

    Detection strategies (in order of confidence):
    1. Metadata markers (HIGH): Specific keys in adata.uns
    2. Output signatures (MEDIUM): Expected outputs in obs/obsm/obsp
    3. Distribution analysis (LOW): Statistical properties of data

    Attributes:
        adata: The AnnData object to inspect.
        registry: Function registry with Layer 1 metadata.
        detection_results: Cached detection results.

    Example:
        >>> from omicverse.utils.registry import get_registry
        >>> checker = PrerequisiteChecker(adata, get_registry())
        >>> result = checker.check_function_executed('pca')
        >>> if result.executed:
        ...     print(f"PCA was run (confidence: {result.confidence})")
    """

    def __init__(self, adata: AnnData, registry: Any):
        """Initialize checker with AnnData and function registry.

        Args:
            adata: The AnnData object to inspect.
            registry: Function registry with Layer 1 metadata.
        """
        self.adata = adata
        self.registry = registry
        self.detection_results: Dict[str, DetectionResult] = {}

    def check_function_executed(self, function_name: str) -> DetectionResult:
        """Check if a specific function has been executed.

        Args:
            function_name: Name of the function to check.

        Returns:
            DetectionResult with execution status and confidence.

        Example:
            >>> result = checker.check_function_executed('pca')
            >>> print(result.executed, result.confidence)
        """
        # Check cache first
        if function_name in self.detection_results:
            return self.detection_results[function_name]

        # Get function metadata
        func_meta = self._get_function_metadata(function_name)
        if not func_meta:
            result = DetectionResult(
                function_name=function_name,
                executed=False,
                confidence=0.0,
                evidence=[],
                detection_method='unknown',
            )
            self.detection_results[function_name] = result
            return result

        # Try detection strategies in order of confidence
        evidence_list = []

        # Strategy 1: Metadata markers (HIGH confidence)
        metadata_evidence = self._check_metadata_markers(function_name, func_meta)
        evidence_list.extend(metadata_evidence)

        # Strategy 2: Output signatures (MEDIUM confidence)
        output_evidence = self._check_output_signatures(function_name, func_meta)
        evidence_list.extend(output_evidence)

        # Strategy 3: Distribution analysis (LOW confidence)
        # Only use if no high/medium confidence evidence
        if not evidence_list:
            dist_evidence = self._check_distribution_patterns(function_name, func_meta)
            evidence_list.extend(dist_evidence)

        # Calculate overall confidence and execution status
        executed, confidence, method = self._calculate_confidence(evidence_list)

        result = DetectionResult(
            function_name=function_name,
            executed=executed,
            confidence=confidence,
            evidence=evidence_list,
            detection_method=method,
        )

        # Cache result
        self.detection_results[function_name] = result
        return result

    def check_all_prerequisites(self, function_name: str) -> Dict[str, DetectionResult]:
        """Check if all prerequisites for a function have been executed.

        Args:
            function_name: Name of the target function.

        Returns:
            Dict mapping prerequisite function names to DetectionResults.

        Example:
            >>> results = checker.check_all_prerequisites('leiden')
            >>> for func, result in results.items():
            ...     print(f"{func}: {result.executed}")
        """
        func_meta = self._get_function_metadata(function_name)
        if not func_meta:
            return {}

        prerequisites = func_meta.get('prerequisites', {})
        required_funcs = prerequisites.get('functions', [])

        results = {}
        for prereq_func in required_funcs:
            results[prereq_func] = self.check_function_executed(prereq_func)

        return results

    def get_execution_chain(self) -> List[str]:
        """Reconstruct the likely execution chain of functions.

        Returns:
            List of function names in likely execution order.

        Example:
            >>> chain = checker.get_execution_chain()
            >>> print(" -> ".join(chain))
            qc -> preprocess -> pca -> neighbors -> leiden
        """
        executed_functions = []

        # Check all functions in registry
        all_functions = self._get_all_functions()

        for func_name in all_functions:
            result = self.check_function_executed(func_name)
            if result.executed and result.confidence > 0.5:
                executed_functions.append((func_name, result.confidence))

        # Sort by typical workflow order (using registry order as proxy)
        # In a more sophisticated version, we'd use dependency graph
        executed_functions.sort(key=lambda x: x[1], reverse=True)

        return [func for func, conf in executed_functions]

    def _check_metadata_markers(
        self,
        function_name: str,
        func_meta: Dict[str, Any]
    ) -> List[ExecutionEvidence]:
        """Check for metadata markers in adata.uns.

        This is the highest confidence detection method. Many functions
        write metadata to adata.uns with specific keys.

        Args:
            function_name: Name of the function.
            func_meta: Function metadata from registry.

        Returns:
            List of ExecutionEvidence objects (HIGH confidence).
        """
        evidence = []

        # Get expected metadata markers from 'produces'
        produces = func_meta.get('produces', {})
        uns_keys = produces.get('uns', [])

        for key in uns_keys:
            if self._check_uns_key_exists(key):
                evidence.append(ExecutionEvidence(
                    evidence_type='metadata_marker',
                    confidence=0.95,  # Very high confidence
                    location=f'adata.uns["{key}"]',
                    description=f'Found metadata marker "{key}" in uns',
                    function_name=function_name,
                ))

        # Special case: Check for common metadata patterns
        # pca -> 'pca' in uns
        if function_name == 'pca' and 'pca' in self.adata.uns:
            evidence.append(ExecutionEvidence(
                evidence_type='metadata_marker',
                confidence=0.95,
                location='adata.uns["pca"]',
                description='PCA metadata found in uns',
                function_name='pca',
            ))

        # neighbors -> 'neighbors' in uns
        if function_name == 'neighbors' and 'neighbors' in self.adata.uns:
            evidence.append(ExecutionEvidence(
                evidence_type='metadata_marker',
                confidence=0.95,
                location='adata.uns["neighbors"]',
                description='Neighbors metadata found in uns',
                function_name='neighbors',
            ))

        return evidence

    def _check_output_signatures(
        self,
        function_name: str,
        func_meta: Dict[str, Any]
    ) -> List[ExecutionEvidence]:
        """Check for expected output signatures.

        This is medium confidence detection. We check if the expected
        outputs are present in obs, obsm, obsp, etc.

        Args:
            function_name: Name of the function.
            func_meta: Function metadata from registry.

        Returns:
            List of ExecutionEvidence objects (MEDIUM confidence).
        """
        evidence = []

        produces = func_meta.get('produces', {})

        # Check obs columns
        obs_cols = produces.get('obs', [])
        for col in obs_cols:
            if col in self.adata.obs.columns:
                evidence.append(ExecutionEvidence(
                    evidence_type='output_signature',
                    confidence=0.75,  # Medium-high confidence
                    location=f'adata.obs["{col}"]',
                    description=f'Found expected output column "{col}"',
                    function_name=function_name,
                ))

        # Check obsm keys
        obsm_keys = produces.get('obsm', [])
        for key in obsm_keys:
            if key in self.adata.obsm.keys():
                evidence.append(ExecutionEvidence(
                    evidence_type='output_signature',
                    confidence=0.80,  # High-medium confidence
                    location=f'adata.obsm["{key}"]',
                    description=f'Found expected embedding "{key}"',
                    function_name=function_name,
                ))

        # Check obsp keys
        obsp_keys = produces.get('obsp', [])
        for key in obsp_keys:
            if key in self.adata.obsp.keys():
                evidence.append(ExecutionEvidence(
                    evidence_type='output_signature',
                    confidence=0.80,
                    location=f'adata.obsp["{key}"]',
                    description=f'Found expected pairwise array "{key}"',
                    function_name=function_name,
                ))

        return evidence

    def _check_distribution_patterns(
        self,
        function_name: str,
        func_meta: Dict[str, Any]
    ) -> List[ExecutionEvidence]:
        """Check for distribution patterns in data.

        This is the lowest confidence detection method. We examine
        statistical properties of the data to infer preprocessing steps.

        Args:
            function_name: Name of the function.
            func_meta: Function metadata from registry.

        Returns:
            List of ExecutionEvidence objects (LOW confidence).
        """
        evidence = []

        # Only use for specific preprocessing functions
        # where we can infer from data properties

        if function_name == 'scale':
            # Check if data appears to be scaled (mean ~0, std ~1)
            if self._check_scaled_data():
                evidence.append(ExecutionEvidence(
                    evidence_type='distribution_pattern',
                    confidence=0.40,  # Low confidence
                    location='adata.X',
                    description='Data appears scaled (mean≈0, std≈1)',
                    function_name='scale',
                ))

        if function_name == 'preprocess':
            # Check if data appears normalized
            if self._check_normalized_data():
                evidence.append(ExecutionEvidence(
                    evidence_type='distribution_pattern',
                    confidence=0.35,
                    location='adata.X',
                    description='Data appears normalized (library-size corrected)',
                    function_name='preprocess',
                ))

        return evidence

    def _calculate_confidence(
        self,
        evidence_list: List[ExecutionEvidence]
    ) -> Tuple[bool, float, str]:
        """Calculate overall confidence from evidence.

        Args:
            evidence_list: List of ExecutionEvidence objects.

        Returns:
            Tuple of (executed: bool, confidence: float, method: str).
        """
        if not evidence_list:
            return False, 0.0, 'no_evidence'

        # Find highest confidence evidence
        max_evidence = max(evidence_list, key=lambda e: e.confidence)

        # If we have high confidence evidence, trust it
        if max_evidence.confidence >= 0.85:
            return True, max_evidence.confidence, max_evidence.evidence_type

        # For medium confidence, require multiple pieces or high single confidence
        medium_evidence = [e for e in evidence_list if e.confidence >= 0.60]
        if len(medium_evidence) >= 2:
            # Multiple medium confidence -> high confidence
            avg_confidence = sum(e.confidence for e in medium_evidence) / len(medium_evidence)
            return True, min(0.90, avg_confidence + 0.10), 'multiple_evidence'

        if max_evidence.confidence >= 0.70:
            return True, max_evidence.confidence, max_evidence.evidence_type

        # Low confidence - mark as uncertain
        if max_evidence.confidence >= 0.30:
            return True, max_evidence.confidence, max_evidence.evidence_type

        return False, max_evidence.confidence, max_evidence.evidence_type

    def _check_uns_key_exists(self, key: str) -> bool:
        """Check if a key exists in adata.uns.

        Handles nested keys like 'neighbors.params.n_neighbors'.

        Args:
            key: The key to check (may be nested with dots).

        Returns:
            True if key exists, False otherwise.
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self.adata.uns
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return True
        else:
            return key in self.adata.uns

    def _check_scaled_data(self) -> bool:
        """Check if data appears to be scaled.

        Returns:
            True if data looks scaled, False otherwise.
        """
        try:
            # Sample a subset for efficiency
            X = self.adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()

            # Check first 1000 cells and 100 genes
            X_sample = X[:min(1000, X.shape[0]), :min(100, X.shape[1])]

            mean = np.mean(X_sample, axis=0)
            std = np.std(X_sample, axis=0)

            # Check if mean is close to 0 and std close to 1
            mean_near_zero = np.abs(mean).mean() < 0.5
            std_near_one = np.abs(std - 1.0).mean() < 0.5

            return mean_near_zero and std_near_one
        except Exception:
            return False

    def _check_normalized_data(self) -> bool:
        """Check if data appears to be normalized.

        Returns:
            True if data looks normalized, False otherwise.
        """
        try:
            X = self.adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()

            # Check first 1000 cells
            X_sample = X[:min(1000, X.shape[0]), :]

            # Normalized data typically has similar library sizes per cell
            lib_sizes = X_sample.sum(axis=1)
            cv = np.std(lib_sizes) / np.mean(lib_sizes)

            # Low coefficient of variation suggests normalization
            return cv < 0.3
        except Exception:
            return False

    def _get_function_metadata(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a function from the registry.

        Args:
            function_name: Name of the function.

        Returns:
            Metadata dict or None if not found.
        """
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
        """Get list of all functions in registry.

        Returns:
            List of function names.
        """
        try:
            if hasattr(self.registry, 'get_all_functions'):
                return self.registry.get_all_functions()
            elif hasattr(self.registry, 'functions'):
                return list(self.registry.functions.keys())
            else:
                return list(self.registry.keys())
        except Exception:
            return []
