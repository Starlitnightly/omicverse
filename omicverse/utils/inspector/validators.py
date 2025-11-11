"""
Data structure validators for AnnData objects.

This module provides validators that check for the presence and validity
of required data structures in AnnData objects.
"""

from typing import List, Optional
import numpy as np
from anndata import AnnData

from .data_structures import (
    ObsCheckResult,
    ObsmCheckResult,
    ObspCheckResult,
    UnsCheckResult,
    LayersCheckResult,
    DataCheckResult,
)


class DataValidators:
    """Validates data structure requirements in AnnData objects.

    This class provides methods to check if required data structures
    (obs, obsm, obsp, uns, layers) exist and are properly formatted.

    Attributes:
        adata: The AnnData object to validate.

    Example:
        >>> validator = DataValidators(adata)
        >>> result = validator.check_obsm(['X_pca', 'X_umap'])
        >>> if not result.is_valid:
        ...     print(f"Missing: {result.missing_keys}")
    """

    def __init__(self, adata: AnnData):
        """Initialize validator with an AnnData object.

        Args:
            adata: The AnnData object to validate.
        """
        self.adata = adata

    def check_obs(self, required_columns: List[str]) -> ObsCheckResult:
        """Check if required columns exist in adata.obs.

        Args:
            required_columns: List of column names that must exist.

        Returns:
            ObsCheckResult with validation details.

        Example:
            >>> result = validator.check_obs(['leiden', 'louvain'])
            >>> print(result.is_valid)
        """
        if not required_columns:
            return ObsCheckResult(
                is_valid=True,
                required_columns=[],
                missing_columns=[],
                present_columns=[],
            )

        present = []
        missing = []
        issues = []

        for col in required_columns:
            if col in self.adata.obs.columns:
                present.append(col)

                # Check for common issues
                if self.adata.obs[col].isna().any():
                    issues.append(f"Column '{col}' contains NaN values")

            else:
                missing.append(col)

        is_valid = len(missing) == 0

        return ObsCheckResult(
            is_valid=is_valid,
            required_columns=required_columns,
            missing_columns=missing,
            present_columns=present,
            issues=issues,
        )

    def check_obsm(self, required_keys: List[str]) -> ObsmCheckResult:
        """Check if required embeddings exist in adata.obsm.

        Args:
            required_keys: List of obsm keys that must exist (e.g., 'X_pca').

        Returns:
            ObsmCheckResult with validation details.

        Example:
            >>> result = validator.check_obsm(['X_pca', 'X_umap'])
            >>> print(result.missing_keys)
        """
        if not required_keys:
            return ObsmCheckResult(
                is_valid=True,
                required_keys=[],
                missing_keys=[],
                present_keys=[],
            )

        present = []
        missing = []
        issues = []
        shape_info = {}

        for key in required_keys:
            if key in self.adata.obsm:
                present.append(key)
                shape_info[key] = self.adata.obsm[key].shape

                # Check for common issues
                arr = self.adata.obsm[key]
                if arr.shape[0] != self.adata.n_obs:
                    issues.append(f"Key '{key}' has incorrect shape: {arr.shape}")

                if np.any(np.isnan(arr)):
                    issues.append(f"Key '{key}' contains NaN values")

            else:
                missing.append(key)

        is_valid = len(missing) == 0

        return ObsmCheckResult(
            is_valid=is_valid,
            required_keys=required_keys,
            missing_keys=missing,
            present_keys=present,
            shape_info=shape_info,
            issues=issues,
        )

    def check_obsp(self, required_keys: List[str]) -> ObspCheckResult:
        """Check if required pairwise arrays exist in adata.obsp.

        Args:
            required_keys: List of obsp keys that must exist (e.g., 'connectivities').

        Returns:
            ObspCheckResult with validation details.

        Example:
            >>> result = validator.check_obsp(['connectivities', 'distances'])
            >>> print(result.is_valid)
        """
        if not required_keys:
            return ObspCheckResult(
                is_valid=True,
                required_keys=[],
                missing_keys=[],
                present_keys=[],
            )

        present = []
        missing = []
        issues = []
        is_sparse = {}

        for key in required_keys:
            if key in self.adata.obsp:
                present.append(key)

                # Check if sparse
                from scipy.sparse import issparse
                arr = self.adata.obsp[key]
                is_sparse[key] = issparse(arr)

                # Check shape
                expected_shape = (self.adata.n_obs, self.adata.n_obs)
                if arr.shape != expected_shape:
                    issues.append(
                        f"Key '{key}' has incorrect shape: {arr.shape}, "
                        f"expected {expected_shape}"
                    )

            else:
                missing.append(key)

        is_valid = len(missing) == 0

        return ObspCheckResult(
            is_valid=is_valid,
            required_keys=required_keys,
            missing_keys=missing,
            present_keys=present,
            is_sparse=is_sparse,
            issues=issues,
        )

    def check_uns(self, required_keys: List[str]) -> UnsCheckResult:
        """Check if required unstructured data exists in adata.uns.

        Args:
            required_keys: List of uns keys that must exist.

        Returns:
            UnsCheckResult with validation details.

        Example:
            >>> result = validator.check_uns(['neighbors', 'pca'])
            >>> print(result.present_keys)
        """
        if not required_keys:
            return UnsCheckResult(
                is_valid=True,
                required_keys=[],
                missing_keys=[],
                present_keys=[],
            )

        present = []
        missing = []
        issues = []
        nested_structure = {}

        for key in required_keys:
            if key in self.adata.uns:
                present.append(key)

                # Store structure info for nested dicts
                value = self.adata.uns[key]
                if isinstance(value, dict):
                    nested_structure[key] = list(value.keys())

            else:
                missing.append(key)

        is_valid = len(missing) == 0

        return UnsCheckResult(
            is_valid=is_valid,
            required_keys=required_keys,
            missing_keys=missing,
            present_keys=present,
            nested_structure=nested_structure,
            issues=issues,
        )

    def check_layers(self, required_keys: List[str]) -> LayersCheckResult:
        """Check if required layers exist in adata.layers.

        Args:
            required_keys: List of layer keys that must exist.

        Returns:
            LayersCheckResult with validation details.

        Example:
            >>> result = validator.check_layers(['counts', 'normalized'])
            >>> print(result.missing_keys)
        """
        if not required_keys:
            return LayersCheckResult(
                is_valid=True,
                required_keys=[],
                missing_keys=[],
                present_keys=[],
            )

        present = []
        missing = []
        issues = []
        shape_info = {}

        for key in required_keys:
            if key in self.adata.layers:
                present.append(key)
                shape_info[key] = self.adata.layers[key].shape

                # Check shape matches X
                layer_shape = self.adata.layers[key].shape
                x_shape = self.adata.X.shape
                if layer_shape != x_shape:
                    issues.append(
                        f"Layer '{key}' shape {layer_shape} doesn't match "
                        f"X shape {x_shape}"
                    )

            else:
                missing.append(key)

        is_valid = len(missing) == 0

        return LayersCheckResult(
            is_valid=is_valid,
            required_keys=required_keys,
            missing_keys=missing,
            present_keys=present,
            shape_info=shape_info,
            issues=issues,
        )

    def check_all_requirements(
        self,
        requires: dict,
    ) -> DataCheckResult:
        """Check all data requirements specified in a 'requires' dict.

        Args:
            requires: Dict with keys 'obs', 'obsm', 'obsp', 'uns', 'layers'
                     mapping to lists of required keys.

        Returns:
            DataCheckResult with comprehensive validation details.

        Example:
            >>> requires = {
            ...     'obsm': ['X_pca'],
            ...     'obsp': ['connectivities', 'distances']
            ... }
            >>> result = validator.check_all_requirements(requires)
            >>> print(result.is_valid)
        """
        # Extract requirements
        obs_cols = requires.get('obs', [])
        obsm_keys = requires.get('obsm', [])
        obsp_keys = requires.get('obsp', [])
        uns_keys = requires.get('uns', [])
        layers_keys = requires.get('layers', [])

        # Run all checks
        obs_result = self.check_obs(obs_cols) if obs_cols else None
        obsm_result = self.check_obsm(obsm_keys) if obsm_keys else None
        obsp_result = self.check_obsp(obsp_keys) if obsp_keys else None
        uns_result = self.check_uns(uns_keys) if uns_keys else None
        layers_result = self.check_layers(layers_keys) if layers_keys else None

        # Overall validity
        is_valid = all([
            obs_result.is_valid if obs_result else True,
            obsm_result.is_valid if obsm_result else True,
            obsp_result.is_valid if obsp_result else True,
            uns_result.is_valid if uns_result else True,
            layers_result.is_valid if layers_result else True,
        ])

        return DataCheckResult(
            is_valid=is_valid,
            obs_result=obs_result,
            obsm_result=obsm_result,
            obsp_result=obsp_result,
            uns_result=uns_result,
            layers_result=layers_result,
        )
