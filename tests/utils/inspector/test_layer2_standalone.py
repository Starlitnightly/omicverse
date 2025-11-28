"""
Standalone test for Layer 2 Phase 1 implementation.

This test imports the inspector modules directly without importing
the full omicverse package.
"""

import sys
import os
from pathlib import Path

# Add the omicverse directory to sys.path so we can import omicverse.utils.inspector
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData

# Now we can import from omicverse.utils.inspector
# But we need to avoid importing the full omicverse package
# Import the specific modules directly
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Import data_structures first (no dependencies on other local modules)
data_structures = import_module_from_path(
    'omicverse.utils.inspector.data_structures',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/data_structures.py')
)

# Import validators (depends on data_structures)
validators = import_module_from_path(
    'omicverse.utils.inspector.validators',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/validators.py')
)

# Get the classes we need
DataValidators = validators.DataValidators
ValidationResult = data_structures.ValidationResult
DataCheckResult = data_structures.DataCheckResult
ObsCheckResult = data_structures.ObsCheckResult
ObsmCheckResult = data_structures.ObsmCheckResult
ObspCheckResult = data_structures.ObspCheckResult
UnsCheckResult = data_structures.UnsCheckResult
LayersCheckResult = data_structures.LayersCheckResult


def create_test_adata():
    """Create a minimal test AnnData object."""
    X = np.random.rand(100, 50)
    obs = pd.DataFrame({'cell_type': ['A'] * 50 + ['B'] * 50}, index=[f'cell_{i}' for i in range(100)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(50)])
    return AnnData(X=X, obs=obs, var=var)


def test_check_obs_valid():
    """Test obs validation with valid columns."""
    print("Testing obs validation (valid case)...")
    adata = create_test_adata()
    validator = DataValidators(adata)

    result = validator.check_obs(['cell_type'])

    assert result.is_valid, "Should be valid"
    assert 'cell_type' in result.present_columns, "cell_type should be present"
    assert len(result.missing_columns) == 0, "No columns should be missing"
    print("✓ test_check_obs_valid passed")


def test_check_obs_missing():
    """Test obs validation with missing columns."""
    print("Testing obs validation (missing case)...")
    adata = create_test_adata()
    validator = DataValidators(adata)

    result = validator.check_obs(['cell_type', 'leiden', 'batch'])

    assert not result.is_valid, "Should be invalid"
    assert 'cell_type' in result.present_columns, "cell_type should be present"
    assert 'leiden' in result.missing_columns, "leiden should be missing"
    assert 'batch' in result.missing_columns, "batch should be missing"
    print("✓ test_check_obs_missing passed")


def test_check_obsm_valid():
    """Test obsm validation with valid keys."""
    print("Testing obsm validation (valid case)...")
    adata = create_test_adata()
    adata.obsm['X_pca'] = np.random.rand(100, 50)
    adata.obsm['X_umap'] = np.random.rand(100, 2)

    validator = DataValidators(adata)
    result = validator.check_obsm(['X_pca', 'X_umap'])

    assert result.is_valid, "Should be valid"
    assert 'X_pca' in result.present_keys, "X_pca should be present"
    assert 'X_umap' in result.present_keys, "X_umap should be present"
    assert len(result.missing_keys) == 0, "No keys should be missing"
    print("✓ test_check_obsm_valid passed")


def test_check_obsm_missing():
    """Test obsm validation with missing keys."""
    print("Testing obsm validation (missing case)...")
    adata = create_test_adata()
    adata.obsm['X_pca'] = np.random.rand(100, 50)

    validator = DataValidators(adata)
    result = validator.check_obsm(['X_pca', 'X_umap'])

    assert not result.is_valid, "Should be invalid"
    assert 'X_pca' in result.present_keys, "X_pca should be present"
    assert 'X_umap' in result.missing_keys, "X_umap should be missing"
    print("✓ test_check_obsm_missing passed")


def test_check_obsp_valid():
    """Test obsp validation with valid keys."""
    print("Testing obsp validation (valid case)...")
    adata = create_test_adata()
    adata.obsp['connectivities'] = csr_matrix((100, 100))
    adata.obsp['distances'] = csr_matrix((100, 100))

    validator = DataValidators(adata)
    result = validator.check_obsp(['connectivities', 'distances'])

    assert result.is_valid, "Should be valid"
    assert 'connectivities' in result.present_keys, "connectivities should be present"
    assert 'distances' in result.present_keys, "distances should be present"
    assert result.is_sparse['connectivities'], "connectivities should be sparse"
    print("✓ test_check_obsp_valid passed")


def test_check_obsp_missing():
    """Test obsp validation with missing keys."""
    print("Testing obsp validation (missing case)...")
    adata = create_test_adata()

    validator = DataValidators(adata)
    result = validator.check_obsp(['connectivities', 'distances'])

    assert not result.is_valid, "Should be invalid"
    assert 'connectivities' in result.missing_keys, "connectivities should be missing"
    assert 'distances' in result.missing_keys, "distances should be missing"
    print("✓ test_check_obsp_missing passed")


def test_check_uns_valid():
    """Test uns validation with valid keys."""
    print("Testing uns validation (valid case)...")
    adata = create_test_adata()
    adata.uns['neighbors'] = {'params': {'n_neighbors': 15}}
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    validator = DataValidators(adata)
    result = validator.check_uns(['neighbors', 'pca'])

    assert result.is_valid, "Should be valid"
    assert 'neighbors' in result.present_keys, "neighbors should be present"
    assert 'pca' in result.present_keys, "pca should be present"
    print("✓ test_check_uns_valid passed")


def test_check_uns_missing():
    """Test uns validation with missing keys."""
    print("Testing uns validation (missing case)...")
    adata = create_test_adata()
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    validator = DataValidators(adata)
    result = validator.check_uns(['neighbors', 'pca'])

    assert not result.is_valid, "Should be invalid"
    assert 'neighbors' in result.missing_keys, "neighbors should be missing"
    assert 'pca' in result.present_keys, "pca should be present"
    print("✓ test_check_uns_missing passed")


def test_check_layers_valid():
    """Test layers validation with valid keys."""
    print("Testing layers validation (valid case)...")
    adata = create_test_adata()
    adata.layers['counts'] = adata.X.copy()
    adata.layers['normalized'] = adata.X.copy()

    validator = DataValidators(adata)
    result = validator.check_layers(['counts', 'normalized'])

    assert result.is_valid, "Should be valid"
    assert 'counts' in result.present_keys, "counts should be present"
    assert 'normalized' in result.present_keys, "normalized should be present"
    print("✓ test_check_layers_valid passed")


def test_check_layers_missing():
    """Test layers validation with missing keys."""
    print("Testing layers validation (missing case)...")
    adata = create_test_adata()
    adata.layers['counts'] = adata.X.copy()

    validator = DataValidators(adata)
    result = validator.check_layers(['counts', 'normalized'])

    assert not result.is_valid, "Should be invalid"
    assert 'counts' in result.present_keys, "counts should be present"
    assert 'normalized' in result.missing_keys, "normalized should be missing"
    print("✓ test_check_layers_missing passed")


def test_check_all_requirements():
    """Test comprehensive validation of all requirements."""
    print("Testing comprehensive validation...")
    adata = create_test_adata()
    adata.obsm['X_pca'] = np.random.rand(100, 50)
    adata.obsp['connectivities'] = csr_matrix((100, 100))
    adata.obs['leiden'] = ['0'] * 50 + ['1'] * 50

    validator = DataValidators(adata)

    # All present
    requires = {
        'obs': ['leiden'],
        'obsm': ['X_pca'],
        'obsp': ['connectivities'],
    }
    result = validator.check_all_requirements(requires)
    assert result.is_valid, "All requirements should be satisfied"

    # Some missing
    requires = {
        'obs': ['leiden', 'batch'],
        'obsm': ['X_pca', 'X_umap'],
        'obsp': ['connectivities', 'distances'],
    }
    result = validator.check_all_requirements(requires)
    assert not result.is_valid, "Some requirements should be missing"
    assert 'batch' in result.obs_result.missing_columns, "batch should be missing"
    assert 'X_umap' in result.obsm_result.missing_keys, "X_umap should be missing"
    assert 'distances' in result.obsp_result.missing_keys, "distances should be missing"
    print("✓ test_check_all_requirements passed")


def test_empty_requirements():
    """Test that empty requirements pass validation."""
    print("Testing empty requirements...")
    adata = create_test_adata()
    validator = DataValidators(adata)

    result = validator.check_all_requirements({})
    assert result.is_valid, "Empty requirements should be valid"

    result = validator.check_obs([])
    assert result.is_valid, "Empty obs requirements should be valid"

    result = validator.check_obsm([])
    assert result.is_valid, "Empty obsm requirements should be valid"
    print("✓ test_empty_requirements passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Layer 2 Phase 1 - Standalone Unit Tests")
    print("=" * 60)
    print()

    tests = [
        test_check_obs_valid,
        test_check_obs_missing,
        test_check_obsm_valid,
        test_check_obsm_missing,
        test_check_obsp_valid,
        test_check_obsp_missing,
        test_check_uns_valid,
        test_check_uns_missing,
        test_check_layers_valid,
        test_check_layers_missing,
        test_check_all_requirements,
        test_empty_requirements,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All Layer 2 Phase 1 unit tests PASSED!")
        print("\nValidation Summary:")
        print("   ✓ DataValidators class works correctly")
        print("   ✓ All 5 validator methods (obs, obsm, obsp, uns, layers)")
        print("   ✓ Comprehensive validation (check_all_requirements)")
        print("   ✓ Edge case handling (empty requirements)")
        print("\nPhase 1 Status: ✅ COMPLETE")
        print("\nNext Steps:")
        print("   - Begin Phase 2: PrerequisiteChecker implementation")
        print("   - Detect executed functions via metadata markers")
        print("   - Implement confidence scoring")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
