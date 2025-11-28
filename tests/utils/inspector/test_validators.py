"""
Unit tests for DataValidators class.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData

from omicverse.utils.inspector.validators import DataValidators


def create_test_adata():
    """Create a minimal test AnnData object."""
    X = np.random.rand(100, 50)
    obs = pd.DataFrame({'cell_type': ['A'] * 50 + ['B'] * 50}, index=[f'cell_{i}' for i in range(100)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(50)])
    return AnnData(X=X, obs=obs, var=var)


def test_check_obs_valid():
    """Test obs validation with valid columns."""
    adata = create_test_adata()
    validator = DataValidators(adata)

    result = validator.check_obs(['cell_type'])

    assert result.is_valid
    assert 'cell_type' in result.present_columns
    assert len(result.missing_columns) == 0


def test_check_obs_missing():
    """Test obs validation with missing columns."""
    adata = create_test_adata()
    validator = DataValidators(adata)

    result = validator.check_obs(['cell_type', 'leiden', 'batch'])

    assert not result.is_valid
    assert 'cell_type' in result.present_columns
    assert 'leiden' in result.missing_columns
    assert 'batch' in result.missing_columns


def test_check_obsm_valid():
    """Test obsm validation with valid keys."""
    adata = create_test_adata()
    adata.obsm['X_pca'] = np.random.rand(100, 50)
    adata.obsm['X_umap'] = np.random.rand(100, 2)

    validator = DataValidators(adata)
    result = validator.check_obsm(['X_pca', 'X_umap'])

    assert result.is_valid
    assert 'X_pca' in result.present_keys
    assert 'X_umap' in result.present_keys
    assert len(result.missing_keys) == 0


def test_check_obsm_missing():
    """Test obsm validation with missing keys."""
    adata = create_test_adata()
    adata.obsm['X_pca'] = np.random.rand(100, 50)

    validator = DataValidators(adata)
    result = validator.check_obsm(['X_pca', 'X_umap'])

    assert not result.is_valid
    assert 'X_pca' in result.present_keys
    assert 'X_umap' in result.missing_keys


def test_check_obsp_valid():
    """Test obsp validation with valid keys."""
    adata = create_test_adata()
    adata.obsp['connectivities'] = csr_matrix((100, 100))
    adata.obsp['distances'] = csr_matrix((100, 100))

    validator = DataValidators(adata)
    result = validator.check_obsp(['connectivities', 'distances'])

    assert result.is_valid
    assert 'connectivities' in result.present_keys
    assert 'distances' in result.present_keys
    assert result.is_sparse['connectivities']


def test_check_obsp_missing():
    """Test obsp validation with missing keys."""
    adata = create_test_adata()

    validator = DataValidators(adata)
    result = validator.check_obsp(['connectivities', 'distances'])

    assert not result.is_valid
    assert 'connectivities' in result.missing_keys
    assert 'distances' in result.missing_keys


def test_check_uns_valid():
    """Test uns validation with valid keys."""
    adata = create_test_adata()
    adata.uns['neighbors'] = {'params': {'n_neighbors': 15}}
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    validator = DataValidators(adata)
    result = validator.check_uns(['neighbors', 'pca'])

    assert result.is_valid
    assert 'neighbors' in result.present_keys
    assert 'pca' in result.present_keys


def test_check_uns_missing():
    """Test uns validation with missing keys."""
    adata = create_test_adata()
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    validator = DataValidators(adata)
    result = validator.check_uns(['neighbors', 'pca'])

    assert not result.is_valid
    assert 'neighbors' in result.missing_keys
    assert 'pca' in result.present_keys


def test_check_layers_valid():
    """Test layers validation with valid keys."""
    adata = create_test_adata()
    adata.layers['counts'] = adata.X.copy()
    adata.layers['normalized'] = adata.X.copy()

    validator = DataValidators(adata)
    result = validator.check_layers(['counts', 'normalized'])

    assert result.is_valid
    assert 'counts' in result.present_keys
    assert 'normalized' in result.present_keys


def test_check_layers_missing():
    """Test layers validation with missing keys."""
    adata = create_test_adata()
    adata.layers['counts'] = adata.X.copy()

    validator = DataValidators(adata)
    result = validator.check_layers(['counts', 'normalized'])

    assert not result.is_valid
    assert 'counts' in result.present_keys
    assert 'normalized' in result.missing_keys


def test_check_all_requirements():
    """Test comprehensive validation of all requirements."""
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
    assert result.is_valid

    # Some missing
    requires = {
        'obs': ['leiden', 'batch'],
        'obsm': ['X_pca', 'X_umap'],
        'obsp': ['connectivities', 'distances'],
    }
    result = validator.check_all_requirements(requires)
    assert not result.is_valid
    assert 'batch' in result.obs_result.missing_columns
    assert 'X_umap' in result.obsm_result.missing_keys
    assert 'distances' in result.obsp_result.missing_keys


def test_empty_requirements():
    """Test that empty requirements pass validation."""
    adata = create_test_adata()
    validator = DataValidators(adata)

    result = validator.check_all_requirements({})
    assert result.is_valid

    result = validator.check_obs([])
    assert result.is_valid

    result = validator.check_obsm([])
    assert result.is_valid


if __name__ == '__main__':
    # Run tests
    print("Running DataValidators tests...")

    test_check_obs_valid()
    print("✓ test_check_obs_valid")

    test_check_obs_missing()
    print("✓ test_check_obs_missing")

    test_check_obsm_valid()
    print("✓ test_check_obsm_valid")

    test_check_obsm_missing()
    print("✓ test_check_obsm_missing")

    test_check_obsp_valid()
    print("✓ test_check_obsp_valid")

    test_check_obsp_missing()
    print("✓ test_check_obsp_missing")

    test_check_uns_valid()
    print("✓ test_check_uns_valid")

    test_check_uns_missing()
    print("✓ test_check_uns_missing")

    test_check_layers_valid()
    print("✓ test_check_layers_valid")

    test_check_layers_missing()
    print("✓ test_check_layers_missing")

    test_check_all_requirements()
    print("✓ test_check_all_requirements")

    test_empty_requirements()
    print("✓ test_empty_requirements")

    print("\n✅ All tests passed!")
