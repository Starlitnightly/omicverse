"""
Unit tests for PrerequisiteChecker class.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData

from omicverse.utils.inspector.prerequisite_checker import PrerequisiteChecker, DetectionResult


# Mock registry for testing
class MockRegistry:
    """Mock registry with test function metadata."""

    def __init__(self):
        self.functions = {
            'pca': {
                'prerequisites': {'functions': ['preprocess'], 'optional_functions': []},
                'requires': {},
                'produces': {
                    'obsm': ['X_pca'],
                    'uns': ['pca'],
                },
                'auto_fix': 'none',
            },
            'neighbors': {
                'prerequisites': {'functions': ['pca'], 'optional_functions': []},
                'requires': {'obsm': ['X_pca']},
                'produces': {
                    'obsp': ['connectivities', 'distances'],
                    'uns': ['neighbors'],
                },
                'auto_fix': 'none',
            },
            'leiden': {
                'prerequisites': {'functions': ['neighbors'], 'optional_functions': []},
                'requires': {'obsp': ['connectivities', 'distances']},
                'produces': {'obs': ['leiden']},
                'auto_fix': 'none',
            },
            'preprocess': {
                'prerequisites': {'functions': ['qc'], 'optional_functions': []},
                'requires': {},
                'produces': {},
                'auto_fix': 'auto',
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


def create_test_adata():
    """Create a minimal test AnnData object."""
    X = np.random.rand(100, 50)
    obs = pd.DataFrame({'cell_type': ['A'] * 50 + ['B'] * 50}, index=[f'cell_{i}' for i in range(100)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(50)])
    return AnnData(X=X, obs=obs, var=var)


def test_check_function_not_executed():
    """Test detection when function has not been executed."""
    print("Testing function not executed...")
    adata = create_test_adata()
    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('pca')

    # PCA not executed - no markers present
    assert not result.executed or result.confidence < 0.5, "PCA should not be detected as executed"
    print("✓ test_check_function_not_executed passed")


def test_metadata_marker_detection():
    """Test high confidence detection via metadata markers."""
    print("Testing metadata marker detection...")
    adata = create_test_adata()

    # Add PCA metadata marker
    adata.uns['pca'] = {
        'variance_ratio': [0.1, 0.05, 0.03],
        'variance': [10.0, 5.0, 3.0],
    }

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('pca')

    # PCA should be detected with high confidence
    assert result.executed, "PCA should be detected as executed"
    assert result.confidence >= 0.85, f"Confidence should be high (>= 0.85), got {result.confidence}"
    assert result.detection_method in ['metadata_marker', 'multiple_evidence'], \
        f"Should use metadata marker detection, got {result.detection_method}"
    print(f"✓ test_metadata_marker_detection passed (confidence: {result.confidence:.2f})")


def test_output_signature_detection():
    """Test medium confidence detection via output signatures."""
    print("Testing output signature detection...")
    adata = create_test_adata()

    # Add PCA output (obsm) but no metadata
    adata.obsm['X_pca'] = np.random.rand(100, 50)

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('pca')

    # PCA should be detected with medium confidence
    assert result.executed, "PCA should be detected as executed"
    assert 0.70 <= result.confidence < 0.95, \
        f"Confidence should be medium (0.70-0.95), got {result.confidence}"
    print(f"✓ test_output_signature_detection passed (confidence: {result.confidence:.2f})")


def test_multiple_evidence_high_confidence():
    """Test that multiple pieces of evidence increase confidence."""
    print("Testing multiple evidence detection...")
    adata = create_test_adata()

    # Add both metadata marker and output signature
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}
    adata.obsm['X_pca'] = np.random.rand(100, 50)

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('pca')

    # Should have very high confidence with multiple evidence
    assert result.executed, "PCA should be detected as executed"
    assert result.confidence >= 0.85, \
        f"Confidence should be very high with multiple evidence, got {result.confidence}"
    assert len(result.evidence) >= 2, "Should have multiple pieces of evidence"
    print(f"✓ test_multiple_evidence_high_confidence passed (confidence: {result.confidence:.2f}, evidence: {len(result.evidence)})")


def test_neighbors_detection():
    """Test detection of neighbors function."""
    print("Testing neighbors detection...")
    adata = create_test_adata()

    # Add neighbors outputs
    adata.obsp['connectivities'] = csr_matrix((100, 100))
    adata.obsp['distances'] = csr_matrix((100, 100))
    adata.uns['neighbors'] = {
        'params': {'n_neighbors': 15, 'method': 'umap'},
    }

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('neighbors')

    # Neighbors should be detected with high confidence
    assert result.executed, "neighbors should be detected as executed"
    assert result.confidence >= 0.80, \
        f"Confidence should be high, got {result.confidence}"
    print(f"✓ test_neighbors_detection passed (confidence: {result.confidence:.2f})")


def test_check_all_prerequisites():
    """Test checking all prerequisites for a function."""
    print("Testing check_all_prerequisites...")
    adata = create_test_adata()

    # Set up data as if PCA was run
    adata.obsm['X_pca'] = np.random.rand(100, 50)
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    # Check prerequisites for neighbors (which requires PCA)
    results = checker.check_all_prerequisites('neighbors')

    # Should check for 'pca' prerequisite
    assert 'pca' in results, "Should check for pca prerequisite"
    assert results['pca'].executed, "PCA should be detected as executed"
    assert results['pca'].confidence >= 0.70, "PCA detection should have good confidence"
    print(f"✓ test_check_all_prerequisites passed (pca confidence: {results['pca'].confidence:.2f})")


def test_check_all_prerequisites_missing():
    """Test checking prerequisites when they're missing."""
    print("Testing check_all_prerequisites with missing prerequisites...")
    adata = create_test_adata()
    # No PCA outputs added

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    # Check prerequisites for neighbors (which requires PCA)
    results = checker.check_all_prerequisites('neighbors')

    # Should check for 'pca' prerequisite
    assert 'pca' in results, "Should check for pca prerequisite"
    # PCA should not be detected or have low confidence
    assert not results['pca'].executed or results['pca'].confidence < 0.5, \
        "PCA should not be detected as executed"
    print("✓ test_check_all_prerequisites_missing passed")


def test_leiden_detection():
    """Test detection of leiden clustering."""
    print("Testing leiden detection...")
    adata = create_test_adata()

    # Add leiden output
    adata.obs['leiden'] = ['0'] * 50 + ['1'] * 50

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('leiden')

    # Leiden should be detected
    assert result.executed, "leiden should be detected as executed"
    assert result.confidence >= 0.70, \
        f"Confidence should be reasonable, got {result.confidence}"
    print(f"✓ test_leiden_detection passed (confidence: {result.confidence:.2f})")


def test_detection_result_caching():
    """Test that detection results are cached."""
    print("Testing detection result caching...")
    adata = create_test_adata()
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    # First call
    result1 = checker.check_function_executed('pca')

    # Second call should return cached result
    result2 = checker.check_function_executed('pca')

    assert result1 is result2, "Should return same cached object"
    assert 'pca' in checker.detection_results, "Result should be in cache"
    print("✓ test_detection_result_caching passed")


def test_detection_result_string_representation():
    """Test DetectionResult string representation."""
    print("Testing DetectionResult __str__...")
    adata = create_test_adata()
    adata.uns['pca'] = {'variance_ratio': [0.1, 0.05]}

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('pca')

    # Should produce readable string
    str_repr = str(result)
    assert 'pca' in str_repr, "Function name should be in string"
    assert 'confidence' in str_repr.lower(), "Confidence should be mentioned"
    print(f"✓ test_detection_result_string_representation passed: {str_repr}")


def test_unknown_function():
    """Test handling of unknown function."""
    print("Testing unknown function handling...")
    adata = create_test_adata()
    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    result = checker.check_function_executed('nonexistent_function')

    # Should return result with executed=False, confidence=0.0
    assert not result.executed, "Unknown function should not be detected as executed"
    assert result.confidence == 0.0, "Confidence should be 0.0"
    assert result.detection_method == 'unknown', "Detection method should be 'unknown'"
    print("✓ test_unknown_function passed")


def test_nested_uns_key():
    """Test detection with nested uns keys."""
    print("Testing nested uns key detection...")
    adata = create_test_adata()

    # Add nested structure
    adata.uns['neighbors'] = {
        'params': {
            'n_neighbors': 15,
            'method': 'umap',
        }
    }

    registry = MockRegistry()
    checker = PrerequisiteChecker(adata, registry)

    # Test _check_uns_key_exists with nested key
    assert checker._check_uns_key_exists('neighbors'), "Should find 'neighbors'"
    assert checker._check_uns_key_exists('neighbors.params'), "Should find nested 'neighbors.params'"
    assert checker._check_uns_key_exists('neighbors.params.n_neighbors'), "Should find deeply nested key"
    assert not checker._check_uns_key_exists('neighbors.params.nonexistent'), "Should not find nonexistent nested key"
    print("✓ test_nested_uns_key passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PrerequisiteChecker - Unit Tests")
    print("=" * 60)
    print()

    tests = [
        test_check_function_not_executed,
        test_metadata_marker_detection,
        test_output_signature_detection,
        test_multiple_evidence_high_confidence,
        test_neighbors_detection,
        test_check_all_prerequisites,
        test_check_all_prerequisites_missing,
        test_leiden_detection,
        test_detection_result_caching,
        test_detection_result_string_representation,
        test_unknown_function,
        test_nested_uns_key,
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
        print("\n✅ All PrerequisiteChecker tests PASSED!")
        print("\nValidation Summary:")
        print("   ✓ Metadata marker detection (HIGH confidence)")
        print("   ✓ Output signature detection (MEDIUM confidence)")
        print("   ✓ Multiple evidence aggregation")
        print("   ✓ Prerequisite chain checking")
        print("   ✓ Result caching")
        print("   ✓ Edge case handling")
        print("\nPhase 2 Status: Core functionality complete")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
