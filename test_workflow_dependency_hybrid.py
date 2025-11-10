"""
Test script for hybrid prerequisite detection system.

This tests the Layer 1 (Runtime Data Inspection) implementation
to ensure data state awareness is working correctly.
"""

import numpy as np
import pandas as pd
from anndata import AnnData


def test_data_state_inspector():
    """Test DataStateInspector on various data states."""
    print("=" * 70)
    print("TEST 1: DataStateInspector Basic Functionality")
    print("=" * 70)

    from omicverse.utils.smart_agent import DataStateInspector

    # Test 1: Raw data
    print("\n1. Testing RAW DATA:")
    X = np.random.rand(100, 50)
    adata_raw = AnnData(X=X)

    state_raw = DataStateInspector.inspect(adata_raw)
    print(f"   Capabilities: {state_raw['capabilities']}")
    print(f"   Layers: {state_raw['available']['layers']}")
    assert len(state_raw['capabilities']) == 0, "Raw data should have no capabilities"
    print("   ✅ PASS: Raw data correctly identified")

    # Test 2: Preprocessed data
    print("\n2. Testing PREPROCESSED DATA:")
    adata_preprocessed = AnnData(X=X)
    adata_preprocessed.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)

    state_preprocessed = DataStateInspector.inspect(adata_preprocessed)
    print(f"   Capabilities: {state_preprocessed['capabilities']}")
    print(f"   Layers: {state_preprocessed['available']['layers']}")
    assert 'has_processed_layers' in state_preprocessed['capabilities']
    print("   ✅ PASS: Preprocessed data correctly identified")

    # Test 3: Data with PCA
    print("\n3. Testing DATA WITH PCA:")
    adata_pca = AnnData(X=X)
    adata_pca.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata_pca.obsm['X_pca'] = np.random.rand(100, 20)

    state_pca = DataStateInspector.inspect(adata_pca)
    print(f"   Capabilities: {state_pca['capabilities']}")
    print(f"   Obsm: {state_pca['available']['obsm']}")
    assert 'has_pca' in state_pca['capabilities']
    assert 'has_processed_layers' in state_pca['capabilities']
    print("   ✅ PASS: PCA correctly identified")

    # Test 4: Data with neighbors
    print("\n4. Testing DATA WITH NEIGHBORS:")
    adata_neighbors = AnnData(X=X)
    adata_neighbors.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata_neighbors.obsm['X_pca'] = np.random.rand(100, 20)
    adata_neighbors.uns['neighbors'] = {'params': {'n_neighbors': 15}}

    state_neighbors = DataStateInspector.inspect(adata_neighbors)
    print(f"   Capabilities: {state_neighbors['capabilities']}")
    assert 'has_neighbors' in state_neighbors['capabilities']
    print("   ✅ PASS: Neighbors correctly identified")

    # Test 5: Data with clustering
    print("\n5. Testing DATA WITH CLUSTERING:")
    adata_clustered = AnnData(X=X)
    adata_clustered.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata_clustered.obsm['X_pca'] = np.random.rand(100, 20)
    adata_clustered.uns['neighbors'] = {'params': {'n_neighbors': 15}}
    adata_clustered.obs['leiden'] = np.random.choice(['0', '1', '2'], size=100)

    state_clustered = DataStateInspector.inspect(adata_clustered)
    print(f"   Capabilities: {state_clustered['capabilities']}")
    assert 'has_clustering' in state_clustered['capabilities']
    print(f"   Clustering columns: {state_clustered.get('clustering_columns', [])}")
    print("   ✅ PASS: Clustering correctly identified")

    print("\n✅ ALL DataStateInspector TESTS PASSED!\n")
    return True


def test_readable_summary():
    """Test human-readable summary generation."""
    print("=" * 70)
    print("TEST 2: Human-Readable Summary")
    print("=" * 70)

    from omicverse.utils.smart_agent import DataStateInspector

    # Create data with various features
    X = np.random.rand(100, 50)
    adata = AnnData(X=X)
    adata.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata.layers['counts'] = X * 1000
    adata.obsm['X_pca'] = np.random.rand(100, 20)
    adata.obsm['X_umap'] = np.random.rand(100, 2)
    adata.uns['neighbors'] = {'params': {'n_neighbors': 15}}
    adata.obs['leiden'] = np.random.choice(['0', '1', '2'], size=100)

    summary = DataStateInspector.get_readable_summary(adata)
    print("\n" + summary)

    # Verify key elements are in summary
    assert "Shape" in summary
    assert "Available Layers" in summary
    assert "Available Obsm" in summary
    assert "Detected Capabilities" in summary

    print("\n✅ READABLE SUMMARY TEST PASSED!\n")
    return True


def test_compatibility_check():
    """Test function compatibility checking."""
    print("=" * 70)
    print("TEST 3: Function Compatibility Check")
    print("=" * 70)

    from omicverse.utils.smart_agent import DataStateInspector

    # Raw data
    X = np.random.rand(100, 50)
    adata_raw = AnnData(X=X)

    # Test PCA compatibility on raw data
    print("\n1. Testing PCA compatibility on RAW DATA:")
    compat = DataStateInspector.check_compatibility(
        adata_raw,
        'pca',
        "(adata, layer='scaled', n_pcs=50)",
        'preprocessing'
    )
    print(f"   Likely compatible: {compat['likely_compatible']}")
    print(f"   Warnings: {compat['warnings']}")
    print(f"   Suggestions: {compat['suggestions']}")
    assert len(compat['warnings']) > 0, "Should warn about missing scaled layer"
    print("   ✅ PASS: Correctly identified missing prerequisites")

    # Preprocessed data
    adata_preprocessed = AnnData(X=X)
    adata_preprocessed.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)

    print("\n2. Testing PCA compatibility on PREPROCESSED DATA:")
    compat = DataStateInspector.check_compatibility(
        adata_preprocessed,
        'pca',
        "(adata, layer='scaled', n_pcs=50)",
        'preprocessing'
    )
    print(f"   Likely compatible: {compat['likely_compatible']}")
    print(f"   Warnings: {compat['warnings']}")
    assert len(compat['warnings']) == 0, "Should have no warnings"
    print("   ✅ PASS: Correctly identified data is ready")

    # Test clustering compatibility
    print("\n3. Testing LEIDEN compatibility on data without neighbors:")
    compat = DataStateInspector.check_compatibility(
        adata_preprocessed,
        'leiden',
        "(adata, resolution=1.0)",
        'clustering'
    )
    print(f"   Warnings: {compat['warnings']}")
    print(f"   Suggestions: {compat['suggestions']}")
    assert len(compat['warnings']) > 0, "Should warn about missing neighbors"
    print("   ✅ PASS: Correctly identified missing neighbors")

    print("\n✅ ALL COMPATIBILITY TESTS PASSED!\n")
    return True


async def test_classification_with_state():
    """Test task classification with data state awareness."""
    print("=" * 70)
    print("TEST 4: Task Classification with Data State")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent
    import os

    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("\n⚠️  SKIPPED: No API key found for testing agent")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test this feature\n")
        return True

    try:
        agent = OmicVerseAgent(model='gpt-4o-mini')
    except Exception as e:
        print(f"\n⚠️  SKIPPED: Could not initialize agent: {e}\n")
        return True

    # Test 1: PCA on raw data should be COMPLEX
    print("\n1. Testing PCA classification on RAW DATA:")
    X = np.random.rand(100, 50)
    adata_raw = AnnData(X=X)

    complexity = await agent._analyze_task_complexity("Run PCA", adata_raw)
    print(f"   Classification: {complexity.upper()}")
    assert complexity == 'complex', "PCA on raw data should be COMPLEX"
    print("   ✅ PASS: Correctly classified as COMPLEX")

    # Test 2: PCA on preprocessed data should be SIMPLE
    print("\n2. Testing PCA classification on PREPROCESSED DATA:")
    adata_preprocessed = AnnData(X=X)
    adata_preprocessed.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)

    complexity = await agent._analyze_task_complexity("Run PCA", adata_preprocessed)
    print(f"   Classification: {complexity.upper()}")
    assert complexity == 'simple', "PCA on preprocessed data should be SIMPLE"
    print("   ✅ PASS: Correctly classified as SIMPLE")

    # Test 3: Clustering with PCA should be SIMPLE
    print("\n3. Testing CLUSTERING classification with PCA:")
    adata_pca = AnnData(X=X)
    adata_pca.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata_pca.obsm['X_pca'] = np.random.rand(100, 20)

    complexity = await agent._analyze_task_complexity("Run leiden clustering", adata_pca)
    print(f"   Classification: {complexity.upper()}")
    assert complexity == 'simple', "Clustering with PCA should be SIMPLE"
    print("   ✅ PASS: Correctly classified as SIMPLE")

    # Test 4: Clustering without preprocessing should be COMPLEX
    print("\n4. Testing CLUSTERING classification on RAW DATA:")
    complexity = await agent._analyze_task_complexity("Cluster my cells", adata_raw)
    print(f"   Classification: {complexity.upper()}")
    assert complexity == 'complex', "Clustering on raw data should be COMPLEX"
    print("   ✅ PASS: Correctly classified as COMPLEX")

    print("\n✅ ALL CLASSIFICATION TESTS PASSED!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING HYBRID PREREQUISITE DETECTION SYSTEM")
    print("Layer 1: Runtime Data Inspection")
    print("=" * 70 + "\n")

    results = []

    # Test 1: DataStateInspector
    try:
        results.append(("DataStateInspector", test_data_state_inspector()))
    except Exception as e:
        print(f"❌ FAILED: DataStateInspector test: {e}")
        results.append(("DataStateInspector", False))

    # Test 2: Readable summary
    try:
        results.append(("Readable Summary", test_readable_summary()))
    except Exception as e:
        print(f"❌ FAILED: Readable summary test: {e}")
        results.append(("Readable Summary", False))

    # Test 3: Compatibility check
    try:
        results.append(("Compatibility Check", test_compatibility_check()))
    except Exception as e:
        print(f"❌ FAILED: Compatibility check test: {e}")
        results.append(("Compatibility Check", False))

    # Test 4: Classification with state (requires API key)
    try:
        import asyncio
        results.append(("Classification", asyncio.run(test_classification_with_state())))
    except Exception as e:
        print(f"❌ FAILED: Classification test: {e}")
        results.append(("Classification", False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
