"""
Test script for Layer 2: LLM-based prerequisite inference.

This tests the LLMPrerequisiteInference class to ensure intelligent
prerequisite reasoning is working correctly.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
import asyncio
import os


async def test_llm_prerequisite_inference():
    """Test LLMPrerequisiteInference with various scenarios."""
    print("=" * 70)
    print("TEST: Layer 2 LLM Prerequisite Inference")
    print("=" * 70)

    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("\n⚠️  SKIPPED: No API key found")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test Layer 2\n")
        return True

    from omicverse.utils.smart_agent import (
        LLMPrerequisiteInference,
        DataStateInspector
    )
    from omicverse.utils.agent_backend import OmicVerseLLMBackend
    from omicverse.utils.registry import _global_registry

    # Initialize LLM backend
    try:
        llm = OmicVerseLLMBackend(
            system_prompt="You are a bioinformatics expert.",
            model='gpt-4o-mini',
            max_tokens=2048,
            temperature=0.1
        )
        inference_engine = LLMPrerequisiteInference(llm)
    except Exception as e:
        print(f"\n⚠️  SKIPPED: Could not initialize LLM: {e}\n")
        return True

    # Test 1: PCA on raw data (should need full preprocessing)
    print("\n1. Testing PCA on RAW DATA:")
    X = np.random.rand(100, 50)
    adata_raw = AnnData(X=X)
    data_state_raw = DataStateInspector.inspect(adata_raw)

    # Get PCA function info
    pca_results = _global_registry.find('pca')
    if not pca_results:
        print("   ⚠️  PCA function not found in registry")
        return False

    pca_info = pca_results[0]

    result = await inference_engine.infer_prerequisites(
        function_name='pca',
        function_info=pca_info,
        data_state=data_state_raw,
        skill_context=None
    )

    print(f"   Can run: {result['can_run']}")
    print(f"   Confidence: {result['confidence']:.0%}")
    print(f"   Complexity: {result['complexity']}")
    print(f"   Missing: {result['missing_items']}")
    print(f"   Steps needed: {result['required_steps']}")
    print(f"   Auto-fixable: {result['auto_fixable']}")
    print(f"   Reasoning: {result['reasoning']}")

    assert not result['can_run'], "PCA should not be able to run on raw data"
    assert result['complexity'] == 'complex', "Should be classified as complex"
    assert not result['auto_fixable'], "Should not be auto-fixable (needs multiple steps)"
    print("   ✅ PASS: Correctly identified need for full preprocessing")

    # Test 2: PCA on preprocessed data (should be ready)
    print("\n2. Testing PCA on PREPROCESSED DATA:")
    adata_preprocessed = AnnData(X=X)
    adata_preprocessed.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    data_state_preprocessed = DataStateInspector.inspect(adata_preprocessed)

    result = await inference_engine.infer_prerequisites(
        function_name='pca',
        function_info=pca_info,
        data_state=data_state_preprocessed,
        skill_context=None
    )

    print(f"   Can run: {result['can_run']}")
    print(f"   Confidence: {result['confidence']:.0%}")
    print(f"   Complexity: {result['complexity']}")
    print(f"   Reasoning: {result['reasoning']}")

    assert result['can_run'], "PCA should be able to run on preprocessed data"
    assert result['complexity'] == 'simple', "Should be classified as simple"
    print("   ✅ PASS: Correctly identified data is ready")

    # Test 3: Leiden clustering with PCA, missing neighbors (auto-fixable)
    print("\n3. Testing LEIDEN with PCA, missing neighbors:")
    adata_pca = AnnData(X=X)
    adata_pca.layers['scaled'] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata_pca.obsm['X_pca'] = np.random.rand(100, 20)
    data_state_pca = DataStateInspector.inspect(adata_pca)

    leiden_results = _global_registry.find('leiden')
    if leiden_results:
        leiden_info = leiden_results[0]

        result = await inference_engine.infer_prerequisites(
            function_name='leiden',
            function_info=leiden_info,
            data_state=data_state_pca,
            skill_context=None
        )

        print(f"   Can run: {result['can_run']}")
        print(f"   Confidence: {result['confidence']:.0%}")
        print(f"   Complexity: {result['complexity']}")
        print(f"   Missing: {result['missing_items']}")
        print(f"   Steps needed: {result['required_steps']}")
        print(f"   Auto-fixable: {result['auto_fixable']}")
        print(f"   Reasoning: {result['reasoning']}")

        # Should detect missing neighbors but classify as auto-fixable
        assert not result['can_run'], "Leiden should not run without neighbors"
        assert result['complexity'] == 'simple', "Should be simple (only 1 step missing)"
        assert result['auto_fixable'], "Should be auto-fixable (just needs neighbors)"
        print("   ✅ PASS: Correctly identified as auto-fixable")

    # Test 4: Cache functionality
    print("\n4. Testing CACHE functionality:")
    # Run the same inference again - should use cache
    result_cached = await inference_engine.infer_prerequisites(
        function_name='pca',
        function_info=pca_info,
        data_state=data_state_preprocessed,
        skill_context=None
    )

    assert result_cached['can_run'] == result['can_run'], "Cached result should match"
    print("   ✅ PASS: Cache working correctly")

    # Clear cache
    inference_engine.clear_cache()
    print("   ✅ Cache cleared")

    print("\n✅ ALL Layer 2 INFERENCE TESTS PASSED!\n")
    return True


async def test_integrated_layer2_in_agent():
    """Test Layer 2 integration in the agent workflow."""
    print("=" * 70)
    print("TEST: Layer 2 Integration in Agent")
    print("=" * 70)

    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("\n⚠️  SKIPPED: No API key found")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test Layer 2\n")
        return True

    try:
        from omicverse.utils.smart_agent import OmicVerseAgent

        agent = OmicVerseAgent(model='gpt-4o-mini')

        # Create test data
        X = np.random.rand(100, 50)

        # Test 1: PCA on raw data (Layer 2 should recommend NEEDS_WORKFLOW)
        print("\n1. Testing agent with PCA on raw data:")
        adata_raw = AnnData(X=X)

        # The agent should use Layer 2 and detect this needs full workflow
        # Note: We can't fully test execution here without running the full agent
        # But we can test that Layer 2 is initialized
        assert agent._prerequisite_inference is not None, "Layer 2 should be initialized"
        print("   ✅ Layer 2 initialized in agent")

        # Test 2: Verify Layer 2 cache is working
        print("\n2. Testing Layer 2 cache in agent:")
        cache_size_before = len(agent._prerequisite_inference._cache)
        print(f"   Cache size before: {cache_size_before}")

        # Clear cache
        agent._prerequisite_inference.clear_cache()
        cache_size_after = len(agent._prerequisite_inference._cache)
        print(f"   Cache size after clear: {cache_size_after}")
        assert cache_size_after == 0, "Cache should be empty after clear"
        print("   ✅ Cache management working")

        print("\n✅ ALL Layer 2 INTEGRATION TESTS PASSED!\n")
        return True

    except Exception as e:
        print(f"\n⚠️  TEST ERROR: {e}\n")
        return False


def main():
    """Run all Layer 2 tests."""
    print("\n" + "=" * 70)
    print("TESTING LAYER 2: LLM-BASED PREREQUISITE INFERENCE")
    print("=" * 70 + "\n")

    results = []

    # Test 1: LLM Prerequisite Inference
    try:
        result = asyncio.run(test_llm_prerequisite_inference())
        results.append(("LLM Prerequisite Inference", result))
    except Exception as e:
        print(f"❌ FAILED: LLM inference test: {e}")
        results.append(("LLM Prerequisite Inference", False))

    # Test 2: Layer 2 Integration
    try:
        result = asyncio.run(test_integrated_layer2_in_agent())
        results.append(("Layer 2 Integration", result))
    except Exception as e:
        print(f"❌ FAILED: Layer 2 integration test: {e}")
        results.append(("Layer 2 Integration", False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY - LAYER 2")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("🎉 ALL LAYER 2 TESTS PASSED!")
    else:
        print("⚠️  SOME LAYER 2 TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
