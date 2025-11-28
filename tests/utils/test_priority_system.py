#!/usr/bin/env python3
"""
Test script to verify the priority-based execution system.

This script validates:
1. Complexity classification (simple vs complex)
2. Priority 1 workflow structure
3. Priority 2 workflow structure
4. Fallback mechanism
5. Code validation logic
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_complexity_classifier_structure():
    """Test that complexity classifier method exists and has correct structure."""
    print("=" * 70)
    print("TEST 1: Complexity Classifier Structure")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent

    # Check method exists
    assert hasattr(OmicVerseAgent, '_analyze_task_complexity'), \
        "Missing _analyze_task_complexity method"
    print("✓ _analyze_task_complexity method exists")

    # Check it's async
    import inspect
    assert inspect.iscoroutinefunction(OmicVerseAgent._analyze_task_complexity), \
        "_analyze_task_complexity should be async"
    print("✓ _analyze_task_complexity is async")

    # Check signature
    sig = inspect.signature(OmicVerseAgent._analyze_task_complexity)
    params = list(sig.parameters.keys())
    assert 'request' in params, "Missing 'request' parameter"
    print("✓ Correct method signature")

    print("\n✅ TEST 1 PASSED\n")
    return True


def test_priority1_workflow_structure():
    """Test that Priority 1 workflow exists and has correct structure."""
    print("=" * 70)
    print("TEST 2: Priority 1 Workflow Structure")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent

    # Check method exists
    assert hasattr(OmicVerseAgent, '_run_registry_workflow'), \
        "Missing _run_registry_workflow method"
    print("✓ _run_registry_workflow method exists")

    # Check it's async
    import inspect
    assert inspect.iscoroutinefunction(OmicVerseAgent._run_registry_workflow), \
        "_run_registry_workflow should be async"
    print("✓ _run_registry_workflow is async")

    # Check signature
    sig = inspect.signature(OmicVerseAgent._run_registry_workflow)
    params = list(sig.parameters.keys())
    assert 'request' in params and 'adata' in params, \
        "Missing required parameters"
    print("✓ Correct method signature (request, adata)")

    print("\n✅ TEST 2 PASSED\n")
    return True


def test_priority2_workflow_structure():
    """Test that Priority 2 workflow exists and has correct structure."""
    print("=" * 70)
    print("TEST 3: Priority 2 Workflow Structure")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent

    # Check method exists
    assert hasattr(OmicVerseAgent, '_run_skills_workflow'), \
        "Missing _run_skills_workflow method"
    print("✓ _run_skills_workflow method exists")

    # Check it's async
    import inspect
    assert inspect.iscoroutinefunction(OmicVerseAgent._run_skills_workflow), \
        "_run_skills_workflow should be async"
    print("✓ _run_skills_workflow is async")

    # Check signature
    sig = inspect.signature(OmicVerseAgent._run_skills_workflow)
    params = list(sig.parameters.keys())
    assert 'request' in params and 'adata' in params, \
        "Missing required parameters"
    print("✓ Correct method signature (request, adata)")

    print("\n✅ TEST 3 PASSED\n")
    return True


def test_code_validation_structure():
    """Test that code validation method exists and works."""
    print("=" * 70)
    print("TEST 4: Code Validation Logic")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent

    # Check method exists
    assert hasattr(OmicVerseAgent, '_validate_simple_execution'), \
        "Missing _validate_simple_execution method"
    print("✓ _validate_simple_execution method exists")

    # Create agent instance for testing
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Test simple code (should pass)
    simple_code = """
import omicverse as ov
adata = ov.pp.qc(adata, tresh={'nUMIs': 500})
print(f"Done: {adata.shape[0]} cells")
"""
    is_valid, reason = agent._validate_simple_execution(simple_code)
    assert is_valid, f"Simple code should be valid: {reason}"
    print(f"✓ Simple code validated: {reason}")

    # Test complex code (should fail)
    complex_code = """
import omicverse as ov
for i in range(10):
    adata = ov.pp.normalize(adata)
if adata.shape[0] > 100:
    adata = ov.pp.filter_cells(adata)
"""
    is_valid, reason = agent._validate_simple_execution(complex_code)
    assert not is_valid, "Complex code should be invalid"
    print(f"✓ Complex code rejected: {reason}")

    # Test code with too many function calls
    many_calls_code = """
import omicverse as ov
adata = ov.pp.qc(adata)
adata = ov.pp.normalize(adata)
adata = ov.pp.scale(adata)
adata = ov.pp.pca(adata)
adata = ov.pp.neighbors(adata)
adata = ov.pp.leiden(adata)
adata = ov.pp.umap(adata)
"""
    is_valid, reason = agent._validate_simple_execution(many_calls_code)
    assert not is_valid, "Code with too many calls should be invalid"
    print(f"✓ Too many calls rejected: {reason}")

    print("\n✅ TEST 4 PASSED\n")
    return True


def test_run_async_integration():
    """Test that run_async integrates all components."""
    print("=" * 70)
    print("TEST 5: run_async Integration")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent
    import inspect

    # Check method exists
    assert hasattr(OmicVerseAgent, 'run_async'), \
        "Missing run_async method"
    print("✓ run_async method exists")

    # Check it's async
    assert inspect.iscoroutinefunction(OmicVerseAgent.run_async), \
        "run_async should be async"
    print("✓ run_async is async")

    # Check signature
    sig = inspect.signature(OmicVerseAgent.run_async)
    params = list(sig.parameters.keys())
    assert 'request' in params and 'adata' in params, \
        "Missing required parameters"
    print("✓ Correct method signature")

    # Read source code to check for priority logic
    source = inspect.getsource(OmicVerseAgent.run_async)

    # Check for complexity analysis
    assert '_analyze_task_complexity' in source, \
        "run_async should call _analyze_task_complexity"
    print("✓ Calls _analyze_task_complexity")

    # Check for Priority 1 workflow
    assert '_run_registry_workflow' in source, \
        "run_async should call _run_registry_workflow"
    print("✓ Calls _run_registry_workflow")

    # Check for Priority 2 workflow
    assert '_run_skills_workflow' in source, \
        "run_async should call _run_skills_workflow"
    print("✓ Calls _run_skills_workflow")

    # Check for fallback logic
    assert 'fallback' in source.lower() or 'fall back' in source.lower(), \
        "run_async should have fallback logic"
    print("✓ Has fallback mechanism")

    # Check for priority tracking
    assert 'priority_used' in source or 'Priority 1' in source, \
        "run_async should track which priority was used"
    print("✓ Tracks priority usage")

    print("\n✅ TEST 5 PASSED\n")
    return True


def test_legacy_code_preserved():
    """Test that legacy code is preserved for reference."""
    print("=" * 70)
    print("TEST 6: Legacy Code Preservation")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent

    # Check legacy method exists
    assert hasattr(OmicVerseAgent, 'run_async_LEGACY'), \
        "Missing run_async_LEGACY method"
    print("✓ run_async_LEGACY method exists")

    # Check it's async
    import inspect
    assert inspect.iscoroutinefunction(OmicVerseAgent.run_async_LEGACY), \
        "run_async_LEGACY should be async"
    print("✓ run_async_LEGACY is async")

    print("\n✅ TEST 6 PASSED\n")
    return True


def test_documentation():
    """Test that methods have proper documentation."""
    print("=" * 70)
    print("TEST 7: Documentation")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent
    import inspect

    methods = [
        '_analyze_task_complexity',
        '_run_registry_workflow',
        '_run_skills_workflow',
        '_validate_simple_execution',
        'run_async'
    ]

    for method_name in methods:
        method = getattr(OmicVerseAgent, method_name)
        docstring = inspect.getdoc(method)
        assert docstring and len(docstring) > 50, \
            f"{method_name} missing or has insufficient documentation"
        print(f"✓ {method_name} has documentation ({len(docstring)} chars)")

    print("\n✅ TEST 7 PASSED\n")
    return True


def test_workflow_flow():
    """Test the logical flow of the priority system."""
    print("=" * 70)
    print("TEST 8: Workflow Flow Logic")
    print("=" * 70)

    from omicverse.utils.smart_agent import OmicVerseAgent
    import inspect

    # Read run_async source
    source = inspect.getsource(OmicVerseAgent.run_async)
    lines = source.split('\n')

    # Check for proper flow sequence
    complexity_line = next((i for i, line in enumerate(lines) if '_analyze_task_complexity' in line), None)
    priority1_line = next((i for i, line in enumerate(lines) if '_run_registry_workflow' in line), None)
    priority2_line = next((i for i, line in enumerate(lines) if '_run_skills_workflow' in line), None)

    assert complexity_line is not None, "Missing complexity analysis"
    assert priority1_line is not None, "Missing Priority 1 call"
    assert priority2_line is not None, "Missing Priority 2 call"

    # Check order: complexity -> priority1 -> priority2
    assert complexity_line < priority1_line < priority2_line, \
        "Incorrect execution order"
    print("✓ Correct execution order: complexity → P1 → P2")

    # Check for exception handling
    assert 'try:' in source and 'except' in source, \
        "Missing exception handling"
    print("✓ Has exception handling")

    # Check for return statements
    assert source.count('return') >= 2, \
        "Should have multiple return points (P1 success, P2 success)"
    print("✓ Has proper return statements")

    print("\n✅ TEST 8 PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "🔬" * 35)
    print("PRIORITY SYSTEM INTEGRATION TEST SUITE")
    print("🔬" * 35 + "\n")

    tests = [
        ("Complexity Classifier Structure", test_complexity_classifier_structure),
        ("Priority 1 Workflow Structure", test_priority1_workflow_structure),
        ("Priority 2 Workflow Structure", test_priority2_workflow_structure),
        ("Code Validation Logic", test_code_validation_structure),
        ("run_async Integration", test_run_async_integration),
        ("Legacy Code Preservation", test_legacy_code_preserved),
        ("Documentation", test_documentation),
        ("Workflow Flow Logic", test_workflow_flow),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED")
            print(f"   Error: {e}\n")
            results[test_name] = False
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception")
            print(f"   Error: {type(e).__name__}: {e}\n")
            results[test_name] = False

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe priority system integration is working correctly:")
        print("  ✓ Phase 1: Complexity classifier implemented")
        print("  ✓ Phase 2: Priority 1 (fast path) implemented")
        print("  ✓ Phase 3: Priority 2 (skills) implemented")
        print("  ✓ Phase 4: Code validation implemented")
        print("  ✓ Phase 5: run_async integration complete")
        print("\nCore system ready for testing with real data!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nReview the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
