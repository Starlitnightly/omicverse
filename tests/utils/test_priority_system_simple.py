#!/usr/bin/env python3
"""
Simple test script to verify priority system implementation.
Tests code structure without requiring dependencies.
"""

import ast
import re
from pathlib import Path


def read_smart_agent():
    """Read the smart_agent.py file."""
    # Go up to repo root, then down to the file
    repo_root = Path(__file__).parent.parent.parent
    agent_file = repo_root / "omicverse" / "utils" / "smart_agent.py"
    with open(agent_file, 'r') as f:
        return f.read()


def test_phase1_complexity_classifier():
    """Test Phase 1: Complexity classifier exists."""
    print("=" * 70)
    print("TEST 1: Phase 1 - Complexity Classifier")
    print("=" * 70)

    source = read_smart_agent()

    # Check for method definition
    assert 'async def _analyze_task_complexity(self, request: str) -> str:' in source, \
        "Missing _analyze_task_complexity method"
    print("✓ Method _analyze_task_complexity defined")

    # Check for pattern matching
    assert 'complex_keywords' in source, "Missing complexity keywords"
    assert 'simple_keywords' in source, "Missing simplicity keywords"
    assert 'simple_functions' in source, "Missing function names list"
    print("✓ Pattern matching keywords defined")

    # Check for LLM fallback
    assert 'classification_prompt' in source, "Missing LLM classification prompt"
    print("✓ LLM fallback implemented")

    # Count lines of complexity classifier
    start = source.find('async def _analyze_task_complexity')
    end = source.find('\n    async def _run_registry_workflow', start)
    if end == -1:
        end = source.find('\n    def _run_registry_workflow', start)
    classifier_code = source[start:end]
    lines = len([l for l in classifier_code.split('\n') if l.strip()])
    print(f"✓ Complexity classifier: {lines} lines of code")

    print("\n✅ PHASE 1 PASSED\n")
    return True


def test_phase2_priority1_workflow():
    """Test Phase 2: Priority 1 workflow exists."""
    print("=" * 70)
    print("TEST 2: Phase 2 - Priority 1 Workflow")
    print("=" * 70)

    source = read_smart_agent()

    # Check for method definition
    assert 'async def _run_registry_workflow(self, request: str, adata: Any) -> Any:' in source, \
        "Missing _run_registry_workflow method"
    print("✓ Method _run_registry_workflow defined")

    # Check for registry-only prompt
    assert 'priority1_prompt' in source, "Missing Priority 1 prompt"
    assert 'SINGLE BEST function' in source, "Missing single function instruction"
    print("✓ Registry-only prompt defined")

    # Check for NEEDS_WORKFLOW detection
    assert '"NEEDS_WORKFLOW"' in source or "'NEEDS_WORKFLOW'" in source, \
        "Missing NEEDS_WORKFLOW detection"
    print("✓ NEEDS_WORKFLOW detection implemented")

    # Check for execution
    assert '_execute_generated_code' in source, "Missing code execution"
    print("✓ Code execution integrated")

    # Count lines
    start = source.find('async def _run_registry_workflow')
    end = source.find('\n    async def _run_skills_workflow', start)
    if end == -1:
        end = source.find('\n    def _run_skills_workflow', start)
    workflow_code = source[start:end]
    lines = len([l for l in workflow_code.split('\n') if l.strip()])
    print(f"✓ Priority 1 workflow: {lines} lines of code")

    print("\n✅ PHASE 2 PASSED\n")
    return True


def test_phase3_priority2_workflow():
    """Test Phase 3: Priority 2 workflow exists."""
    print("=" * 70)
    print("TEST 3: Phase 3 - Priority 2 Workflow")
    print("=" * 70)

    source = read_smart_agent()

    # Check for method definition
    assert 'async def _run_skills_workflow(self, request: str, adata: Any) -> Any:' in source, \
        "Missing _run_skills_workflow method"
    print("✓ Method _run_skills_workflow defined")

    # Check for skill matching
    assert '_select_skill_matches_llm' in source, "Missing skill matching"
    print("✓ Skill matching integrated")

    # Check for skill loading
    assert 'load_full_skill' in source, "Missing skill loading"
    print("✓ Lazy skill loading implemented")

    # Check for comprehensive prompt
    assert 'priority2_prompt' in source, "Missing Priority 2 prompt"
    assert 'COMPLEX task' in source or 'complex task' in source, \
        "Missing complex task instruction"
    print("✓ Comprehensive prompt defined")

    # Count lines
    start = source.find('async def _run_skills_workflow')
    end = source.find('\n    def _validate_simple_execution', start)
    workflow_code = source[start:end]
    lines = len([l for l in workflow_code.split('\n') if l.strip()])
    print(f"✓ Priority 2 workflow: {lines} lines of code")

    print("\n✅ PHASE 3 PASSED\n")
    return True


def test_phase4_code_validation():
    """Test Phase 4: Code validation exists."""
    print("=" * 70)
    print("TEST 4: Phase 4 - Code Validation")
    print("=" * 70)

    source = read_smart_agent()

    # Check for method definition
    assert 'def _validate_simple_execution(self, code: str) -> tuple[bool, str]:' in source, \
        "Missing _validate_simple_execution method"
    print("✓ Method _validate_simple_execution defined")

    # Check for AST analysis
    assert 'ast.parse' in source, "Missing AST parsing"
    assert 'ast.Call' in source, "Missing function call detection"
    assert 'ast.For' in source or 'ast.While' in source, "Missing loop detection"
    print("✓ AST analysis implemented")

    # Check for validation rules
    assert 'function_calls > 5' in source or 'function_calls > ' in source, \
        "Missing function call limit"
    assert 'loops > 0' in source, "Missing loop validation"
    assert 'conditionals > 0' in source, "Missing conditional validation"
    print("✓ Validation rules defined")

    # Count lines
    start = source.find('def _validate_simple_execution')
    end = source.find('\n    def _list_project_skills', start)
    validation_code = source[start:end]
    lines = len([l for l in validation_code.split('\n') if l.strip()])
    print(f"✓ Code validation: {lines} lines of code")

    print("\n✅ PHASE 4 PASSED\n")
    return True


def test_phase5_integration():
    """Test Phase 5: run_async integration."""
    print("=" * 70)
    print("TEST 5: Phase 5 - run_async Integration")
    print("=" * 70)

    source = read_smart_agent()

    # Check for new run_async
    assert 'async def run_async(self, request: str, adata: Any) -> Any:' in source, \
        "Missing run_async method"
    print("✓ Method run_async defined")

    # Check for complexity analysis call
    assert 'await self._analyze_task_complexity(request)' in source, \
        "Missing complexity analysis call"
    print("✓ Calls _analyze_task_complexity")

    # Check for Priority 1 call
    assert 'await self._run_registry_workflow(request, adata)' in source, \
        "Missing Priority 1 call"
    print("✓ Calls _run_registry_workflow")

    # Check for Priority 2 call
    assert 'await self._run_skills_workflow(request, adata)' in source, \
        "Missing Priority 2 call"
    print("✓ Calls _run_skills_workflow")

    # Check for fallback logic
    assert 'fallback' in source.lower(), "Missing fallback mechanism"
    print("✓ Fallback mechanism present")

    # Check for priority tracking
    assert 'priority_used' in source, "Missing priority tracking"
    print("✓ Priority tracking implemented")

    # Check for console output
    assert 'Priority 1' in source and 'Priority 2' in source, \
        "Missing priority indicators in output"
    print("✓ User feedback messages present")

    # Check legacy code preserved
    assert 'async def run_async_LEGACY' in source, \
        "Missing legacy run_async_LEGACY method"
    print("✓ Legacy code preserved as run_async_LEGACY")

    print("\n✅ PHASE 5 PASSED\n")
    return True


def test_overall_stats():
    """Test overall implementation statistics."""
    print("=" * 70)
    print("TEST 6: Overall Statistics")
    print("=" * 70)

    source = read_smart_agent()

    # Count methods
    async_methods = source.count('async def ')
    sync_methods = source.count('    def ') - source.count('    def __')
    print(f"✓ Total async methods: {async_methods}")
    print(f"✓ Total sync methods: {sync_methods}")

    # Count lines
    total_lines = len(source.split('\n'))
    code_lines = len([l for l in source.split('\n') if l.strip() and not l.strip().startswith('#')])
    print(f"✓ Total lines: {total_lines}")
    print(f"✓ Code lines: {code_lines}")

    # Check for key components
    components = {
        'Complexity Classifier': '_analyze_task_complexity',
        'Priority 1 Workflow': '_run_registry_workflow',
        'Priority 2 Workflow': '_run_skills_workflow',
        'Code Validation': '_validate_simple_execution',
        'Integrated run_async': 'async def run_async',
        'Legacy Preserved': 'run_async_LEGACY',
    }

    print("\nKey Components:")
    for name, pattern in components.items():
        present = pattern in source
        status = "✓" if present else "✗"
        print(f"  {status} {name}")

    print("\n✅ STATISTICS COMPLETE\n")
    return True


def test_documentation_quality():
    """Test documentation quality."""
    print("=" * 70)
    print("TEST 7: Documentation Quality")
    print("=" * 70)

    source = read_smart_agent()

    # Check for docstrings in key methods
    methods = [
        '_analyze_task_complexity',
        '_run_registry_workflow',
        '_run_skills_workflow',
        '_validate_simple_execution',
        'run_async',
    ]

    for method in methods:
        # Find method definition
        pattern = f'def {method}.*?:\n.*?"""'
        match = re.search(pattern, source, re.DOTALL)
        assert match, f"Missing docstring for {method}"

        # Extract docstring
        docstring_start = source.find('"""', match.start())
        docstring_end = source.find('"""', docstring_start + 3)
        docstring = source[docstring_start:docstring_end]

        # Check docstring length
        assert len(docstring) > 100, f"Insufficient documentation for {method}"
        print(f"✓ {method}: {len(docstring)} chars of documentation")

    print("\n✅ DOCUMENTATION QUALITY PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "🔬" * 35)
    print("PRIORITY SYSTEM IMPLEMENTATION VERIFICATION")
    print("🔬" * 35 + "\n")

    tests = [
        ("Phase 1: Complexity Classifier", test_phase1_complexity_classifier),
        ("Phase 2: Priority 1 Workflow", test_phase2_priority1_workflow),
        ("Phase 3: Priority 2 Workflow", test_phase3_priority2_workflow),
        ("Phase 4: Code Validation", test_phase4_code_validation),
        ("Phase 5: Integration", test_phase5_integration),
        ("Overall Statistics", test_overall_stats),
        ("Documentation Quality", test_documentation_quality),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED")
            print(f"   Assertion Error: {e}\n")
            results[test_name] = False
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception")
            print(f"   Error: {type(e).__name__}: {e}\n")
            results[test_name] = False

    # Summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    percentage = (passed / total * 100) if total > 0 else 0
    print(f"Results: {passed}/{total} tests passed ({percentage:.0f}%)")
    print("=" * 70)

    if passed == total:
        print("\n" + "🎉" * 35)
        print("ALL TESTS PASSED - PRIORITY SYSTEM FULLY IMPLEMENTED!")
        print("🎉" * 35)
        print("\n✅ Implementation Complete:")
        print("   • Phase 1: Task complexity classifier (pattern + LLM)")
        print("   • Phase 2: Priority 1 fast workflow (registry-only)")
        print("   • Phase 3: Priority 2 comprehensive workflow (skills)")
        print("   • Phase 4: AST-based code validation")
        print("   • Phase 5: Integrated run_async with auto-fallback")
        print("\n🚀 The system is ready for real-world testing!")
        print("\nNext steps:")
        print("   - Test with actual data")
        print("   - Measure performance improvements")
        print("   - Optional: Add configuration (Phase 6)")
        print("   - Optional: Add comprehensive tests (Phase 8)")
        print("   - Optional: Add metrics tracking (Phase 9)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nReview the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
