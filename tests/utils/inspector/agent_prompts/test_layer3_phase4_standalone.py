"""
Integration tests for Layer 3 Phase 4: AutoPrerequisiteInserter.

Tests the automatic prerequisite insertion system.
"""

import sys
import importlib.util
from pathlib import Path


def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# First, import data_structures to get shared classes
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
data_structures = import_module_from_path(
    'omicverse.utils.inspector.data_structures',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/data_structures.py')
)

# Get ComplexityLevel
ComplexityLevel = data_structures.ComplexityLevel

# Now manually load and execute auto_prerequisite_inserter code with injected dependencies
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Read auto_prerequisite_inserter code and exec it
with open(str(PROJECT_ROOT / 'omicverse/utils/inspector/auto_prerequisite_inserter.py'), 'r') as f:
    code = f.read()
    # Replace the imports with our local variables
    code = code.replace(
        'from .data_structures import ComplexityLevel',
        '# Imports handled by test harness'
    )
    # Execute in a namespace with our dependencies
    namespace = {
        'ComplexityLevel': ComplexityLevel,
        'List': List,
        'Dict': Dict,
        'Any': Any,
        'Optional': Optional,
        'Set': Set,
        'Tuple': Tuple,
        'dataclass': dataclass,
        'field': field,
        'Enum': Enum,
        're': re,
    }
    exec(code, namespace)

# Get classes from namespace
AutoPrerequisiteInserter = namespace['AutoPrerequisiteInserter']
InsertionResult = namespace['InsertionResult']
InsertionPolicy = namespace['InsertionPolicy']
auto_insert_prerequisites = namespace['auto_insert_prerequisites']


# Mock registry for testing
class MockRegistry:
    """Mock registry for testing."""

    def __init__(self):
        self.functions = {
            'scale': {
                'prerequisites': {'required': [], 'optional': []},
            },
            'pca': {
                'prerequisites': {'required': ['scale'], 'optional': []},
            },
            'neighbors': {
                'prerequisites': {'required': ['pca'], 'optional': []},
            },
            'umap': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
            },
            'leiden': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


# Test functions
def test_inserter_initialization():
    """Test AutoPrerequisiteInserter initialization."""
    print("Testing AutoPrerequisiteInserter initialization...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    assert inserter.registry is registry
    assert len(inserter.SIMPLE_PREREQUISITES) > 0
    assert len(inserter.COMPLEX_PREREQUISITES) > 0

    print("✓ test_inserter_initialization passed")


def test_can_auto_insert_simple():
    """Test can_auto_insert for simple prerequisites."""
    print("Testing can_auto_insert (simple)...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # All simple
    assert inserter.can_auto_insert(['scale', 'pca'])
    assert inserter.can_auto_insert(['neighbors'])
    assert inserter.can_auto_insert([])

    # Has complex
    assert not inserter.can_auto_insert(['qc'])
    assert not inserter.can_auto_insert(['scale', 'qc'])
    assert not inserter.can_auto_insert(['preprocess'])

    # Unknown
    assert not inserter.can_auto_insert(['unknown_function'])

    print("✓ test_can_auto_insert_simple passed")


def test_determine_policy_auto_insert():
    """Test policy determination for auto-insert case."""
    print("Testing policy determination (auto-insert)...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # Simple prerequisites
    policy = inserter._determine_policy(['scale', 'pca'], ComplexityLevel.LOW)
    assert policy == InsertionPolicy.AUTO_INSERT

    policy2 = inserter._determine_policy(['neighbors'], ComplexityLevel.LOW)
    assert policy2 == InsertionPolicy.AUTO_INSERT

    print("✓ test_determine_policy_auto_insert passed")


def test_determine_policy_escalate():
    """Test policy determination for escalation case."""
    print("Testing policy determination (escalate)...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # Complex with HIGH complexity
    policy = inserter._determine_policy(
        ['qc', 'scale', 'pca', 'neighbors'],
        ComplexityLevel.HIGH
    )
    assert policy == InsertionPolicy.ESCALATE

    # Many prerequisites
    policy2 = inserter._determine_policy(
        ['qc', 'normalize', 'scale', 'pca'],
        ComplexityLevel.HIGH
    )
    assert policy2 == InsertionPolicy.ESCALATE

    print("✓ test_determine_policy_escalate passed")


def test_determine_policy_manual():
    """Test policy determination for manual case."""
    print("Testing policy determination (manual)...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # Complex but MEDIUM complexity
    policy = inserter._determine_policy(['qc'], ComplexityLevel.MEDIUM)
    assert policy == InsertionPolicy.MANUAL

    # Unknown prerequisite
    policy2 = inserter._determine_policy(['unknown_func'], ComplexityLevel.LOW)
    assert policy2 == InsertionPolicy.MANUAL

    print("✓ test_determine_policy_manual passed")


def test_auto_insert_single():
    """Test auto-insertion of single prerequisite."""
    print("Testing auto-insertion (single prerequisite)...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.pca(adata, n_pcs=50)"
    result = inserter.insert_prerequisites(code, ['scale'])

    assert result.inserted
    assert result.insertion_policy == InsertionPolicy.AUTO_INSERT
    assert 'scale' in result.inserted_prerequisites
    assert 'ov.pp.scale(adata)' in result.modified_code
    assert 'ov.pp.pca(adata, n_pcs=50)' in result.modified_code
    assert '# Auto-inserted prerequisites' in result.modified_code

    # Check order: scale should come before pca
    scale_pos = result.modified_code.find('scale')
    pca_pos = result.modified_code.find('pca')
    assert scale_pos < pca_pos

    print("✓ test_auto_insert_single passed")


def test_auto_insert_multiple():
    """Test auto-insertion of multiple prerequisites."""
    print("Testing auto-insertion (multiple prerequisites)...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.leiden(adata, resolution=1.0)"
    result = inserter.insert_prerequisites(code, ['pca', 'neighbors'])

    assert result.inserted
    assert len(result.inserted_prerequisites) == 2
    assert 'pca' in result.inserted_prerequisites
    assert 'neighbors' in result.inserted_prerequisites
    assert 'ov.pp.pca(adata' in result.modified_code
    assert 'ov.pp.neighbors(adata' in result.modified_code
    assert 'ov.pp.leiden(adata' in result.modified_code

    # Check order: pca should come before neighbors
    pca_pos = result.modified_code.find('pca')
    neighbors_pos = result.modified_code.find('neighbors')
    leiden_pos = result.modified_code.find('leiden')
    assert pca_pos < neighbors_pos < leiden_pos

    print("✓ test_auto_insert_multiple passed")


def test_auto_insert_with_dependencies():
    """Test auto-insertion with dependency chain."""
    print("Testing auto-insertion with dependencies...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.umap(adata)"
    # Missing neighbors, but neighbors requires pca which requires scale
    result = inserter.insert_prerequisites(code, ['neighbors'])

    assert result.inserted
    assert 'neighbors' in result.inserted_prerequisites
    # Should only insert what was requested, not transitive dependencies
    # (those should be detected separately by the validation system)

    print("✓ test_auto_insert_with_dependencies passed")


def test_no_insertion_needed():
    """Test when no prerequisites are missing."""
    print("Testing no insertion needed...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.leiden(adata, resolution=1.0)"
    result = inserter.insert_prerequisites(code, [])

    assert not result.inserted
    assert result.modified_code == code
    assert "No missing prerequisites" in result.explanation

    print("✓ test_no_insertion_needed passed")


def test_escalation_suggestion():
    """Test escalation suggestion for complex prerequisites."""
    print("Testing escalation suggestion...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.leiden(adata, resolution=1.0)"
    result = inserter.insert_prerequisites(
        code,
        ['qc', 'normalize', 'scale', 'pca', 'neighbors'],
        ComplexityLevel.HIGH
    )

    assert not result.inserted
    assert result.insertion_policy == InsertionPolicy.ESCALATE
    assert result.alternative_suggestion is not None
    assert 'preprocess' in result.alternative_suggestion.lower()
    assert 'qc' in result.explanation.lower() or 'normalize' in result.explanation.lower()

    print("✓ test_escalation_suggestion passed")


def test_manual_suggestion():
    """Test manual suggestion for complex prerequisites."""
    print("Testing manual suggestion...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.leiden(adata)"
    result = inserter.insert_prerequisites(
        code,
        ['qc'],
        ComplexityLevel.MEDIUM
    )

    assert not result.inserted
    assert result.insertion_policy == InsertionPolicy.MANUAL
    assert result.alternative_suggestion is not None
    assert 'qc' in result.alternative_suggestion

    print("✓ test_manual_suggestion passed")


def test_resolve_dependencies():
    """Test dependency resolution (topological sort)."""
    print("Testing dependency resolution...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # Test with dependencies: neighbors -> pca -> scale
    prereqs = ['neighbors', 'scale', 'pca']
    ordered = inserter._resolve_dependencies(prereqs)

    scale_idx = ordered.index('scale')
    pca_idx = ordered.index('pca')
    neighbors_idx = ordered.index('neighbors')

    # scale should come first, then pca, then neighbors
    assert scale_idx < pca_idx < neighbors_idx

    print("✓ test_resolve_dependencies passed")


def test_insertion_result_structure():
    """Test InsertionResult structure."""
    print("Testing InsertionResult structure...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    result = inserter.insert_prerequisites(
        "ov.pp.leiden(adata)",
        ['pca', 'neighbors']
    )

    # Check all fields are present
    assert hasattr(result, 'inserted')
    assert hasattr(result, 'original_code')
    assert hasattr(result, 'modified_code')
    assert hasattr(result, 'inserted_prerequisites')
    assert hasattr(result, 'insertion_policy')
    assert hasattr(result, 'estimated_time_seconds')
    assert hasattr(result, 'explanation')
    assert hasattr(result, 'alternative_suggestion')

    # Check types
    assert isinstance(result.inserted, bool)
    assert isinstance(result.original_code, str)
    assert isinstance(result.modified_code, str)
    assert isinstance(result.inserted_prerequisites, list)
    assert isinstance(result.explanation, str)

    print("✓ test_insertion_result_structure passed")


def test_convenience_function():
    """Test auto_insert_prerequisites convenience function."""
    print("Testing auto_insert_prerequisites convenience function...")

    registry = MockRegistry()

    result = auto_insert_prerequisites(
        registry=registry,
        code="ov.pp.leiden(adata)",
        missing_prerequisites=['neighbors'],
        complexity=ComplexityLevel.LOW
    )

    assert isinstance(result, InsertionResult)
    assert result.inserted

    print("✓ test_convenience_function passed")


def test_time_estimation():
    """Test time estimation for inserted prerequisites."""
    print("Testing time estimation...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # Single prerequisite
    result1 = inserter.insert_prerequisites("ov.pp.pca(adata)", ['scale'])
    assert result1.estimated_time_seconds > 0

    # Multiple prerequisites - should sum the times
    result2 = inserter.insert_prerequisites(
        "ov.pp.umap(adata)",
        ['pca', 'neighbors']
    )
    assert result2.estimated_time_seconds > result1.estimated_time_seconds

    print("✓ test_time_estimation passed")


def test_code_formatting():
    """Test that code formatting is preserved."""
    print("Testing code formatting...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    code = "ov.pp.leiden(adata, resolution=1.0)"
    result = inserter.insert_prerequisites(code, ['neighbors'])

    # Check for proper comments
    assert '# Auto-inserted prerequisites' in result.modified_code
    assert '# Original code' in result.modified_code

    # Check that original code is preserved
    assert 'ov.pp.leiden(adata, resolution=1.0)' in result.modified_code

    print("✓ test_code_formatting passed")


def test_mixed_prerequisites():
    """Test handling of mixed simple and complex prerequisites."""
    print("Testing mixed prerequisites...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    # Mix of simple and complex
    result = inserter.insert_prerequisites(
        "ov.pp.leiden(adata)",
        ['scale', 'qc'],  # scale is simple, qc is complex
        ComplexityLevel.MEDIUM
    )

    assert not result.inserted  # Should not auto-insert due to qc
    assert result.insertion_policy == InsertionPolicy.MANUAL
    assert result.alternative_suggestion is not None

    print("✓ test_mixed_prerequisites passed")


def test_empty_code():
    """Test handling of empty code."""
    print("Testing empty code...")

    registry = MockRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    result = inserter.insert_prerequisites("", ['scale'])

    assert result.inserted
    assert 'ov.pp.scale(adata)' in result.modified_code

    print("✓ test_empty_code passed")


# Run all tests
def run_tests():
    """Run all Layer 3 Phase 4 tests."""
    print("="*60)
    print("Layer 3 Phase 4 - AutoPrerequisiteInserter Tests")
    print("="*60)
    print()

    tests = [
        test_inserter_initialization,
        test_can_auto_insert_simple,
        test_determine_policy_auto_insert,
        test_determine_policy_escalate,
        test_determine_policy_manual,
        test_auto_insert_single,
        test_auto_insert_multiple,
        test_auto_insert_with_dependencies,
        test_no_insertion_needed,
        test_escalation_suggestion,
        test_manual_suggestion,
        test_resolve_dependencies,
        test_insertion_result_structure,
        test_convenience_function,
        test_time_estimation,
        test_code_formatting,
        test_mixed_prerequisites,
        test_empty_code,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("="*60)

    if failed == 0:
        print()
        print("✅ All Phase 4 tests PASSED!")
        print()
        print("AutoPrerequisiteInserter Validation:")
        print("   ✓ Initialization and configuration")
        print("   ✓ Simple prerequisite detection")
        print("   ✓ Policy determination (AUTO_INSERT/ESCALATE/MANUAL)")
        print("   ✓ Auto-insertion (single and multiple)")
        print("   ✓ Dependency resolution and ordering")
        print("   ✓ Escalation suggestions")
        print("   ✓ Manual configuration guidance")
        print("   ✓ Time estimation")
        print("   ✓ Code formatting and comments")
        print()
        print("Phase 4 Status: ✅ COMPLETE")
        print()
        print("🎉 Layer 3 COMPLETE - All 4 Phases Implemented!")
    else:
        print()
        print("❌ Some tests failed:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
