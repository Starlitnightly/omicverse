"""
Integration tests for Layer 3 Phase 3: WorkflowEscalator.

Tests the workflow complexity analysis and intelligent escalation system.
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

# Get ComplexityLevel and Suggestion
ComplexityLevel = data_structures.ComplexityLevel
Suggestion = data_structures.Suggestion

# Now manually load and execute workflow_escalator code with injected dependencies
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Read workflow_escalator code and exec it
with open(str(PROJECT_ROOT / 'omicverse/utils/inspector/workflow_escalator.py'), 'r') as f:
    code = f.read()
    # Replace the imports with our local variables
    code = code.replace(
        'from .data_structures import Suggestion, ComplexityLevel',
        '# Imports handled by test harness'
    )
    # Execute in a namespace with our dependencies
    namespace = {
        'Suggestion': Suggestion,
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
WorkflowEscalator = namespace['WorkflowEscalator']
EscalationResult = namespace['EscalationResult']
EscalationStrategy = namespace['EscalationStrategy']
analyze_and_escalate = namespace['analyze_and_escalate']


# Mock registry for testing
class MockRegistry:
    """Mock registry for testing."""

    def __init__(self):
        self.functions = {
            'qc': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'obs': ['n_genes', 'n_counts']},
            },
            'normalize': {
                'prerequisites': {'required': ['qc'], 'optional': []},
                'requires': {},
                'produces': {'layers': ['normalized']},
            },
            'highly_variable_genes': {
                'prerequisites': {'required': ['normalize'], 'optional': []},
                'requires': {},
                'produces': {'var': ['highly_variable']},
            },
            'scale': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'X': []},
            },
            'pca': {
                'prerequisites': {'required': ['scale'], 'optional': []},
                'requires': {},
                'produces': {'obsm': ['X_pca'], 'uns': ['pca']},
            },
            'neighbors': {
                'prerequisites': {'required': ['pca'], 'optional': []},
                'requires': {'obsm': ['X_pca']},
                'produces': {'obsp': ['connectivities', 'distances'], 'uns': ['neighbors']},
            },
            'umap': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
                'requires': {'obsp': ['connectivities']},
                'produces': {'obsm': ['X_umap']},
            },
            'leiden': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
                'requires': {'obsp': ['connectivities']},
                'produces': {'obs': ['leiden']},
            },
            'tsne': {
                'prerequisites': {'required': ['pca'], 'optional': []},
                'requires': {'obsm': ['X_pca']},
                'produces': {'obsm': ['X_tsne']},
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


# Test functions
def test_escalator_initialization():
    """Test WorkflowEscalator initialization."""
    print("Testing WorkflowEscalator initialization...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    assert escalator.registry is registry
    assert len(escalator.HIGH_LEVEL_FUNCTIONS) > 0
    assert len(escalator.COMPLEX_TRIGGERS) > 0

    print("✓ test_escalator_initialization passed")


def test_complexity_analysis_low():
    """Test complexity analysis for LOW complexity."""
    print("Testing complexity analysis (LOW)...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Single missing prerequisite
    missing = ['scale']
    complexity = escalator.analyze_complexity(missing)
    assert complexity == ComplexityLevel.LOW

    # No missing prerequisites
    missing_empty = []
    complexity_empty = escalator.analyze_complexity(missing_empty)
    assert complexity_empty == ComplexityLevel.LOW

    print("✓ test_complexity_analysis_low passed")


def test_complexity_analysis_medium():
    """Test complexity analysis for MEDIUM complexity."""
    print("Testing complexity analysis (MEDIUM)...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Two missing prerequisites
    missing = ['scale', 'pca']
    complexity = escalator.analyze_complexity(missing)
    assert complexity == ComplexityLevel.MEDIUM

    # Three missing prerequisites
    missing3 = ['scale', 'pca', 'neighbors']
    complexity3 = escalator.analyze_complexity(missing3)
    assert complexity3 == ComplexityLevel.MEDIUM

    print("✓ test_complexity_analysis_medium passed")


def test_complexity_analysis_high():
    """Test complexity analysis for HIGH complexity."""
    print("Testing complexity analysis (HIGH)...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Four or more missing prerequisites
    missing = ['scale', 'pca', 'neighbors', 'umap']
    complexity = escalator.analyze_complexity(missing)
    assert complexity == ComplexityLevel.HIGH

    # Contains complex trigger (qc)
    missing_qc = ['qc', 'scale']
    complexity_qc = escalator.analyze_complexity(missing_qc)
    assert complexity_qc == ComplexityLevel.HIGH

    # Contains complex trigger (preprocess)
    missing_preprocess = ['preprocess']
    complexity_preprocess = escalator.analyze_complexity(missing_preprocess)
    assert complexity_preprocess == ComplexityLevel.HIGH

    print("✓ test_complexity_analysis_high passed")


def test_dependency_depth():
    """Test dependency depth calculation."""
    print("Testing dependency depth calculation...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Single function with no dependencies
    depth1 = escalator._calculate_dependency_depth(['scale'])
    assert depth1 >= 1

    # Function with dependencies: neighbors -> pca -> scale
    depth2 = escalator._calculate_dependency_depth(['neighbors'])
    assert depth2 >= 2

    # Multiple functions
    depth3 = escalator._calculate_dependency_depth(['pca', 'neighbors'])
    assert depth3 >= 2

    print("✓ test_dependency_depth passed")


def test_should_escalate_low():
    """Test escalation decision for LOW complexity (no escalation)."""
    print("Testing escalation decision (LOW - no escalation)...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Simple case: single missing prerequisite
    result = escalator.should_escalate(
        target_function='pca',
        missing_prerequisites=['scale'],
        missing_data={}
    )

    assert not result.should_escalate
    assert result.complexity == ComplexityLevel.LOW
    assert result.strategy == EscalationStrategy.NO_ESCALATION
    assert result.escalated_suggestion is None
    assert "simple" in result.explanation.lower()

    print("✓ test_should_escalate_low passed")


def test_should_escalate_medium():
    """Test escalation decision for MEDIUM complexity (workflow chain)."""
    print("Testing escalation decision (MEDIUM - workflow chain)...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Medium case: 2-3 missing prerequisites
    result = escalator.should_escalate(
        target_function='umap',
        missing_prerequisites=['pca', 'neighbors'],
        missing_data={'obsm': ['X_pca'], 'obsp': ['connectivities']}
    )

    assert result.should_escalate
    assert result.complexity == ComplexityLevel.MEDIUM
    assert result.strategy == EscalationStrategy.WORKFLOW_CHAIN
    assert result.escalated_suggestion is not None
    assert "pca" in result.escalated_suggestion.code
    assert "neighbors" in result.escalated_suggestion.code
    assert "umap" in result.escalated_suggestion.code

    print("✓ test_should_escalate_medium passed")


def test_should_escalate_high():
    """Test escalation decision for HIGH complexity (high-level function)."""
    print("Testing escalation decision (HIGH - high-level function)...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # High case: many missing prerequisites including qc
    result = escalator.should_escalate(
        target_function='leiden',
        missing_prerequisites=['qc', 'normalize', 'scale', 'pca', 'neighbors'],
        missing_data={}
    )

    assert result.should_escalate
    assert result.complexity == ComplexityLevel.HIGH
    assert result.strategy == EscalationStrategy.HIGH_LEVEL_FUNCTION
    assert result.escalated_suggestion is not None
    assert "preprocess" in result.escalated_suggestion.code
    assert result.escalated_suggestion.priority == 'HIGH'

    print("✓ test_should_escalate_high passed")


def test_escalate_to_preprocess():
    """Test escalation to preprocess() high-level function."""
    print("Testing escalation to preprocess()...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Case requiring preprocess
    result = escalator.should_escalate(
        target_function='umap',
        missing_prerequisites=['qc', 'scale', 'pca'],
        missing_data={}
    )

    assert result.should_escalate
    assert result.escalated_suggestion is not None
    assert "ov.pp.preprocess" in result.escalated_suggestion.code
    assert "umap" in result.escalated_suggestion.code
    assert result.escalated_suggestion.suggestion_type == 'workflow_escalation'

    print("✓ test_escalate_to_preprocess passed")


def test_generate_workflow_chain():
    """Test workflow chain generation for medium complexity."""
    print("Testing workflow chain generation...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Medium complexity: generate ordered chain
    result = escalator.should_escalate(
        target_function='leiden',
        missing_prerequisites=['pca', 'neighbors'],
        missing_data={}
    )

    assert result.should_escalate
    assert result.strategy == EscalationStrategy.WORKFLOW_CHAIN
    assert result.escalated_suggestion is not None

    code = result.escalated_suggestion.code
    # Check that functions appear in correct order
    pca_pos = code.find('pca')
    neighbors_pos = code.find('neighbors')
    leiden_pos = code.find('leiden')

    assert pca_pos < neighbors_pos < leiden_pos, "Functions not in correct order"

    print("✓ test_generate_workflow_chain passed")


def test_topological_sort():
    """Test topological sort for dependency ordering."""
    print("Testing topological sort...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Test with dependencies: neighbors -> pca -> scale
    functions = ['neighbors', 'scale', 'pca']
    sorted_funcs = escalator._topological_sort(functions)

    # scale should come first (no dependencies)
    # pca should come next (depends on scale)
    # neighbors should come last (depends on pca)
    scale_idx = sorted_funcs.index('scale')
    pca_idx = sorted_funcs.index('pca')
    neighbors_idx = sorted_funcs.index('neighbors')

    assert scale_idx < pca_idx < neighbors_idx

    print("✓ test_topological_sort passed")


def test_can_use_preprocess():
    """Test detection of when preprocess() can be used."""
    print("Testing preprocess() detection...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Case where preprocess can be used (2+ functions it replaces)
    missing1 = ['qc', 'scale', 'pca']
    can_use1 = escalator._can_use_preprocess(missing1)
    assert can_use1, "Should use preprocess when multiple functions can be replaced"

    # Case where preprocess should not be used (< 2 functions)
    missing2 = ['scale']
    can_use2 = escalator._can_use_preprocess(missing2)
    assert not can_use2, "Should not use preprocess for single simple function"

    # Case with normalize and highly_variable_genes
    missing3 = ['normalize', 'highly_variable_genes']
    can_use3 = escalator._can_use_preprocess(missing3)
    assert can_use3, "Should use preprocess for normalize + HVG"

    print("✓ test_can_use_preprocess passed")


def test_get_default_code():
    """Test default code generation for functions."""
    print("Testing default code generation...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Test known functions
    pca_code = escalator._get_default_code('pca')
    assert 'ov.pp.pca' in pca_code
    assert 'n_pcs' in pca_code

    neighbors_code = escalator._get_default_code('neighbors')
    assert 'ov.pp.neighbors' in neighbors_code
    assert 'n_neighbors' in neighbors_code

    # Test unknown function (fallback)
    unknown_code = escalator._get_default_code('unknown_func')
    assert 'ov.pp.unknown_func' in unknown_code

    print("✓ test_get_default_code passed")


def test_escalation_result_structure():
    """Test EscalationResult structure."""
    print("Testing EscalationResult structure...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    result = escalator.should_escalate(
        target_function='leiden',
        missing_prerequisites=['scale', 'pca', 'neighbors'],
        missing_data={}
    )

    # Check all fields are present
    assert hasattr(result, 'should_escalate')
    assert hasattr(result, 'complexity')
    assert hasattr(result, 'strategy')
    assert hasattr(result, 'escalated_suggestion')
    assert hasattr(result, 'original_workflow')
    assert hasattr(result, 'dependency_depth')
    assert hasattr(result, 'num_missing')
    assert hasattr(result, 'has_complex_prerequisites')
    assert hasattr(result, 'explanation')

    # Check values
    assert isinstance(result.should_escalate, bool)
    assert isinstance(result.complexity, ComplexityLevel)
    assert isinstance(result.strategy, EscalationStrategy)
    assert result.num_missing == 3
    assert isinstance(result.explanation, str)

    print("✓ test_escalation_result_structure passed")


def test_convenience_function():
    """Test analyze_and_escalate convenience function."""
    print("Testing analyze_and_escalate convenience function...")

    registry = MockRegistry()

    result = analyze_and_escalate(
        registry=registry,
        target_function='leiden',
        missing_prerequisites=['pca', 'neighbors'],
        missing_data={}
    )

    assert isinstance(result, EscalationResult)
    assert result.should_escalate
    assert result.complexity == ComplexityLevel.MEDIUM

    print("✓ test_convenience_function passed")


def test_complex_triggers():
    """Test that complex triggers cause HIGH complexity."""
    print("Testing complex triggers...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    # Test each complex trigger
    for trigger in ['qc', 'preprocess', 'batch_correct', 'highly_variable_genes']:
        result = escalator.should_escalate(
            target_function='leiden',
            missing_prerequisites=[trigger],
            missing_data={}
        )

        assert result.complexity == ComplexityLevel.HIGH, f"{trigger} should trigger HIGH complexity"
        assert result.has_complex_prerequisites

    print("✓ test_complex_triggers passed")


def test_empty_prerequisites():
    """Test handling of empty prerequisites list."""
    print("Testing empty prerequisites...")

    registry = MockRegistry()
    escalator = WorkflowEscalator(registry)

    result = escalator.should_escalate(
        target_function='leiden',
        missing_prerequisites=[],
        missing_data={}
    )

    assert not result.should_escalate
    assert result.complexity == ComplexityLevel.LOW
    assert result.num_missing == 0

    print("✓ test_empty_prerequisites passed")


# Run all tests
def run_tests():
    """Run all Layer 3 Phase 3 tests."""
    print("="*60)
    print("Layer 3 Phase 3 - WorkflowEscalator Tests")
    print("="*60)
    print()

    tests = [
        test_escalator_initialization,
        test_complexity_analysis_low,
        test_complexity_analysis_medium,
        test_complexity_analysis_high,
        test_dependency_depth,
        test_should_escalate_low,
        test_should_escalate_medium,
        test_should_escalate_high,
        test_escalate_to_preprocess,
        test_generate_workflow_chain,
        test_topological_sort,
        test_can_use_preprocess,
        test_get_default_code,
        test_escalation_result_structure,
        test_convenience_function,
        test_complex_triggers,
        test_empty_prerequisites,
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
        print("✅ All Phase 3 tests PASSED!")
        print()
        print("WorkflowEscalator Validation:")
        print("   ✓ Complexity analysis (LOW/MEDIUM/HIGH)")
        print("   ✓ Dependency depth calculation")
        print("   ✓ Escalation decision logic")
        print("   ✓ High-level function escalation (preprocess)")
        print("   ✓ Workflow chain generation")
        print("   ✓ Topological sort for ordering")
        print("   ✓ Complex trigger detection")
        print("   ✓ Default code generation")
        print()
        print("Phase 3 Status: ✅ COMPLETE")
        print()
        print("Next: Phase 4 - AutoPrerequisiteInserter")
    else:
        print()
        print("❌ Some tests failed:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
