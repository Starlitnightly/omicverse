"""
Standalone test for Layer 2 Phase 3 implementation (SuggestionEngine).

This test imports the inspector modules directly without importing
the full omicverse package.
"""

import sys
import os
import importlib.util
from pathlib import Path

# Add the omicverse directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Import data_structures first
data_structures = import_module_from_path(
    'omicverse.utils.inspector.data_structures',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/data_structures.py')
)

# Import suggestion_engine
suggestion_engine = import_module_from_path(
    'omicverse.utils.inspector.suggestion_engine',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/suggestion_engine.py')
)

# Get the classes we need
SuggestionEngine = suggestion_engine.SuggestionEngine
WorkflowPlan = suggestion_engine.WorkflowPlan
WorkflowStep = suggestion_engine.WorkflowStep
WorkflowStrategy = suggestion_engine.WorkflowStrategy


# Mock registry for testing
class MockRegistry:
    """Mock registry with test function metadata."""

    def __init__(self):
        self.functions = {
            'qc': {
                'prerequisites': {'functions': [], 'optional_functions': []},
                'requires': {},
                'produces': {},
                'auto_fix': 'auto',
            },
            'preprocess': {
                'prerequisites': {'functions': ['qc'], 'optional_functions': []},
                'requires': {},
                'produces': {},
                'auto_fix': 'auto',
            },
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
            'umap': {
                'prerequisites': {'functions': ['neighbors'], 'optional_functions': []},
                'requires': {'obsp': ['connectivities', 'distances']},
                'produces': {'obsm': ['X_umap']},
                'auto_fix': 'none',
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


def test_suggestion_engine_initialization():
    """Test SuggestionEngine initialization."""
    print("Testing SuggestionEngine initialization...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    assert engine.registry is not None, "Registry should be set"
    assert hasattr(engine, 'function_graph'), "Should have function_graph attribute"
    assert hasattr(engine, 'function_templates'), "Should have function_templates attribute"
    print("✓ test_suggestion_engine_initialization passed")


def test_generate_suggestions_missing_prerequisites():
    """Test generating suggestions for missing prerequisites."""
    print("Testing suggestions for missing prerequisites...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    # leiden requires neighbors
    suggestions = engine.generate_suggestions(
        function_name='leiden',
        missing_prerequisites=['neighbors'],
        missing_data={},
    )

    assert len(suggestions) > 0, "Should generate at least one suggestion"

    # Check for workflow suggestion
    workflow_suggestions = [s for s in suggestions if s.suggestion_type == 'workflow']
    assert len(workflow_suggestions) > 0, "Should have workflow suggestion"

    # Check for prerequisite suggestion
    prereq_suggestions = [s for s in suggestions if s.suggestion_type == 'prerequisite']
    assert len(prereq_suggestions) > 0, "Should have prerequisite suggestion"

    print(f"✓ test_generate_suggestions_missing_prerequisites passed ({len(suggestions)} suggestions)")


def test_generate_suggestions_missing_data():
    """Test generating suggestions for missing data structures."""
    print("Testing suggestions for missing data...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    # Missing obsm (PCA embeddings)
    suggestions = engine.generate_suggestions(
        function_name='neighbors',
        missing_prerequisites=[],
        missing_data={'obsm': ['X_pca']},
    )

    assert len(suggestions) > 0, "Should generate at least one suggestion"

    # Check for direct fix suggestion
    pca_suggestions = [s for s in suggestions if 'pca' in s.code.lower()]
    assert len(pca_suggestions) > 0, "Should suggest running PCA"

    # Check priority
    assert any(s.priority in ['CRITICAL', 'HIGH'] for s in pca_suggestions), \
        "PCA suggestion should be high priority"

    print(f"✓ test_generate_suggestions_missing_data passed ({len(suggestions)} suggestions)")


def test_create_workflow_plan():
    """Test workflow plan creation."""
    print("Testing workflow plan creation...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    # Create workflow for leiden (needs neighbors -> pca -> preprocess)
    plan = engine.create_workflow_plan(
        function_name='leiden',
        missing_prerequisites=['neighbors', 'pca', 'preprocess'],
        strategy=WorkflowStrategy.MINIMAL,
    )

    assert plan is not None, "Should create workflow plan"
    assert plan.strategy == WorkflowStrategy.MINIMAL, "Should use minimal strategy"
    assert len(plan.steps) == 3, "Should have 3 steps"

    # Check steps are in correct order (dependencies first)
    step_names = [step.function_name for step in plan.steps]
    # preprocess should come before pca
    assert step_names.index('preprocess') < step_names.index('pca'), \
        "preprocess should come before pca"
    # pca should come before neighbors
    assert step_names.index('pca') < step_names.index('neighbors'), \
        "pca should come before neighbors"

    # Check total time is calculated
    assert plan.total_time_seconds > 0, "Total time should be positive"

    # Check complexity
    assert plan.complexity in ['LOW', 'MEDIUM', 'HIGH'], "Complexity should be set"

    print(f"✓ test_create_workflow_plan passed ({plan.complexity} complexity, {plan.total_time_seconds}s)")


def test_workflow_step_creation():
    """Test workflow step creation."""
    print("Testing workflow step creation...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    step = engine._create_workflow_step('pca')

    assert step.function_name == 'pca', "Function name should be 'pca'"
    assert len(step.code) > 0, "Code should be generated"
    assert len(step.description) > 0, "Description should exist"
    assert step.estimated_time_seconds > 0, "Time estimate should be positive"
    assert 'ov.pp.pca' in step.code, "Code should call ov.pp.pca"

    print(f"✓ test_workflow_step_creation passed (step: {step})")


def test_dependency_resolution():
    """Test dependency resolution and ordering."""
    print("Testing dependency resolution...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    # Test with multiple functions that have dependencies
    functions = ['neighbors', 'pca', 'preprocess']
    ordered = engine._resolve_dependencies(functions)

    # Check order: preprocess -> pca -> neighbors
    assert ordered.index('preprocess') < ordered.index('pca'), \
        "preprocess should come before pca"
    assert ordered.index('pca') < ordered.index('neighbors'), \
        "pca should come before neighbors"

    print(f"✓ test_dependency_resolution passed (order: {' -> '.join(ordered)})")


def test_suggestion_priorities():
    """Test that suggestions are properly prioritized."""
    print("Testing suggestion priorities...")
    registry = MockRegistry()
    engine = SuggestionEngine(registry)

    suggestions = engine.generate_suggestions(
        function_name='leiden',
        missing_prerequisites=['neighbors', 'pca', 'preprocess'],
        missing_data={'obsm': ['X_pca'], 'obsp': ['connectivities']},
    )

    # Check that suggestions are sorted by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}

    for i in range(len(suggestions) - 1):
        curr_priority = priority_order.get(suggestions[i].priority, 999)
        next_priority = priority_order.get(suggestions[i+1].priority, 999)
        assert curr_priority <= next_priority, \
            f"Suggestions should be sorted by priority ({suggestions[i].priority} before {suggestions[i+1].priority})"

    print(f"✓ test_suggestion_priorities passed")
    print(f"  Priority distribution: {[s.priority for s in suggestions[:5]]}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Layer 2 Phase 3 - SuggestionEngine Tests")
    print("=" * 60)
    print()

    tests = [
        test_suggestion_engine_initialization,
        test_generate_suggestions_missing_prerequisites,
        test_generate_suggestions_missing_data,
        test_create_workflow_plan,
        test_workflow_step_creation,
        test_dependency_resolution,
        test_suggestion_priorities,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All Phase 3 tests PASSED!")
        print("\nSuggestionEngine Validation:")
        print("   ✓ Workflow plan creation and ordering")
        print("   ✓ Dependency resolution (topological sort)")
        print("   ✓ Suggestion generation for prerequisites")
        print("   ✓ Suggestion generation for missing data")
        print("   ✓ Priority-based sorting")
        print("   ✓ Cost-benefit analysis (time estimates)")
        print("\nPhase 3 Status: ✅ COMPLETE")
        print("\nNext Steps:")
        print("   - Create comprehensive Phase 3 completion summary")
        print("   - Update documentation")
        print("   - Commit and push changes")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
