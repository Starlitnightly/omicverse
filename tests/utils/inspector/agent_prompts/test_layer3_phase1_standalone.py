"""
Integration tests for Layer 3 Phase 1: AgentContextInjector.

Tests the context injection system that enhances LLM prompts with
prerequisite state and data structure information.
"""

import sys
import importlib.util
import numpy as np
from anndata import AnnData
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

# Import required modules
data_structures = import_module_from_path(
    'omicverse.utils.inspector.data_structures',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/data_structures.py')
)

validators = import_module_from_path(
    'omicverse.utils.inspector.validators',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/validators.py')
)

prerequisite_checker = import_module_from_path(
    'omicverse.utils.inspector.prerequisite_checker',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/prerequisite_checker.py')
)

suggestion_engine = import_module_from_path(
    'omicverse.utils.inspector.suggestion_engine',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/suggestion_engine.py')
)

llm_formatter = import_module_from_path(
    'omicverse.utils.inspector.llm_formatter',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/llm_formatter.py')
)

inspector_module = import_module_from_path(
    'omicverse.utils.inspector.inspector',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/inspector.py')
)

agent_context_injector = import_module_from_path(
    'omicverse.utils.inspector.agent_context_injector',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/agent_context_injector.py')
)

# Get classes
AgentContextInjector = agent_context_injector.AgentContextInjector
ConversationState = agent_context_injector.ConversationState


# Mock registry for testing
class MockRegistry:
    """Mock registry for testing."""

    def __init__(self):
        self.functions = {
            'qc': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'obs': ['n_genes', 'n_counts']},
                'auto_fix': 'none',
            },
            'preprocess': {
                'prerequisites': {'required': ['qc'], 'optional': []},
                'requires': {},
                'produces': {'layers': ['normalized'], 'var': ['highly_variable']},
                'auto_fix': 'none',
            },
            'pca': {
                'prerequisites': {'required': ['preprocess'], 'optional': []},
                'requires': {},
                'produces': {'obsm': ['X_pca'], 'uns': ['pca']},
                'auto_fix': 'escalate',
            },
            'neighbors': {
                'prerequisites': {'required': ['pca'], 'optional': []},
                'requires': {'obsm': ['X_pca']},
                'produces': {'obsp': ['connectivities', 'distances'], 'uns': ['neighbors']},
                'auto_fix': 'auto',
            },
            'leiden': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
                'requires': {'obsp': ['connectivities']},
                'produces': {'obs': ['leiden']},
                'auto_fix': 'auto',
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


def create_test_adata(with_pca=False, with_neighbors=False):
    """Create test AnnData object."""
    np.random.seed(42)
    X = np.random.rand(100, 50)
    adata = AnnData(X)
    adata.obs_names = [f"Cell_{i}" for i in range(100)]
    adata.var_names = [f"Gene_{i}" for i in range(50)]

    if with_pca:
        adata.obsm['X_pca'] = np.random.rand(100, 50)
        adata.uns['pca'] = {'variance_ratio': np.random.rand(50)}

    if with_neighbors:
        adata.obsp['connectivities'] = np.random.rand(100, 100)
        adata.obsp['distances'] = np.random.rand(100, 100)
        adata.uns['neighbors'] = {'params': {'n_neighbors': 15}}

    return adata


# Test functions
def test_conversation_state():
    """Test ConversationState class."""
    print("Testing ConversationState...")

    state = ConversationState()

    # Test initial state
    assert len(state.executed_functions) == 0
    assert len(state.execution_history) == 0

    # Test adding execution
    state.add_execution('pca')
    assert 'pca' in state.executed_functions
    assert len(state.execution_history) == 1
    assert state.execution_history[0]['function'] == 'pca'

    # Test multiple executions
    state.add_execution('neighbors')
    state.add_execution('leiden')
    assert len(state.executed_functions) == 3
    assert len(state.execution_history) == 3

    # Test snapshot
    test_state = {'test': 'data'}
    state.snapshot_data_state(test_state)
    assert len(state.data_snapshots) == 1
    assert state.data_snapshots[0]['state'] == test_state

    print("✓ test_conversation_state passed")


def test_injector_initialization():
    """Test AgentContextInjector initialization."""
    print("Testing AgentContextInjector initialization...")

    adata = create_test_adata()
    registry = MockRegistry()

    injector = AgentContextInjector(adata, registry)

    assert injector.adata is adata
    assert injector.registry is registry
    assert injector.inspector is not None
    assert injector.conversation_state is not None
    assert len(injector.conversation_state.data_snapshots) == 1  # Initial snapshot

    print("✓ test_injector_initialization passed")


def test_inject_context_basic():
    """Test basic context injection."""
    print("Testing basic context injection...")

    adata = create_test_adata()
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    system_prompt = "You are a helpful bioinformatics assistant."
    enhanced = injector.inject_context(system_prompt)

    # Check that original prompt is included
    assert system_prompt in enhanced

    # Check for key sections
    assert "Current AnnData State" in enhanced
    assert "Prerequisite Handling Instructions" in enhanced
    assert "IMPORTANT" in enhanced

    # Check for data structure mentions
    assert "adata.obsm" in enhanced or "adata.obsp" in enhanced

    print("✓ test_inject_context_basic passed")


def test_inject_context_with_target():
    """Test context injection with target function."""
    print("Testing context injection with target function...")

    adata = create_test_adata(with_pca=True)  # Has PCA
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    system_prompt = "You are a helpful assistant."
    enhanced = injector.inject_context(
        system_prompt,
        target_function='leiden'
    )

    # Check for function-specific context
    assert "Target Function: leiden" in enhanced
    assert "Prerequisites Status" in enhanced or "Missing Prerequisites" in enhanced

    # Check for recommendations
    assert "Recommendations" in enhanced or "Prerequisites Status" in enhanced

    print("✓ test_inject_context_with_target passed")


def test_general_state_section():
    """Test general state section building."""
    print("Testing general state section...")

    adata = create_test_adata(with_pca=True, with_neighbors=True)
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    section = injector._build_general_state_section()

    # Check for key elements
    assert "Current AnnData State" in section
    assert "Available Data Structures" in section
    assert "Analysis Status" in section

    # Check for data structure detection
    assert "X_pca" in section  # Should detect PCA embedding
    assert "connectivities" in section  # Should detect neighbor graph

    # Check for status indicators
    assert "✅" in section or "❌" in section  # Status emojis

    print("✓ test_general_state_section passed")


def test_function_specific_section():
    """Test function-specific section building."""
    print("Testing function-specific section...")

    adata = create_test_adata(with_pca=True)  # Has PCA but no neighbors
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    section = injector._build_function_specific_section('leiden')

    # Check for key elements
    assert "Target Function: leiden" in section
    assert "Prerequisites Status" in section or "Missing Prerequisites" in section

    # Should mention missing neighbors
    assert "neighbors" in section.lower()

    print("✓ test_function_specific_section passed")


def test_function_specific_section_satisfied():
    """Test function-specific section when prerequisites satisfied."""
    print("Testing function-specific section (satisfied)...")

    adata = create_test_adata(with_pca=True, with_neighbors=True)
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    section = injector._build_function_specific_section('leiden')

    # Should indicate prerequisites are satisfied
    assert "satisfied" in section.lower() or "✅" in section

    print("✓ test_function_specific_section_satisfied passed")


def test_update_after_execution():
    """Test updating state after function execution."""
    print("Testing update after execution...")

    adata = create_test_adata()
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    # Initial state
    initial_snapshots = len(injector.conversation_state.data_snapshots)
    assert len(injector.conversation_state.executed_functions) == 0

    # Update after execution
    injector.update_after_execution('pca')

    # Check state updated
    assert 'pca' in injector.conversation_state.executed_functions
    assert len(injector.conversation_state.execution_history) == 1
    assert len(injector.conversation_state.data_snapshots) == initial_snapshots + 1

    # Update again
    injector.update_after_execution('neighbors')
    assert len(injector.conversation_state.executed_functions) == 2

    print("✓ test_update_after_execution passed")


def test_conversation_summary():
    """Test conversation summary generation."""
    print("Testing conversation summary...")

    adata = create_test_adata()
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    # Add some executions
    injector.update_after_execution('pca')
    injector.update_after_execution('neighbors')
    injector.update_after_execution('leiden')

    summary = injector.get_conversation_summary()

    # Check summary content
    assert "Conversation Summary" in summary
    assert "Functions Executed" in summary
    assert "pca" in summary
    assert "neighbors" in summary
    assert "leiden" in summary

    # Check counts
    assert "3" in summary  # 3 functions executed

    print("✓ test_conversation_summary passed")


def test_clear_conversation_state():
    """Test clearing conversation state."""
    print("Testing clear conversation state...")

    adata = create_test_adata()
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    # Add some executions
    injector.update_after_execution('pca')
    injector.update_after_execution('neighbors')
    assert len(injector.conversation_state.executed_functions) == 2

    # Clear state
    injector.clear_conversation_state()

    # Check state cleared
    assert len(injector.conversation_state.executed_functions) == 0
    assert len(injector.conversation_state.execution_history) == 0
    # But snapshots should have initial snapshot
    assert len(injector.conversation_state.data_snapshots) == 1

    print("✓ test_clear_conversation_state passed")


def test_detect_executed_functions():
    """Test detection of executed functions."""
    print("Testing detect executed functions...")

    adata = create_test_adata(with_pca=True, with_neighbors=True)
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    detected = injector._detect_executed_functions()

    # Should be a dict
    assert isinstance(detected, dict)

    # Should have confidence scores for detected functions
    for func_name, confidence in detected.items():
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    print("✓ test_detect_executed_functions passed")


def test_prerequisite_instructions():
    """Test prerequisite instruction generation."""
    print("Testing prerequisite instructions...")

    adata = create_test_adata()
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    instructions = injector._build_prerequisite_instructions()

    # Check for key instruction sections
    assert "IMPORTANT" in instructions
    assert "Prerequisite Handling Instructions" in instructions
    assert "Check Prerequisites First" in instructions
    assert "Auto-Insert Simple Prerequisites" in instructions
    assert "Escalate Complex Workflows" in instructions
    assert "Generate Complete, Executable Code" in instructions

    # Check for examples
    assert "ov.pp.neighbors" in instructions or "neighbors" in instructions
    assert "ov.pp.preprocess" in instructions or "preprocess" in instructions

    print("✓ test_prerequisite_instructions passed")


def test_context_injection_selective():
    """Test selective context injection."""
    print("Testing selective context injection...")

    adata = create_test_adata()
    registry = MockRegistry()
    injector = AgentContextInjector(adata, registry)

    system_prompt = "You are a helpful assistant."

    # Test with only general state
    enhanced1 = injector.inject_context(
        system_prompt,
        include_general_state=True,
        include_function_specific=False,
        include_instructions=False
    )
    assert "Current AnnData State" in enhanced1
    assert "Target Function" not in enhanced1
    assert "IMPORTANT" not in enhanced1

    # Test with only instructions
    enhanced2 = injector.inject_context(
        system_prompt,
        include_general_state=False,
        include_function_specific=False,
        include_instructions=True
    )
    assert "Current AnnData State" not in enhanced2
    assert "IMPORTANT" in enhanced2

    print("✓ test_context_injection_selective passed")


# Run all tests
def run_tests():
    """Run all Layer 3 Phase 1 tests."""
    print("="*60)
    print("Layer 3 Phase 1 - AgentContextInjector Tests")
    print("="*60)
    print()

    tests = [
        test_conversation_state,
        test_injector_initialization,
        test_inject_context_basic,
        test_inject_context_with_target,
        test_general_state_section,
        test_function_specific_section,
        test_function_specific_section_satisfied,
        test_update_after_execution,
        test_conversation_summary,
        test_clear_conversation_state,
        test_detect_executed_functions,
        test_prerequisite_instructions,
        test_context_injection_selective,
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
        print("✅ All Phase 1 tests PASSED!")
        print()
        print("AgentContextInjector Validation:")
        print("   ✓ Context injection working")
        print("   ✓ Conversation state tracking")
        print("   ✓ General state section generation")
        print("   ✓ Function-specific context")
        print("   ✓ Prerequisite instructions")
        print("   ✓ Execution updates")
        print("   ✓ Summary generation")
        print()
        print("Phase 1 Status: ✅ COMPLETE")
        print()
        print("Next: Phase 2 - DataStateValidator")
    else:
        print()
        print("❌ Some tests failed:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
