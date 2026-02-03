"""
Standalone test for Layer 2 Phase 4 implementation (LLMFormatter).

This test imports the inspector modules directly without importing
the full omicverse package.
"""

import sys
import os
import importlib.util
import json
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

# Import prerequisite_checker (required by llm_formatter)
prerequisite_checker = import_module_from_path(
    'omicverse.utils.inspector.prerequisite_checker',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/prerequisite_checker.py')
)

# Import llm_formatter
llm_formatter = import_module_from_path(
    'omicverse.utils.inspector.llm_formatter',
    str(PROJECT_ROOT / 'omicverse/utils/inspector/llm_formatter.py')
)

# Get the classes we need
LLMFormatter = llm_formatter.LLMFormatter
OutputFormat = llm_formatter.OutputFormat
LLMPrompt = llm_formatter.LLMPrompt
ValidationResult = data_structures.ValidationResult
Suggestion = data_structures.Suggestion


def create_test_validation_result(is_valid=False):
    """Create a test ValidationResult."""
    suggestions = [
        Suggestion(
            priority='CRITICAL',
            suggestion_type='prerequisite',
            description='Run prerequisite: preprocess',
            code='ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)',
            explanation='leiden requires preprocess to be executed first.',
            estimated_time='30 seconds',
            estimated_time_seconds=30,
            prerequisites=[],
            impact='Satisfies prerequisite: preprocess',
            auto_executable=False,
        ),
        Suggestion(
            priority='HIGH',
            suggestion_type='direct_fix',
            description='Run PCA to generate embeddings',
            code='ov.pp.pca(adata, n_pcs=50)',
            explanation='PCA is required for downstream analysis.',
            estimated_time='10-60 seconds',
            estimated_time_seconds=30,
            prerequisites=['preprocess'],
            impact='Generates PCA embeddings',
            auto_executable=False,
        ),
    ]

    return ValidationResult(
        function_name='leiden',
        is_valid=is_valid,
        message='Missing requirements for leiden' if not is_valid else 'All requirements satisfied',
        missing_prerequisites=['preprocess', 'pca'] if not is_valid else [],
        missing_data_structures={'obsm': ['X_pca'], 'obsp': ['connectivities']} if not is_valid else {},
        executed_functions=['qc'] if not is_valid else ['qc', 'preprocess', 'pca', 'neighbors'],
        confidence_scores={'qc': 0.95, 'preprocess': 0.2, 'pca': 0.1} if not is_valid else {'qc': 0.95, 'preprocess': 0.95, 'pca': 0.95, 'neighbors': 0.95},
        suggestions=suggestions if not is_valid else [],
    )


def test_llm_formatter_initialization():
    """Test LLMFormatter initialization."""
    print("Testing LLMFormatter initialization...")
    formatter = LLMFormatter()

    assert formatter.output_format == OutputFormat.MARKDOWN, "Default should be MARKDOWN"
    assert formatter.verbose == True, "Default should be verbose"
    print("✓ test_llm_formatter_initialization passed")


def test_format_markdown():
    """Test markdown formatting."""
    print("Testing markdown formatting...")
    formatter = LLMFormatter(output_format=OutputFormat.MARKDOWN)
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_validation_result(result)

    assert '# Validation Result: leiden' in formatted, "Should have header"
    assert '❌ Invalid' in formatted, "Should show invalid status"
    assert 'Missing Prerequisites' in formatted, "Should list prerequisites"
    assert 'Missing Data Structures' in formatted, "Should list data structures"
    assert 'Suggestions' in formatted, "Should have suggestions section"
    assert '```python' in formatted, "Should have code blocks"
    assert 'ov.pp.preprocess' in formatted, "Should include code"
    print("✓ test_format_markdown passed")


def test_format_plain_text():
    """Test plain text formatting."""
    print("Testing plain text formatting...")
    formatter = LLMFormatter(output_format=OutputFormat.PLAIN_TEXT)
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_validation_result(result)

    assert 'Validation Result: leiden' in formatted, "Should have header"
    assert 'INVALID' in formatted, "Should show status"
    assert 'Missing Prerequisites' in formatted, "Should list prerequisites"
    assert 'Suggestions' in formatted, "Should have suggestions"
    assert '```' not in formatted, "Should not have markdown code blocks"
    print("✓ test_format_plain_text passed")


def test_format_json():
    """Test JSON formatting."""
    print("Testing JSON formatting...")
    formatter = LLMFormatter(output_format=OutputFormat.JSON)
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_validation_result(result)

    # Parse JSON to validate
    data = json.loads(formatted)

    assert data['function_name'] == 'leiden', "Should have function name"
    assert data['is_valid'] == False, "Should have valid status"
    assert len(data['missing_prerequisites']) == 2, "Should have prerequisites"
    assert len(data['suggestions']) == 2, "Should have suggestions"
    assert 'obsm' in data['missing_data_structures'], "Should have data structures"
    print("✓ test_format_json passed")


def test_create_agent_prompt():
    """Test agent prompt creation."""
    print("Testing agent prompt creation...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=False)

    prompt = formatter.create_agent_prompt(result, "Fix leiden validation errors")

    assert isinstance(prompt, LLMPrompt), "Should return LLMPrompt"
    assert len(prompt.system_prompt) > 0, "Should have system prompt"
    assert len(prompt.user_prompt) > 0, "Should have user prompt"
    assert 'Fix leiden validation errors' in prompt.user_prompt, "Should include task"
    assert 'leiden' in prompt.user_prompt, "Should include function name"
    assert len(prompt.suggestions) == 2, "Should have formatted suggestions"
    assert prompt.context['function_name'] == 'leiden', "Should have context"
    print("✓ test_create_agent_prompt passed")


def test_format_natural_language():
    """Test natural language formatting."""
    print("Testing natural language formatting...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_natural_language(result)

    assert '❌' in formatted, "Should have error emoji"
    assert 'Cannot run leiden' in formatted, "Should explain the problem"
    assert 'prerequisite function(s) first' in formatted, "Should mention prerequisites"
    assert 'CRITICAL' in formatted, "Should show priorities"
    assert '📋 Recommendations' in formatted, "Should have recommendations"
    print("✓ test_format_natural_language passed")


def test_format_natural_language_valid():
    """Test natural language formatting for valid result."""
    print("Testing natural language formatting (valid)...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=True)

    formatted = formatter.format_natural_language(result)

    assert '✅' in formatted, "Should have success emoji"
    assert 'All requirements are satisfied' in formatted, "Should show success"
    assert 'Detected' in formatted, "Should list detected functions"
    print("✓ test_format_natural_language_valid passed")


def test_format_suggestion():
    """Test individual suggestion formatting."""
    print("Testing suggestion formatting...")
    formatter = LLMFormatter()

    suggestion = Suggestion(
        priority='CRITICAL',
        suggestion_type='prerequisite',
        description='Run PCA',
        code='ov.pp.pca(adata, n_pcs=50)',
        explanation='PCA is required',
        estimated_time='30 seconds',
        estimated_time_seconds=30,
        prerequisites=[],
        impact='Generates PCA embeddings',
        auto_executable=False,
    )

    formatted = formatter.format_suggestion(suggestion)

    assert '[CRITICAL]' in formatted, "Should have priority"
    assert 'Run PCA' in formatted, "Should have description"
    assert '```python' in formatted, "Should have code block"
    assert 'ov.pp.pca' in formatted, "Should have code"
    assert 'Why:' in formatted, "Should have explanation"
    print("✓ test_format_suggestion passed")


def test_format_for_llm_agent_code_generator():
    """Test formatting for code generator agent."""
    print("Testing code generator agent formatting...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_for_llm_agent(result, "code_generator")

    assert 'task' in formatted, "Should have task"
    assert 'Generate executable Python code' in formatted['task'], "Should have generation task"
    assert 'context' in formatted, "Should have context"
    assert 'code_templates' in formatted, "Should have code templates"
    assert len(formatted['code_templates']) == 2, "Should have 2 code templates"
    assert 'constraints' in formatted, "Should have constraints"
    print("✓ test_format_for_llm_agent_code_generator passed")


def test_format_for_llm_agent_explainer():
    """Test formatting for explainer agent."""
    print("Testing explainer agent formatting...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_for_llm_agent(result, "explainer")

    assert 'task' in formatted, "Should have task"
    assert 'Explain what\'s needed' in formatted['task'], "Should have explanation task"
    assert 'explanation_points' in formatted, "Should have explanation points"
    assert 'suggestions' in formatted, "Should have suggestions"
    print("✓ test_format_for_llm_agent_explainer passed")


def test_format_for_llm_agent_debugger():
    """Test formatting for debugger agent."""
    print("Testing debugger agent formatting...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=False)

    formatted = formatter.format_for_llm_agent(result, "debugger")

    assert 'task' in formatted, "Should have task"
    assert 'Debug why' in formatted['task'], "Should have debug task"
    assert 'diagnostic_info' in formatted, "Should have diagnostic info"
    assert 'debug_steps' in formatted, "Should have debug steps"
    assert 'executed_functions' in formatted['diagnostic_info'], "Should have executed functions"
    print("✓ test_format_for_llm_agent_debugger passed")


def test_output_format_override():
    """Test output format override."""
    print("Testing output format override...")
    formatter = LLMFormatter(output_format=OutputFormat.MARKDOWN)
    result = create_test_validation_result(is_valid=False)

    # Override to plain text
    formatted = formatter.format_validation_result(result, format_override=OutputFormat.PLAIN_TEXT)

    assert '```' not in formatted, "Should not have markdown code blocks"
    assert 'Validation Result:' in formatted, "Should have plain text format"
    print("✓ test_output_format_override passed")


def test_prompt_to_dict():
    """Test LLMPrompt to_dict method."""
    print("Testing LLMPrompt to_dict...")
    formatter = LLMFormatter()
    result = create_test_validation_result(is_valid=False)

    prompt = formatter.create_agent_prompt(result)
    prompt_dict = prompt.to_dict()

    assert 'system' in prompt_dict, "Should have system key"
    assert 'user' in prompt_dict, "Should have user key"
    assert 'context' in prompt_dict, "Should have context key"
    assert 'suggestions' in prompt_dict, "Should have suggestions key"
    print("✓ test_prompt_to_dict passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Layer 2 Phase 4 - LLMFormatter Tests")
    print("=" * 60)
    print()

    tests = [
        test_llm_formatter_initialization,
        test_format_markdown,
        test_format_plain_text,
        test_format_json,
        test_create_agent_prompt,
        test_format_natural_language,
        test_format_natural_language_valid,
        test_format_suggestion,
        test_format_for_llm_agent_code_generator,
        test_format_for_llm_agent_explainer,
        test_format_for_llm_agent_debugger,
        test_output_format_override,
        test_prompt_to_dict,
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
        print("\n✅ All Phase 4 tests PASSED!")
        print("\nLLMFormatter Validation:")
        print("   ✓ Multiple output formats (Markdown, Plain Text, JSON, Prompt)")
        print("   ✓ Natural language explanations")
        print("   ✓ Agent-specific formatting (code generator, explainer, debugger)")
        print("   ✓ Prompt template creation")
        print("   ✓ Individual suggestion formatting")
        print("\nPhase 4 Status: ✅ COMPLETE")
        print("\nNext Steps:")
        print("   - Create comprehensive Phase 4 completion summary")
        print("   - Update documentation")
        print("   - Commit and push changes")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
