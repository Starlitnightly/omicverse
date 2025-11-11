"""
Usage examples for DataStateInspector Production API.

This module contains comprehensive examples demonstrating all features
of the DataStateInspector in real-world scenarios.
"""

from anndata import AnnData
import numpy as np
import pandas as pd


def example_basic_validation():
    """Example 1: Basic validation with quick validation function.

    Demonstrates the simplest way to validate prerequisites.
    """
    print("=== Example 1: Basic Validation ===\n")

    # Create sample AnnData
    adata = _create_sample_adata()

    # Quick validation
    from omicverse.utils.inspector import validate_function

    result = validate_function(adata, 'leiden')

    if result.is_valid:
        print("✓ All prerequisites satisfied for leiden!")
    else:
        print(f"✗ Prerequisites missing for leiden:")
        print(f"  Missing prerequisites: {result.missing_prerequisites}")
        print(f"  Missing data: {result.missing_data_structures}")
        print(f"\nSuggestions ({len(result.suggestions)}):")
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            print(f"  {i}. [{suggestion.priority}] {suggestion.description}")
            print(f"     Code: {suggestion.code}")


def example_inspector_creation():
    """Example 2: Creating and using DataStateInspector.

    Demonstrates creating an inspector and performing multiple validations.
    """
    print("\n=== Example 2: Inspector Creation ===\n")

    from omicverse.utils.inspector import create_inspector

    # Create sample data
    adata = _create_sample_adata()

    # Create inspector (automatically loads registry and caches)
    inspector = create_inspector(adata)

    # Validate multiple functions
    functions_to_check = ['pca', 'neighbors', 'leiden', 'umap']

    for func in functions_to_check:
        result = inspector.validate_prerequisites(func)
        status = "✓" if result.is_valid else "✗"
        print(f"{status} {func}: {result.message}")


def example_natural_language_explanation():
    """Example 3: Getting natural language explanations.

    Demonstrates user-friendly output for non-technical users.
    """
    print("\n=== Example 3: Natural Language Explanation ===\n")

    from omicverse.utils.inspector import explain_requirements

    adata = _create_sample_adata()

    # Get natural language explanation
    explanation = explain_requirements(
        adata,
        'leiden',
        format='natural'
    )

    print(explanation)


def example_workflow_suggestions():
    """Example 4: Getting workflow suggestions.

    Demonstrates automatic workflow planning with dependency resolution.
    """
    print("\n=== Example 4: Workflow Suggestions ===\n")

    from omicverse.utils.inspector import get_workflow_suggestions

    adata = _create_sample_adata()

    # Get workflow plan with recommended strategy
    workflow = get_workflow_suggestions(
        adata,
        'leiden',
        strategy='recommended'
    )

    print(f"Workflow Plan: {workflow['strategy']} strategy")
    print(f"Total steps: {workflow['total_steps']}")
    print(f"Estimated time: {workflow['estimated_time']}\n")

    for step in workflow['steps']:
        print(f"Step {step['order']}: {step['function']}")
        print(f"  Description: {step['description']}")
        print(f"  Code: {step['code']}")
        print(f"  Time: {step['time']}")
        print()


def example_decorator():
    """Example 5: Using the prerequisite validation decorator.

    Demonstrates automatic prerequisite checking before function execution.
    """
    print("\n=== Example 5: Decorator Usage ===\n")

    from omicverse.utils.inspector import check_prerequisites

    # Define a function with prerequisite checking
    @check_prerequisites('leiden', raise_on_invalid=False)
    def my_leiden_clustering(adata, resolution=1.0):
        """Custom leiden clustering with validation."""
        print(f"Running leiden clustering with resolution={resolution}")
        # Your clustering implementation here
        adata.obs['leiden'] = np.random.randint(0, 5, adata.n_obs)
        return adata

    adata = _create_sample_adata()

    # This will validate prerequisites before executing
    try:
        result_adata = my_leiden_clustering(adata, resolution=1.5)
        print("✓ Function executed successfully")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")


def example_context_manager():
    """Example 6: Using ValidationContext context manager.

    Demonstrates safe prerequisite validation with context managers.
    """
    print("\n=== Example 6: Context Manager ===\n")

    from omicverse.utils.inspector import ValidationContext

    adata = _create_sample_adata()

    # Use context manager for validation
    with ValidationContext(adata, 'leiden', raise_on_invalid=False) as ctx:
        if ctx.is_valid:
            print("✓ Prerequisites satisfied - running leiden")
            # Run clustering
            adata.obs['leiden'] = np.random.randint(0, 5, adata.n_obs)
        else:
            print("✗ Prerequisites not satisfied")
            print(f"  Message: {ctx.result.message}")
            print(f"  Suggestions: {len(ctx.result.suggestions)} available")


def example_batch_validation():
    """Example 7: Batch validation of multiple functions.

    Demonstrates validating multiple functions at once.
    """
    print("\n=== Example 7: Batch Validation ===\n")

    from omicverse.utils.inspector import batch_validate

    adata = _create_sample_adata()

    # Validate multiple functions at once
    functions = ['qc', 'preprocess', 'pca', 'neighbors', 'leiden', 'umap']
    results = batch_validate(adata, functions)

    print("Batch Validation Results:")
    for func, result in results.items():
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {func}")


def example_validation_report():
    """Example 8: Generate validation reports.

    Demonstrates generating formatted reports for analysis state.
    """
    print("\n=== Example 8: Validation Report ===\n")

    from omicverse.utils.inspector import get_validation_report

    adata = _create_sample_adata()

    # Generate markdown report
    report = get_validation_report(
        adata,
        function_names=['qc', 'preprocess', 'pca', 'neighbors', 'leiden'],
        format='markdown'
    )

    print(report)


def example_llm_formatting():
    """Example 9: LLM-friendly output formats.

    Demonstrates different output formats for LLM consumption.
    """
    print("\n=== Example 9: LLM Formatting ===\n")

    from omicverse.utils.inspector import create_inspector
    from omicverse.utils.inspector import OutputFormat

    adata = _create_sample_adata()
    inspector = create_inspector(adata)

    result = inspector.validate_prerequisites('leiden')

    # Format as markdown
    print("--- Markdown Format ---")
    markdown = inspector.format_for_llm(result, OutputFormat.MARKDOWN)
    print(markdown[:300] + "...\n")

    # Format as JSON
    print("--- JSON Format ---")
    json_output = inspector.format_for_llm(result, OutputFormat.JSON)
    print(json_output[:300] + "...\n")

    # Get LLM prompt
    print("--- LLM Prompt ---")
    prompt = inspector.get_llm_prompt('leiden', "Fix the preprocessing issues")
    print(f"System: {prompt.system_prompt[:100]}...")
    print(f"User: {prompt.user_prompt[:100]}...")


def example_agent_formatting():
    """Example 10: Agent-specific formatting.

    Demonstrates formatting for different types of AI agents.
    """
    print("\n=== Example 10: Agent-Specific Formatting ===\n")

    from omicverse.utils.inspector import create_inspector

    adata = _create_sample_adata()
    inspector = create_inspector(adata)

    result = inspector.validate_prerequisites('leiden')

    # Format for code generator agent
    code_gen_format = inspector.llm_formatter.format_for_llm_agent(
        result,
        agent_type='code_generator'
    )
    print("Code Generator Format:")
    print(f"  Task: {code_gen_format['task'][:60]}...")
    print(f"  Templates: {len(code_gen_format['code_templates'])} available")

    # Format for explainer agent
    explainer_format = inspector.llm_formatter.format_for_llm_agent(
        result,
        agent_type='explainer'
    )
    print("\nExplainer Format:")
    print(f"  Task: {explainer_format['task'][:60]}...")
    print(f"  Points: {len(explainer_format['explanation_points'])} explanation points")

    # Format for debugger agent
    debugger_format = inspector.llm_formatter.format_for_llm_agent(
        result,
        agent_type='debugger'
    )
    print("\nDebugger Format:")
    print(f"  Task: {debugger_format['task'][:60]}...")
    print(f"  Debug Steps: {len(debugger_format['debug_steps'])} steps")


def example_caching():
    """Example 11: Inspector caching for performance.

    Demonstrates caching behavior for improved performance.
    """
    print("\n=== Example 11: Inspector Caching ===\n")

    from omicverse.utils.inspector import create_inspector, clear_inspector_cache
    import time

    adata = _create_sample_adata()

    # First call - creates new inspector
    start = time.time()
    inspector1 = create_inspector(adata, cache=True)
    time1 = time.time() - start

    # Second call - retrieves from cache
    start = time.time()
    inspector2 = create_inspector(adata, cache=True)
    time2 = time.time() - start

    print(f"First call (create): {time1*1000:.2f}ms")
    print(f"Second call (cached): {time2*1000:.2f}ms")
    print(f"Same instance: {inspector1 is inspector2}")

    # Clear cache
    clear_inspector_cache()
    print("\n✓ Cache cleared")


def example_integration_workflow():
    """Example 12: Complete integration workflow.

    Demonstrates a full analysis workflow with validation at each step.
    """
    print("\n=== Example 12: Complete Integration Workflow ===\n")

    from omicverse.utils.inspector import create_inspector

    # Create sample data
    adata = _create_sample_adata()
    inspector = create_inspector(adata)

    # Define analysis workflow
    workflow = [
        ('qc', 'Quality control'),
        ('preprocess', 'Preprocessing'),
        ('pca', 'PCA reduction'),
        ('neighbors', 'Neighbor graph'),
        ('leiden', 'Clustering'),
        ('umap', 'UMAP embedding'),
    ]

    print("Analysis Workflow Validation:\n")

    for func_name, description in workflow:
        result = inspector.validate_prerequisites(func_name)

        print(f"Step: {description} ({func_name})")

        if result.is_valid:
            print(f"  ✓ Ready to execute")
            # Here you would actually execute the function
            # e.g., ov.pp.{func_name}(adata)
        else:
            print(f"  ✗ Not ready - missing:")
            if result.missing_prerequisites:
                print(f"    Prerequisites: {', '.join(result.missing_prerequisites)}")
            if result.missing_data_structures:
                print(f"    Data: {result.missing_data_structures}")

            # Show first suggestion
            if result.suggestions:
                print(f"  💡 Suggestion: {result.suggestions[0].description}")

        print()


# Helper function to create sample AnnData
def _create_sample_adata(n_obs=100, n_vars=50):
    """Create a sample AnnData object for examples."""
    np.random.seed(42)

    # Create random expression matrix
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))

    # Create AnnData
    adata = AnnData(X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    # Add some basic metadata
    adata.obs['n_counts'] = X.sum(axis=1)
    adata.obs['n_genes'] = (X > 0).sum(axis=1)

    return adata


def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        example_basic_validation,
        example_inspector_creation,
        example_natural_language_explanation,
        example_workflow_suggestions,
        example_decorator,
        example_context_manager,
        example_batch_validation,
        example_validation_report,
        example_llm_formatting,
        example_agent_formatting,
        example_caching,
        example_integration_workflow,
    ]

    for example in examples:
        try:
            example()
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("DataStateInspector Usage Examples")
    print("="*60)
    print()

    run_all_examples()

    print("\nAll examples completed!")
