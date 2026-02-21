"""
Tests for the task complexity classifier in OmicVerseAgent.
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def test_complexity_classifier_pattern_matching_simple():
    """Test pattern-based classification for simple tasks."""
    from omicverse.utils.smart_agent import OmicVerseAgent

    # Create a minimal agent instance for testing
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Test simple tasks (should be classified by pattern matching, no LLM call)
    simple_requests = [
        "qc with nUMI>500",
        "quality control",
        "质控 nUMI>500",  # Chinese
        "normalize data",
        "run PCA",
        "leiden clustering",
        "just plot UMAP",
        "only filter cells",
    ]

    for request in simple_requests:
        result = asyncio.run(agent._analyze_task_complexity(request))
        assert result == 'simple', f"Expected 'simple' for '{request}', got '{result}'"
        print(f"✓ '{request}' -> {result}")


def test_complexity_classifier_pattern_matching_complex():
    """Test pattern-based classification for complex tasks."""
    from omicverse.utils.smart_agent import OmicVerseAgent

    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Test complex tasks (should be classified by pattern matching, no LLM call)
    complex_requests = [
        "complete bulk RNA-seq DEG analysis pipeline",
        "full preprocessing workflow",
        "comprehensive spatial deconvolution from start to finish",
        "entire single-cell analysis",
        "perform clustering and then generate visualizations",
        "do quality control followed by normalization and clustering",
    ]

    for request in complex_requests:
        result = asyncio.run(agent._analyze_task_complexity(request))
        assert result == 'complex', f"Expected 'complex' for '{request}', got '{result}'"
        print(f"✓ '{request}' -> {result}")


def test_complexity_classifier_keyword_detection():
    """Test that keyword detection works correctly."""
    from omicverse.utils.smart_agent import OmicVerseAgent

    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Test cases with expected results
    test_cases = [
        # (request, expected_complexity)
        ("qc nUMI>500", "simple"),  # function name detected
        ("quality control with mito<0.2", "simple"),  # function name
        ("normalize", "simple"),  # function name
        ("complete pipeline", "complex"),  # 'complete' + 'pipeline'
        ("full workflow analysis", "complex"),  # 'full' + 'workflow' + 'analysis'
        ("just normalize", "simple"),  # 'just' + function name
        ("only clustering", "simple"),  # 'only' + function name
    ]

    for request, expected in test_cases:
        result = asyncio.run(agent._analyze_task_complexity(request))
        assert result == expected, f"Expected '{expected}' for '{request}', got '{result}'"
        print(f"✓ '{request}' -> {result} (expected: {expected})")


def test_complexity_classifier_multilingual():
    """Test multilingual support (Chinese + English)."""
    from omicverse.utils.smart_agent import OmicVerseAgent

    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Chinese simple tasks
    chinese_simple = [
        "质控",
        "归一化",
        "降维",
        "聚类",
        "可视化",
    ]

    for request in chinese_simple:
        result = asyncio.run(agent._analyze_task_complexity(request))
        assert result == 'simple', f"Expected 'simple' for Chinese '{request}', got '{result}'"
        print(f"✓ '{request}' (Chinese) -> {result}")


def test_complexity_classifier_edge_cases():
    """Test edge cases and ambiguous requests."""
    from omicverse.utils.smart_agent import OmicVerseAgent

    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Edge cases - these may go to LLM or use defaults
    edge_cases = [
        "",  # empty
        "help",  # very short
        "analyze",  # vague
        "process my data",  # vague
    ]

    for request in edge_cases:
        result = asyncio.run(agent._analyze_task_complexity(request))
        # Just verify it returns a valid value, don't assert specific result
        assert result in ['simple', 'complex'], f"Invalid result '{result}' for '{request}'"
        print(f"✓ '{request}' -> {result}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Task Complexity Classifier")
    print("=" * 70)

    print("\n[TEST 1] Pattern Matching - Simple Tasks")
    test_complexity_classifier_pattern_matching_simple()

    print("\n[TEST 2] Pattern Matching - Complex Tasks")
    test_complexity_classifier_pattern_matching_complex()

    print("\n[TEST 3] Keyword Detection")
    test_complexity_classifier_keyword_detection()

    print("\n[TEST 4] Multilingual Support")
    test_complexity_classifier_multilingual()

    print("\n[TEST 5] Edge Cases")
    test_complexity_classifier_edge_cases()

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
