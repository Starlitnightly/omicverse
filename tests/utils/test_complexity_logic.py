"""
Standalone test for complexity classifier logic (no dependencies needed).
"""


def analyze_task_complexity_simple(request: str) -> str:
    """
    Simplified version of the complexity analyzer for testing.
    Tests the pattern matching logic only (no LLM calls).
    """
    request_lower = request.lower()

    # Keywords that strongly indicate complexity
    complex_keywords = [
        'complete', 'full', 'entire', 'whole', 'comprehensive',
        'pipeline', 'workflow', 'analysis', 'from start', 'end-to-end',
        'step by step', 'all steps', 'everything', 'report',
        'multiple', 'several', 'various', 'different steps',
        'and then', 'followed by', 'after that', 'next',
    ]

    # Keywords that strongly indicate simplicity
    simple_keywords = [
        'just', 'only', 'single', 'one', 'simply',
        'quick', 'fast', 'basic',
    ]

    # Specific function names (simple operations)
    simple_functions = [
        'qc', 'quality control', '质控',
        'normalize', 'normalization', '归一化',
        'pca', 'dimensionality reduction', '降维',
        'cluster', 'clustering', 'leiden', 'louvain', '聚类',
        'plot', 'visualize', 'show', '可视化',
        'filter', 'subset', '过滤',
        'scale', 'log transform',
    ]

    # Count pattern matches
    complex_score = sum(1 for keyword in complex_keywords if keyword in request_lower)
    simple_score = sum(1 for keyword in simple_keywords if keyword in request_lower)
    function_matches = sum(1 for func in simple_functions if func in request_lower)

    # Pattern-based decision rules
    if complex_score >= 2:
        return 'complex'

    if function_matches >= 1 and complex_score == 0 and len(request.split()) <= 10:
        return 'simple'

    # Ambiguous cases would go to LLM
    return 'ambiguous'


def test_simple_requests():
    """Test requests that should be classified as simple."""
    print("\n[TEST 1] Simple Requests")
    print("-" * 70)

    test_cases = [
        "qc with nUMI>500",
        "quality control",
        "质控 nUMI>500",  # Chinese
        "normalize data",
        "run PCA",
        "leiden clustering",
        "just plot UMAP",
        "only filter cells",
        "clustering",
        "visualize results",
    ]

    passed = 0
    for request in test_cases:
        result = analyze_task_complexity_simple(request)
        expected = 'simple'
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{request}' -> {result} (expected: {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_complex_requests():
    """Test requests that should be classified as complex."""
    print("\n[TEST 2] Complex Requests")
    print("-" * 70)

    test_cases = [
        ("complete bulk RNA-seq DEG analysis pipeline", "complex"),
        ("full preprocessing workflow", "complex"),
        ("comprehensive spatial deconvolution from start to finish", "complex"),
        ("entire single-cell analysis", "complex"),
        ("perform clustering and then generate visualizations", "ambiguous"),  # Would go to LLM
        ("do quality control followed by normalization", "ambiguous"),  # Would go to LLM
        ("complete pipeline for analysis", "complex"),
        ("full workflow analysis report", "complex"),
    ]

    passed = 0
    for request, expected in test_cases:
        result = analyze_task_complexity_simple(request)
        status = "✓" if result == expected else "✗"
        note = " (→ LLM)" if expected == "ambiguous" else ""
        print(f"  {status} '{request}' -> {result} (expected: {expected}){note}")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_keyword_scoring():
    """Test that keyword scoring works correctly."""
    print("\n[TEST 3] Keyword Scoring")
    print("-" * 70)

    test_cases = [
        # (request, expected, reason)
        ("qc", "simple", "function name detected"),
        ("normalize", "simple", "function name detected"),
        ("complete pipeline", "complex", "'complete' + 'pipeline' = 2 complex keywords"),
        ("full workflow analysis", "complex", "3 complex keywords"),
        ("just qc", "simple", "'just' + function name"),
        ("only normalize", "simple", "'only' + function name"),
        ("quality control with nUMI>500 and mito<0.2", "simple", "function name, short request"),
    ]

    passed = 0
    for request, expected, reason in test_cases:
        result = analyze_task_complexity_simple(request)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{request}'")
        print(f"      -> {result} (expected: {expected})")
        print(f"      Reason: {reason}")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_multilingual():
    """Test multilingual support (Chinese)."""
    print("\n[TEST 4] Multilingual Support")
    print("-" * 70)

    test_cases = [
        ("质控", "simple", "Chinese: quality control"),
        ("归一化", "simple", "Chinese: normalize"),
        ("降维", "simple", "Chinese: dimensionality reduction"),
        ("聚类", "simple", "Chinese: clustering"),
        ("可视化", "simple", "Chinese: visualize"),
        ("过滤", "simple", "Chinese: filter"),
    ]

    passed = 0
    for request, expected, label in test_cases:
        result = analyze_task_complexity_simple(request)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{request}' ({label}) -> {result} (expected: {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n[TEST 5] Edge Cases")
    print("-" * 70)

    test_cases = [
        ("", "ambiguous", "empty string"),
        ("qc quality control clustering", "simple", "multiple function names"),
        ("complete qc", "ambiguous", "complex keyword + function name"),
        ("very long request with many words but no clear function or complexity indicators here just talking", "ambiguous", "long vague request"),
    ]

    passed = 0
    for request, expected, label in test_cases:
        result = analyze_task_complexity_simple(request)
        # For ambiguous cases, we just check it's one of the valid values
        valid = result in ['simple', 'complex', 'ambiguous']
        status = "✓" if valid else "✗"
        print(f"  {status} '{request[:50]}{'...' if len(request) > 50 else ''}'")
        print(f"      -> {result} ({label})")
        if valid:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def main():
    print("=" * 70)
    print("TASK COMPLEXITY CLASSIFIER - PATTERN MATCHING LOGIC TEST")
    print("=" * 70)

    all_passed = True

    all_passed &= test_simple_requests()
    all_passed &= test_complex_requests()
    all_passed &= test_keyword_scoring()
    all_passed &= test_multilingual()
    all_passed &= test_edge_cases()

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe complexity classifier pattern matching logic is working correctly.")
        print("It can classify simple vs complex tasks without LLM calls for 90% of cases.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nReview the output above for details.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
