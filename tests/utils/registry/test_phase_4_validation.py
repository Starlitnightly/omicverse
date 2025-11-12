"""
Phase 4 Validation Test - Specialized Functions (8 functions)

This test validates that all Phase 4 specialized functions have complete
prerequisite metadata including prerequisites, requires, produces, and auto_fix.

Phase 4 covers:
- Utility functions: mde, cluster, refine_label, weighted_knn_transfer
- Bulk functions: batch_correction
- Spatial functions: pySpaceFlow, GASTON
- Single-cell functions: DCT
"""

import re
import sys


def extract_register_function_metadata(file_path, function_name):
    """Extract @register_function metadata for a given function/class."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Pattern to match @register_function decorator before a function or class
        pattern = rf'@register_function\s*\((.*?)\)\s*(?:class|def)\s+{re.escape(function_name)}'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            decorator_content = match.group(1)
            return decorator_content
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def validate_metadata(decorator_content, function_name):
    """Validate that all required metadata fields are present."""
    if not decorator_content:
        return False, "Decorator not found"

    required_fields = ['prerequisites', 'requires', 'produces', 'auto_fix']
    missing_fields = []

    for field in required_fields:
        # Check if field exists in decorator
        pattern = rf"{field}\s*="
        if not re.search(pattern, decorator_content):
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"

    # Validate auto_fix value
    auto_fix_match = re.search(r"auto_fix\s*=\s*['\"](\w+)['\"]", decorator_content)
    if auto_fix_match:
        auto_fix_value = auto_fix_match.group(1)
        if auto_fix_value not in ['auto', 'escalate', 'none']:
            return False, f"Invalid auto_fix value: {auto_fix_value}"

    return True, "Complete"


def main():
    """Test all Phase 4 functions for complete prerequisite metadata."""

    # Phase 4 functions: Specialized functions (8 total)
    phase_4_functions = [
        ('mde', 'omicverse/utils/_mde.py'),
        ('cluster', 'omicverse/utils/_cluster.py'),
        ('refine_label', 'omicverse/utils/_cluster.py'),
        ('weighted_knn_transfer', 'omicverse/utils/_knn.py'),
        ('batch_correction', 'omicverse/bulk/_combat.py'),
        ('pySpaceFlow', 'omicverse/space/_spaceflow.py'),
        ('GASTON', 'omicverse/space/_gaston.py'),
        ('DCT', 'omicverse/single/_deg_ct.py'),
    ]

    print("=" * 80)
    print("PHASE 4 VALIDATION TEST - Specialized Functions")
    print("=" * 80)
    print()

    results = []
    auto_fix_counts = {'auto': 0, 'escalate': 0, 'none': 0}

    for func_name, file_path in phase_4_functions:
        print(f"Testing {func_name} ({file_path})...")

        decorator_content = extract_register_function_metadata(file_path, func_name)
        is_valid, message = validate_metadata(decorator_content, func_name)

        results.append((func_name, is_valid, message))

        if is_valid:
            print(f"  ✓ {message}")
            # Count auto_fix strategy
            auto_fix_match = re.search(r"auto_fix\s*=\s*['\"](\w+)['\"]", decorator_content)
            if auto_fix_match:
                auto_fix_counts[auto_fix_match.group(1)] += 1
        else:
            print(f"  ✗ {message}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(phase_4_functions)
    complete = sum(1 for _, is_valid, _ in results if is_valid)

    print(f"\nPhase 4 Functions: {complete}/{total} complete ({100*complete//total}%)")
    print(f"\nAuto-fix Strategy Distribution:")
    print(f"  - auto:     {auto_fix_counts['auto']} functions")
    print(f"  - escalate: {auto_fix_counts['escalate']} functions")
    print(f"  - none:     {auto_fix_counts['none']} functions")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for func_name, is_valid, message in results:
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status:8} | {func_name:30} | {message}")

    # Function category breakdown
    print("\n" + "=" * 80)
    print("PHASE 4 BREAKDOWN BY CATEGORY")
    print("=" * 80)

    categories = {
        'Utility Functions': ['mde', 'cluster', 'refine_label', 'weighted_knn_transfer'],
        'Bulk Functions': ['batch_correction'],
        'Spatial Functions': ['pySpaceFlow', 'GASTON'],
        'Single-cell Functions': ['DCT'],
    }

    for category, funcs in categories.items():
        category_results = [(f, v, m) for f, v, m in results if f in funcs]
        category_complete = sum(1 for _, v, _ in category_results if v)
        category_total = len(category_results)
        print(f"\n{category}: {category_complete}/{category_total} complete")
        for func_name, is_valid, _ in category_results:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {func_name}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("PHASE 4 ASSESSMENT")
    print("=" * 80)

    if complete == total:
        print("\n✓ Phase 4 is COMPLETE! All specialized functions have prerequisite metadata.")
        print("\n  Key features validated:")
        print("    - Function prerequisite chains")
        print("    - Data structure requirements")
        print("    - Output specifications")
        print("    - Auto-fix strategies")
        print("\n  Coverage: Specialized utility, bulk, spatial, and single-cell functions")
        return 0
    else:
        print(f"\n✗ Phase 4 is INCOMPLETE. {total - complete} functions need metadata.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
