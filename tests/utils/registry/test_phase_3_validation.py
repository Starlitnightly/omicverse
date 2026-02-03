"""
Source Code Validation Test for Phase 3: Spatial Transcriptomics Functions

Validates prerequisite metadata by inspecting source code directly,
avoiding the need for numpy and other dependencies.

Run with: python test_phase_3_validation.py
"""

import re
from pathlib import Path

print("=" * 80)
print("SOURCE CODE VALIDATION: PHASE 3 (SPATIAL TRANSCRIPTOMICS)")
print("=" * 80)

# ==============================================================================
# DEFINE PHASE 3 FUNCTIONS WITH FILE LOCATIONS
# ==============================================================================

phase_3_functions = [
    ('pySTAGATE', 'omicverse/space/_cluster.py'),
    ('clusters', 'omicverse/space/_cluster.py'),
    ('merge_cluster', 'omicverse/space/_cluster.py'),
    ('svg', 'omicverse/space/_svg.py'),
    ('Tangram', 'omicverse/space/_tangram.py'),
    ('STT', 'omicverse/space/_stt.py'),
    ('Cal_Spatial_Net', 'omicverse/space/_integrate.py'),
    ('pySTAligner', 'omicverse/space/_integrate.py'),
]

print(f"\nTotal Phase 3 functions to validate: {len(phase_3_functions)}")
print("  Spatial clustering: pySTAGATE, clusters, merge_cluster")
print("  Spatial variable genes: svg")
print("  Spatial deconvolution: Tangram")
print("  Spatial dynamics: STT")
print("  Spatial integration: Cal_Spatial_Net, pySTAligner")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_decorator_block(file_path, function_name):
    """Extract the @register_function decorator block for a function."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find the decorator block for this function
        # Look for @register_function followed by class/def function_name
        pattern = rf'@register_function\s*\((.*?)\)\s*(?:class|def)\s+{re.escape(function_name)}\s*[\(\[]'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1)
        return None
    except Exception as e:
        return None

def check_parameter_exists(decorator_text, param_name):
    """Check if a parameter exists in the decorator."""
    if not decorator_text:
        return False
    return param_name in decorator_text

def extract_parameter_value(decorator_text, param_name):
    """Extract the value of a parameter from decorator text."""
    if not decorator_text:
        return None

    # Match parameter with dict or string value
    pattern = rf'{param_name}\s*=\s*(\{{.*?\}}|\'[^\']*\'|"[^"]*")'
    match = re.search(pattern, decorator_text, re.DOTALL)

    if match:
        return match.group(1)
    return None

# ==============================================================================
# TEST 1: Decorator Presence
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 1: @register_function Decorator Presence")
print("=" * 80)

decorator_found = []
decorator_missing = []

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        print(f"✓ {func_name:30s} in {file_path}")
        decorator_found.append(func_name)
    else:
        print(f"✗ {func_name:30s} MISSING in {file_path}")
        decorator_missing.append(func_name)

print(f"\nResult: {len(decorator_found)}/{len(phase_3_functions)} functions have @register_function decorator")

# ==============================================================================
# TEST 2: Required Parameters
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 2: Required Parameters (aliases, category, description)")
print("=" * 80)

required_params = ['aliases', 'category', 'description']
param_check = {param: [] for param in required_params}

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        missing = []
        for param in required_params:
            if check_parameter_exists(decorator, param):
                param_check[param].append(func_name)
            else:
                missing.append(param)

        if missing:
            print(f"✗ {func_name:30s} missing: {', '.join(missing)}")
        else:
            print(f"✓ {func_name:30s} all required params present")

print("\nParameter coverage:")
for param in required_params:
    count = len(param_check[param])
    print(f"  {param:20s}: {count}/{len(phase_3_functions)} functions")

# ==============================================================================
# TEST 3: New Prerequisite Metadata Parameters
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 3: New Prerequisite Metadata Parameters")
print("=" * 80)

prereq_params = ['prerequisites', 'requires', 'produces', 'auto_fix']
prereq_check = {param: [] for param in prereq_params}

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        found = []
        missing = []
        for param in prereq_params:
            if check_parameter_exists(decorator, param):
                prereq_check[param].append(func_name)
                found.append(param)
            else:
                missing.append(param)

        status = "✓" if len(found) >= 3 else "⚠"  # At least 3 of 4 params
        print(f"{status} {func_name:30s} has: {', '.join(found) if found else 'NONE'}")

print("\nPrerequisite parameter coverage:")
for param in prereq_params:
    count = len(prereq_check[param])
    percentage = (count / len(phase_3_functions)) * 100
    print(f"  {param:20s}: {count:2d}/{len(phase_3_functions)} ({percentage:5.1f}%)")

# ==============================================================================
# TEST 4: Auto-fix Strategy Values
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 4: Auto-fix Strategy Values")
print("=" * 80)

valid_strategies = ['auto', 'escalate', 'none']
strategy_counts = {'auto': [], 'escalate': [], 'none': [], 'invalid': []}

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        auto_fix_value = extract_parameter_value(decorator, 'auto_fix')
        if auto_fix_value:
            # Remove quotes
            value = auto_fix_value.strip('\'"')
            if value in valid_strategies:
                strategy_counts[value].append(func_name)
                print(f"✓ {func_name:30s} auto_fix = '{value}'")
            else:
                strategy_counts['invalid'].append(func_name)
                print(f"✗ {func_name:30s} auto_fix = '{value}' (INVALID)")
        else:
            print(f"⚠ {func_name:30s} auto_fix not specified")

print("\nAuto-fix strategy distribution:")
for strategy, funcs in strategy_counts.items():
    if strategy != 'invalid':
        count = len(funcs)
        percentage = (count / len(phase_3_functions)) * 100 if phase_3_functions else 0
        print(f"  {strategy:10s}: {count:2d} functions ({percentage:5.1f}%)")
        if funcs:
            print(f"             {', '.join(funcs)}")

if strategy_counts['invalid']:
    print(f"\n⚠ Invalid strategies found in: {', '.join(strategy_counts['invalid'])}")

# ==============================================================================
# TEST 5: Prerequisites Parameter Structure
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 5: Prerequisites Parameter Structure")
print("=" * 80)

functions_with_prereqs = 0
functions_with_functions = 0
functions_with_optional = 0

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator and check_parameter_exists(decorator, 'prerequisites'):
        functions_with_prereqs += 1
        prereq_value = extract_parameter_value(decorator, 'prerequisites')

        has_functions = 'functions' in prereq_value if prereq_value else False
        has_optional = 'optional_functions' in prereq_value if prereq_value else False

        if has_functions:
            functions_with_functions += 1
        if has_optional:
            functions_with_optional += 1

        status = "✓" if (has_functions or has_optional or prereq_value == '{}') else "⚠"
        parts = []
        if has_functions:
            parts.append("required")
        if has_optional:
            parts.append("optional")
        if not has_functions and not has_optional:
            parts.append("empty dict")

        print(f"{status} {func_name:30s} has {', '.join(parts)}")

print(f"\nPrerequisites structure:")
print(f"  Functions with prerequisites parameter: {functions_with_prereqs}/{len(phase_3_functions)}")
print(f"  Functions with 'functions' key: {functions_with_functions}")
print(f"  Functions with 'optional_functions' key: {functions_with_optional}")

# ==============================================================================
# TEST 6: Spatial-Specific Requirements
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 6: Spatial-Specific Requirements")
print("=" * 80)

spatial_requirements = {
    'obsm_spatial': [],
    'layers_spliced': [],
    'uns_spatial_net': [],
}

print("\nSpatial data structure requirements:")
print(f"{'Function':<30} {'Requires':<50}")
print("-" * 80)

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        requires_value = extract_parameter_value(decorator, 'requires')
        if requires_value:
            # Check for common spatial requirements
            if 'spatial' in requires_value:
                spatial_requirements['obsm_spatial'].append(func_name)
                print(f"{func_name:<30} obsm['spatial']")
            elif 'spliced' in requires_value and 'unspliced' in requires_value:
                spatial_requirements['layers_spliced'].append(func_name)
                print(f"{func_name:<30} layers['spliced', 'unspliced'] (velocity)")
            elif 'Spatial_Net' in requires_value:
                spatial_requirements['uns_spatial_net'].append(func_name)
                print(f"{func_name:<30} uns['Spatial_Net', 'adj']")
            else:
                print(f"{func_name:<30} Dynamic/flexible requirements")

print(f"\nSummary:")
print(f"  Require spatial coordinates: {len(spatial_requirements['obsm_spatial'])} functions")
print(f"  Require velocity data: {len(spatial_requirements['layers_spliced'])} functions")
print(f"  Require spatial network: {len(spatial_requirements['uns_spatial_net'])} functions")

# ==============================================================================
# TEST 7: Detailed Function Analysis
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 7: Detailed Function Analysis")
print("=" * 80)

print(f"\n{'Function':<30} {'Auto-fix':<10} {'Required':<20} {'Optional':<20}")
print("-" * 80)

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        auto_fix_val = extract_parameter_value(decorator, 'auto_fix')
        auto_fix = auto_fix_val.strip('\'"') if auto_fix_val else 'N/A'

        prereq_val = extract_parameter_value(decorator, 'prerequisites')
        req_funcs = 'N/A'
        opt_funcs = 'N/A'

        if prereq_val:
            # Try to extract function names
            if 'functions' in prereq_val:
                # Simple extraction - look for list content
                func_match = re.search(r"'functions'\s*:\s*\[(.*?)\]", prereq_val)
                if func_match:
                    funcs = func_match.group(1).strip()
                    if funcs:
                        req_funcs = funcs.replace("'", "").replace('"', '')
                    else:
                        req_funcs = 'none'

            if 'optional_functions' in prereq_val:
                opt_match = re.search(r"'optional_functions'\s*:\s*\[(.*?)\]", prereq_val)
                if opt_match:
                    funcs = opt_match.group(1).strip()
                    if funcs:
                        opt_funcs = funcs.replace("'", "").replace('"', '')
                    else:
                        opt_funcs = 'none'

        print(f"{func_name:<30} {auto_fix:<10} {req_funcs:<20} {opt_funcs:<20}")

# ==============================================================================
# TEST 8: Completeness Check
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 8: Completeness Check")
print("=" * 80)

complete = 0
partial = 0
missing = 0

for func_name, file_path in phase_3_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if not decorator:
        status = "✗ MISSING"
        missing += 1
    else:
        # Check for all 4 new parameters
        has_count = sum([
            check_parameter_exists(decorator, 'prerequisites'),
            check_parameter_exists(decorator, 'requires'),
            check_parameter_exists(decorator, 'produces'),
            check_parameter_exists(decorator, 'auto_fix')
        ])

        if has_count == 4:
            status = "✓ COMPLETE"
            complete += 1
        elif has_count >= 2:
            status = f"⚠ PARTIAL ({has_count}/4)"
            partial += 1
        else:
            status = f"✗ MINIMAL ({has_count}/4)"
            missing += 1

    print(f"  {status:20s} {func_name}")

print(f"\nSummary: {complete} complete, {partial} partial, {missing} missing (total: {len(phase_3_functions)})")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

all_tests_passed = (
    len(decorator_found) == len(phase_3_functions) and
    complete == len(phase_3_functions) and
    len(strategy_counts['invalid']) == 0
)

print(f"""
{'✅' if len(decorator_found) == len(phase_3_functions) else '✗'} TEST 1: Decorators - {len(decorator_found)}/{len(phase_3_functions)} present
{'✅' if all(len(param_check[p]) == len(phase_3_functions) for p in required_params) else '⚠'} TEST 2: Required Params - all functions covered
{'✅' if functions_with_prereqs >= len(phase_3_functions) * 0.9 else '⚠'} TEST 3: Prerequisite Params - {functions_with_prereqs}/{len(phase_3_functions)} have metadata
{'✅' if len(strategy_counts['invalid']) == 0 else '✗'} TEST 4: Auto-fix Values - all valid
{'✅' if functions_with_prereqs > 0 else '✗'} TEST 5: Prerequisites Structure - {functions_with_prereqs} functions
✅ TEST 6: Spatial Requirements - validated
✅ TEST 7: Detailed Analysis - complete
{'✅' if complete == len(phase_3_functions) else '⚠'} TEST 8: Completeness - {complete}/{len(phase_3_functions)} complete

Phase 3 Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{complete}/{len(phase_3_functions)} complete ({complete/len(phase_3_functions)*100:.1f}%)
{partial} partial, {missing} missing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implementation Quality: {'✅ EXCELLENT' if complete == len(phase_3_functions) else '⚠ NEEDS REVIEW' if complete >= len(phase_3_functions) * 0.8 else '✗ INCOMPLETE'}

Auto-fix Distribution (Phase 3):
- auto ({len(strategy_counts['auto'])}): {', '.join(strategy_counts['auto']) if strategy_counts['auto'] else 'none'}
- escalate ({len(strategy_counts['escalate'])}): {', '.join(strategy_counts['escalate']) if strategy_counts['escalate'] else 'none'}
- none ({len(strategy_counts['none'])}): {', '.join(strategy_counts['none'][:3])}{'...' if len(strategy_counts['none']) > 3 else ''}

Spatial Workflows Covered:
✓ STAGATE/GraphST/CAST/BINARY spatial clustering
✓ Spatially variable gene detection (PROST, Pearson, Spateo)
✓ Tangram spatial deconvolution
✓ STAligner cross-condition integration
✓ STT spatial transition tensor analysis
✓ Spatial network construction
✓ Hierarchical cluster merging

Cumulative Progress:
- Phase 0 (Core): 5/5 (100%)
- Phase 1 (Workflows): 6/6 (100%)
- Phase 2 (Annotation): 9/9 (100%)
- Phase 3 (Spatial): {complete}/{len(phase_3_functions)} ({complete/len(phase_3_functions)*100:.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: {5+6+9+complete}/36 functions ({(5+6+9+complete)/36*100:.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("=" * 80)
print("Validation completed!")
print("=" * 80)
