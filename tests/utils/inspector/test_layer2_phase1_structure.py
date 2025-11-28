"""
Dependency-free structural validation for Layer 2 Phase 1 implementation.

This script validates the implementation structure without requiring
numpy, scipy, or anndata dependencies.
"""

import sys
import inspect
from typing import get_type_hints


def test_module_imports():
    """Test that all modules can be imported."""
    print("\n=== Testing Module Imports ===")

    try:
        from omicverse.utils.inspector import data_structures
        print("✓ data_structures module imported")
    except ImportError as e:
        print(f"✗ Failed to import data_structures: {e}")
        return False

    try:
        from omicverse.utils.inspector import validators
        print("✓ validators module imported")
    except ImportError as e:
        print(f"✗ Failed to import validators: {e}")
        return False

    try:
        from omicverse.utils.inspector import inspector
        print("✓ inspector module imported")
    except ImportError as e:
        print(f"✗ Failed to import inspector: {e}")
        return False

    try:
        from omicverse.utils.inspector import (
            DataStateInspector,
            DataValidators,
            ValidationResult,
            DataCheckResult,
        )
        print("✓ Main classes imported from package")
    except ImportError as e:
        print(f"✗ Failed to import main classes: {e}")
        return False

    return True


def test_data_structures():
    """Test that all data structure classes are defined correctly."""
    print("\n=== Testing Data Structures ===")

    try:
        from omicverse.utils.inspector.data_structures import (
            ValidationResult,
            DataCheckResult,
            ObsCheckResult,
            ObsmCheckResult,
            ObspCheckResult,
            UnsCheckResult,
            LayersCheckResult,
            Suggestion,
            ExecutionEvidence,
            ExecutionState,
        )

        classes = [
            ValidationResult,
            DataCheckResult,
            ObsCheckResult,
            ObsmCheckResult,
            ObspCheckResult,
            UnsCheckResult,
            LayersCheckResult,
            Suggestion,
            ExecutionEvidence,
            ExecutionState,
        ]

        for cls in classes:
            print(f"✓ {cls.__name__} class defined")

            # Check if it's a dataclass
            if hasattr(cls, '__dataclass_fields__'):
                print(f"  - Is a dataclass with {len(cls.__dataclass_fields__)} fields")

        # Test ValidationResult has get_summary method
        if hasattr(ValidationResult, 'get_summary'):
            print("✓ ValidationResult.get_summary() method exists")
        else:
            print("✗ ValidationResult.get_summary() method missing")
            return False

        # Test DataCheckResult has all_missing_structures property
        if hasattr(DataCheckResult, 'all_missing_structures'):
            print("✓ DataCheckResult.all_missing_structures property exists")
        else:
            print("✗ DataCheckResult.all_missing_structures property missing")
            return False

        return True

    except ImportError as e:
        print(f"✗ Failed to import data structures: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing data structures: {e}")
        return False


def test_validators_class():
    """Test that DataValidators class is defined correctly."""
    print("\n=== Testing DataValidators Class ===")

    try:
        from omicverse.utils.inspector.validators import DataValidators

        # Check required methods exist
        required_methods = [
            'check_obs',
            'check_obsm',
            'check_obsp',
            'check_uns',
            'check_layers',
            'check_all_requirements',
        ]

        for method_name in required_methods:
            if hasattr(DataValidators, method_name):
                method = getattr(DataValidators, method_name)
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                print(f"✓ DataValidators.{method_name}() exists with params: {params}")
            else:
                print(f"✗ DataValidators.{method_name}() missing")
                return False

        # Check __init__ method
        if hasattr(DataValidators, '__init__'):
            sig = inspect.signature(DataValidators.__init__)
            params = list(sig.parameters.keys())
            if 'adata' in params:
                print(f"✓ DataValidators.__init__() has 'adata' parameter")
            else:
                print(f"✗ DataValidators.__init__() missing 'adata' parameter")
                return False

        return True

    except ImportError as e:
        print(f"✗ Failed to import DataValidators: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing DataValidators: {e}")
        return False


def test_inspector_class():
    """Test that DataStateInspector class is defined correctly."""
    print("\n=== Testing DataStateInspector Class ===")

    try:
        from omicverse.utils.inspector.inspector import DataStateInspector

        # Check required methods exist
        required_methods = [
            'validate_prerequisites',
            'check_data_requirements',
            'get_validation_summary',
        ]

        for method_name in required_methods:
            if hasattr(DataStateInspector, method_name):
                method = getattr(DataStateInspector, method_name)
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                print(f"✓ DataStateInspector.{method_name}() exists with params: {params}")
            else:
                print(f"✗ DataStateInspector.{method_name}() missing")
                return False

        # Check __init__ method
        if hasattr(DataStateInspector, '__init__'):
            sig = inspect.signature(DataStateInspector.__init__)
            params = list(sig.parameters.keys())
            if 'adata' in params and 'registry' in params:
                print(f"✓ DataStateInspector.__init__() has required parameters")
            else:
                print(f"✗ DataStateInspector.__init__() missing required parameters")
                return False

        # Check private methods exist
        private_methods = [
            '_get_function_metadata',
            '_generate_data_suggestions',
        ]

        for method_name in private_methods:
            if hasattr(DataStateInspector, method_name):
                print(f"✓ DataStateInspector.{method_name}() helper method exists")
            else:
                print(f"✗ DataStateInspector.{method_name}() helper method missing")
                return False

        return True

    except ImportError as e:
        print(f"✗ Failed to import DataStateInspector: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing DataStateInspector: {e}")
        return False


def test_package_exports():
    """Test that package exports are correct."""
    print("\n=== Testing Package Exports ===")

    try:
        import omicverse.utils.inspector as inspector_module

        # Check __all__ is defined
        if hasattr(inspector_module, '__all__'):
            print(f"✓ __all__ defined with {len(inspector_module.__all__)} exports")

            # Verify key exports
            required_exports = [
                'DataStateInspector',
                'DataValidators',
                'ValidationResult',
                'DataCheckResult',
            ]

            for export in required_exports:
                if export in inspector_module.__all__:
                    print(f"  ✓ '{export}' in __all__")
                else:
                    print(f"  ✗ '{export}' missing from __all__")
                    return False
        else:
            print("✗ __all__ not defined in package")
            return False

        # Check version is defined
        if hasattr(inspector_module, '__version__'):
            print(f"✓ __version__ defined: {inspector_module.__version__}")
        else:
            print("✗ __version__ not defined")
            return False

        return True

    except ImportError as e:
        print(f"✗ Failed to import inspector package: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing package exports: {e}")
        return False


def test_documentation():
    """Test that documentation exists."""
    print("\n=== Testing Documentation ===")

    import os

    inspector_dir = 'omicverse/utils/inspector'

    # Check README exists
    readme_path = os.path.join(inspector_dir, 'README.md')
    if os.path.exists(readme_path):
        size = os.path.getsize(readme_path)
        print(f"✓ README.md exists ({size} bytes)")
    else:
        print("✗ README.md missing")
        return False

    # Check test directory exists
    test_dir = os.path.join(inspector_dir, 'tests')
    if os.path.exists(test_dir):
        print(f"✓ tests/ directory exists")

        # Check test file exists
        test_file = os.path.join(test_dir, 'test_validators.py')
        if os.path.exists(test_file):
            size = os.path.getsize(test_file)
            print(f"✓ test_validators.py exists ({size} bytes)")
        else:
            print("✗ test_validators.py missing")
            return False
    else:
        print("✗ tests/ directory missing")
        return False

    # Check module docstrings
    try:
        from omicverse.utils.inspector import data_structures, validators, inspector

        for module, name in [(data_structures, 'data_structures'),
                              (validators, 'validators'),
                              (inspector, 'inspector')]:
            if module.__doc__:
                print(f"✓ {name}.py has module docstring")
            else:
                print(f"✗ {name}.py missing module docstring")

        return True

    except Exception as e:
        print(f"✗ Error checking docstrings: {e}")
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\n=== Testing File Structure ===")

    import os

    inspector_dir = 'omicverse/utils/inspector'

    expected_files = [
        '__init__.py',
        'data_structures.py',
        'validators.py',
        'inspector.py',
        'README.md',
        'tests/__init__.py',
        'tests/test_validators.py',
    ]

    all_exist = True
    for file in expected_files:
        path = os.path.join(inspector_dir, file)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ {file} exists ({size} bytes)")
        else:
            print(f"✗ {file} missing")
            all_exist = False

    return all_exist


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Layer 2 Phase 1 - Structural Validation")
    print("=" * 60)

    results = {
        'File Structure': test_file_structure(),
        'Module Imports': test_module_imports(),
        'Data Structures': test_data_structures(),
        'DataValidators Class': test_validators_class(),
        'DataStateInspector Class': test_inspector_class(),
        'Package Exports': test_package_exports(),
        'Documentation': test_documentation(),
    }

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:12} {test_name}")

    total = len(results)
    passed = sum(results.values())

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print("=" * 60)

    if passed == total:
        print("\n✅ Layer 2 Phase 1 structure validation PASSED!")
        print("   All components are correctly defined and structured.")
        print("\nNext steps:")
        print("   - Install dependencies (numpy, scipy, anndata) for full unit tests")
        print("   - Run: python omicverse/utils/inspector/tests/test_validators.py")
        print("   - Create integration tests with actual AnnData objects")
        return 0
    else:
        print(f"\n❌ Layer 2 Phase 1 structure validation FAILED!")
        print(f"   {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
