#!/usr/bin/env python3
"""
Test script to verify that the OmicVerse Function Registry works with the agent.
"""

import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_registry_basic():
    """Test 1: Check if registry is populated"""
    print("=" * 70)
    print("TEST 1: Checking if Function Registry is populated")
    print("=" * 70)

    from omicverse._registry import _global_registry

    # Count unique functions
    unique_functions = set()
    categories = set()

    for entry in _global_registry._registry.values():
        unique_functions.add(entry['full_name'])
        categories.add(entry['category'])

    print(f"✅ Registry contains {len(unique_functions)} unique functions")
    print(f"✅ Registry has {len(categories)} categories: {sorted(categories)}")

    return len(unique_functions) > 0


def test_registry_search():
    """Test 2: Check if registry search works"""
    print("\n" + "=" * 70)
    print("TEST 2: Testing Registry Search Functionality")
    print("=" * 70)

    from omicverse._registry import _global_registry

    # Test Chinese search
    results = _global_registry.find("质控")
    print(f"\n🔍 Search for '质控' (quality control in Chinese):")
    if results:
        print(f"   ✅ Found {len(results)} match(es)")
        print(f"   📦 {results[0]['full_name']}")
        print(f"   🏷️  Aliases: {results[0]['aliases']}")
    else:
        print("   ❌ No results")
        return False

    # Test English search
    results = _global_registry.find("pca")
    print(f"\n🔍 Search for 'pca':")
    if results:
        print(f"   ✅ Found {len(results)} match(es)")
        print(f"   📦 {results[0]['full_name']}")
    else:
        print("   ❌ No results")
        return False

    return True


def test_agent_integration():
    """Test 3: Check if agent uses the registry"""
    print("\n" + "=" * 70)
    print("TEST 3: Checking Agent Integration with Registry")
    print("=" * 70)

    try:
        from omicverse.utils.smart_agent import OmicVerseAgent
        from omicverse._registry import _global_registry

        # Check if agent imports registry
        print("\n✅ Agent successfully imports registry module")

        # Check if agent has registry methods
        agent_class_methods = dir(OmicVerseAgent)
        registry_methods = [
            '_get_registry_stats',
            '_get_available_functions_info',
            '_search_functions',
            '_get_function_details'
        ]

        found_methods = [m for m in registry_methods if m in agent_class_methods]
        print(f"✅ Agent has {len(found_methods)}/{len(registry_methods)} registry integration methods:")
        for method in found_methods:
            print(f"   - {method}")

        return len(found_methods) == len(registry_methods)

    except Exception as e:
        print(f"❌ Error checking agent integration: {e}")
        return False


def test_agent_system_prompt():
    """Test 4: Check if registry info is injected into system prompt"""
    print("\n" + "=" * 70)
    print("TEST 4: Checking System Prompt Injection")
    print("=" * 70)

    try:
        # Read the smart_agent.py file to verify prompt injection
        agent_file = PROJECT_ROOT / "omicverse" / "utils" / "smart_agent.py"
        with open(agent_file, 'r') as f:
            content = f.read()

        # Check for key integration points
        checks = [
            ("from .registry import _global_registry", "Registry import"),
            ("functions_info = self._get_available_functions_info()", "Get functions info"),
            ("instructions = \"\"\"", "System prompt definition"),
            ("Here are all the currently registered functions", "Registry injection"),
        ]

        print()
        for pattern, description in checks:
            if pattern in content:
                print(f"   ✅ {description}: Found")
            else:
                print(f"   ❌ {description}: NOT FOUND")
                return False

        return True

    except Exception as e:
        print(f"❌ Error checking system prompt: {e}")
        return False


def test_user_api():
    """Test 5: Check if user-facing API works"""
    print("\n" + "=" * 70)
    print("TEST 5: Testing User-Facing API")
    print("=" * 70)

    try:
        import omicverse as ov

        # Check if API functions are available
        api_functions = ['find_function', 'list_functions', 'help']

        print()
        for func_name in api_functions:
            if hasattr(ov, func_name):
                print(f"   ✅ ov.{func_name}() is available")
            else:
                print(f"   ❌ ov.{func_name}() NOT available")
                return False

        return True

    except Exception as e:
        print(f"❌ Error checking user API: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🔬" * 35)
    print("OmicVerse Function Registry Integration Test")
    print("🔬" * 35 + "\n")

    tests = [
        ("Registry Population", test_registry_basic),
        ("Registry Search", test_registry_search),
        ("Agent Integration", test_agent_integration),
        ("System Prompt Injection", test_agent_system_prompt),
        ("User-Facing API", test_user_api),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The Function Registry is working correctly with ov.agent!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
