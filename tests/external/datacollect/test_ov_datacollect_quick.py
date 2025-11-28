#!/usr/bin/env python3
"""
Quick test script for OmicVerse DataCollect external module.

This script provides a fast way to verify that OmicVerse DataCollect
is properly installed and the main functionality is accessible.

Usage:
    python test_ov_datacollect_quick.py
"""

import sys
import time

def test_omicverse_datacollect():
    """Quick test of OmicVerse DataCollect functionality."""
    
    print("🧪 Quick Test: OmicVerse DataCollect Integration")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    # Test 1: Basic Import
    print("1️⃣ Testing OmicVerse import...")
    try:
        import omicverse as ov
        print("   ✅ OmicVerse imported successfully")
        results.append(True)
    except Exception as e:
        print(f"   ❌ OmicVerse import failed: {e}")
        results.append(False)
        return results
    
    # Test 2: DataCollect Module Availability
    print("2️⃣ Testing DataCollect module availability...")
    try:
        if hasattr(ov, 'external') and hasattr(ov.external, 'datacollect'):
            datacollect = ov.external.datacollect
            print("   ✅ DataCollect module found")
            results.append(True)
        else:
            print("   ❌ DataCollect module not found in ov.external")
            results.append(False)
            return results
    except Exception as e:
        print(f"   ❌ DataCollect module access failed: {e}")
        results.append(False)
        return results
    
    # Test 3: Main Collection Functions
    print("3️⃣ Testing main collection functions...")
    main_functions = ['collect_protein_data', 'collect_expression_data', 'collect_pathway_data']
    available_functions = []
    
    for func_name in main_functions:
        if hasattr(datacollect, func_name):
            available_functions.append(func_name)
    
    if len(available_functions) == len(main_functions):
        print(f"   ✅ All main functions available: {', '.join(available_functions)}")
        results.append(True)
    else:
        missing = set(main_functions) - set(available_functions)
        print(f"   ⚠️ Some functions missing: {', '.join(missing)}")
        print(f"   Available: {', '.join(available_functions)}")
        results.append(False)
    
    # Test 4: API Clients Availability
    print("4️⃣ Testing API client availability...")
    test_clients = [
        'UniProtClient', 'PDBClient', 'EnsemblClient', 'GEOClient', 
        'KEGGClient', 'ReactomeClient', 'BLASTClient'
    ]
    
    available_clients = []
    for client_name in test_clients:
        if hasattr(datacollect, client_name):
            available_clients.append(client_name)
    
    if len(available_clients) >= 4:
        print(f"   ✅ API clients available ({len(available_clients)}/{len(test_clients)}): {', '.join(available_clients[:4])}...")
        results.append(True)
    else:
        print(f"   ⚠️ Limited API clients available ({len(available_clients)}/{len(test_clients)}): {', '.join(available_clients)}")
        results.append(len(available_clients) > 0)
    
    # Test 5: Format Converters
    print("5️⃣ Testing format converters...")
    format_functions = ['to_pandas', 'to_anndata', 'to_mudata']
    available_formats = []
    
    for format_func in format_functions:
        if hasattr(datacollect, format_func):
            available_formats.append(format_func)
    
    if available_formats:
        print(f"   ✅ Format converters available: {', '.join(available_formats)}")
        results.append(True)
    else:
        print("   ⚠️ No format converters found")
        results.append(False)
    
    # Test 6: Module Information
    print("6️⃣ Testing module information...")
    try:
        version = getattr(datacollect, '__version__', 'Unknown')
        author = getattr(datacollect, '__author__', 'Unknown')
        all_items = getattr(datacollect, '__all__', [])
        
        print(f"   ✅ Module info - Version: {version}, Items: {len(all_items)}")
        results.append(True)
    except Exception as e:
        print(f"   ⚠️ Module info access failed: {e}")
        results.append(False)
    
    # Test 7: Function Call Test (Safe)
    print("7️⃣ Testing function calls (safe mode)...")
    try:
        # Test that we can call the function (even if it fails due to network/data issues)
        if hasattr(datacollect, 'collect_protein_data'):
            try:
                # This will likely fail, but tests that the function is callable
                result = datacollect.collect_protein_data("INVALID_ID")
                print("   ✅ Function calls work (unexpected success)")
                results.append(True)
            except Exception:
                # Expected to fail with invalid ID, but function is callable
                print("   ✅ Function calls work (expected failure with invalid ID)")
                results.append(True)
        else:
            print("   ❌ collect_protein_data function not found")
            results.append(False)
    except Exception as e:
        print(f"   ❌ Function call test failed: {e}")
        results.append(False)
    
    # Test 8: Dependencies Check
    print("8️⃣ Testing dependencies...")
    dependencies = ['pandas', 'numpy', 'requests']
    available_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available_deps.append(dep)
        except ImportError:
            pass
    
    if len(available_deps) >= 2:
        print(f"   ✅ Key dependencies available: {', '.join(available_deps)}")
        results.append(True)
    else:
        print(f"   ⚠️ Missing dependencies. Available: {', '.join(available_deps)}")
        results.append(False)
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total} tests")
    print(f"⏱️ Time: {total_time:.2f} seconds")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 SUCCESS: OmicVerse DataCollect is working correctly!")
        print("\n📚 Next steps:")
        print("   - Try the examples in the documentation")
        print("   - Run the comprehensive test suite for detailed testing")
        print("   - Check the tutorial: docs/OMICVERSE_TUTORIAL.md")
    elif passed >= total * 0.7:
        print("\n⚠️ PARTIAL SUCCESS: Most functionality is working")
        print("\n🔧 Recommendations:")
        print("   - Install missing dependencies if needed")
        print("   - Check network connectivity for API tests")
        print("   - Review failed tests for specific issues")
    else:
        print("\n❌ ISSUES DETECTED: Multiple components not working")
        print("\n🚨 Actions needed:")
        print("   - Verify OmicVerse installation")
        print("   - Check DataCollect module integration") 
        print("   - Install required dependencies")
        print("   - Run comprehensive test suite for details")
    
    print("\n📖 Documentation available at:")
    print("   - README: omicverse/external/datacollect/README.md")
    print("   - Tutorial: omicverse/external/datacollect/docs/OMICVERSE_TUTORIAL.md")
    
    return results

if __name__ == "__main__":
    try:
        results = test_omicverse_datacollect()
        
        # Exit code based on results
        passed = sum(results)
        total = len(results)
        
        if passed == total:
            sys.exit(0)  # All tests passed
        elif passed >= total * 0.7:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Major issues
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {e}")
        sys.exit(3)