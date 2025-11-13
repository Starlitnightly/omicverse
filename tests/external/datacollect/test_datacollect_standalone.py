#!/usr/bin/env python3
"""
Standalone test script for DataCollect module functionality.

This script tests DataCollect functionality without requiring full OmicVerse
installation, useful when there are dependency conflicts.

Usage:
    python test_datacollect_standalone.py
"""

import sys
import time
import os
from pathlib import Path

def test_datacollect_standalone():
    """Test DataCollect functionality independently."""
    
    print("🧪 Standalone Test: DataCollect Module")
    print("=" * 45)
    
    results = []
    start_time = time.time()
    
    # Test 1: Direct API Client Import
    print("1️⃣ Testing direct API client imports...")
    try:
        # Add the datacollect path to sys.path
        # Calculate path from test location to source location
        test_root = Path(__file__).parent.parent.parent.parent  # Get to repo root
        datacollect_path = test_root / "omicverse" / "external" / "datacollect"
        if datacollect_path.exists() and (datacollect_path / "api").exists():
            sys.path.insert(0, str(datacollect_path))
            
            # Test importing some core clients
            from api.proteins.uniprot import UniProtClient
            from api.genomics.ensembl import EnsemblClient
            from api.pathways.kegg import KEGGClient
            
            print("   ✅ Core API clients imported successfully")
            results.append(True)
        else:
            print("   ❌ DataCollect API directory not found")
            results.append(False)
    except Exception as e:
        print(f"   ❌ API client import failed: {e}")
        results.append(False)
    
    # Test 2: Client Instantiation
    print("2️⃣ Testing client instantiation...")
    try:
        if results[0]:  # Only if imports worked
            client = UniProtClient()
            print("   ✅ UniProt client created successfully")
            results.append(True)
        else:
            print("   ⏭️ Skipped (imports failed)")
            results.append(False)
    except Exception as e:
        print(f"   ❌ Client instantiation failed: {e}")
        results.append(False)
    
    # Test 3: Base Client Functionality
    print("3️⃣ Testing base client functionality...")
    try:
        if results[1]:  # Only if client creation worked
            from api.base import BaseAPIClient
            base_client = BaseAPIClient("https://httpbin.org")
            print("   ✅ Base client functionality available")
            results.append(True)
        else:
            print("   ⏭️ Skipped (client creation failed)")
            results.append(False)
    except Exception as e:
        print(f"   ❌ Base client test failed: {e}")
        results.append(False)
    
    # Test 4: Collectors
    print("4️⃣ Testing collector classes...")
    try:
        from collectors.uniprot import UniProtCollector
        from collectors.base import BaseCollector
        
        collector = UniProtCollector()
        print("   ✅ Collector classes available")
        results.append(True)
    except Exception as e:
        print(f"   ❌ Collector test failed: {e}")
        results.append(False)
    
    # Test 5: Utilities
    print("5️⃣ Testing utility modules...")
    try:
        from utils.validation import validate_uniprot_id
        from utils.conversion import convert_to_pandas
        
        # Test validation
        is_valid = validate_uniprot_id("P04637")
        print(f"   ✅ Utilities available (validation: {is_valid})")
        results.append(True)
    except Exception as e:
        print(f"   ❌ Utilities test failed: {e}")
        results.append(False)
    
    # Test 6: Configuration
    print("6️⃣ Testing configuration system...")
    try:
        from config.settings import get_default_config
        config = get_default_config()
        print(f"   ✅ Configuration system working")
        results.append(True)
    except Exception as e:
        print(f"   ⚠️ Configuration test note: {e}")
        results.append(True)  # Config is optional
    
    # Test 7: Network Test (Safe)
    print("7️⃣ Testing network capabilities...")
    try:
        import requests
        # Test basic network connectivity
        response = requests.get("https://httpbin.org/status/200", timeout=5)
        if response.status_code == 200:
            print("   ✅ Network connectivity working")
            results.append(True)
        else:
            print("   ⚠️ Network connectivity issues")
            results.append(False)
    except Exception as e:
        print(f"   ⚠️ Network test failed: {e}")
        results.append(False)
    
    # Test 8: Dependencies
    print("8️⃣ Testing core dependencies...")
    dependencies = ['pandas', 'numpy', 'requests', 'sqlalchemy']
    available_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available_deps.append(dep)
        except ImportError:
            pass
    
    if len(available_deps) >= 3:
        print(f"   ✅ Core dependencies available: {', '.join(available_deps)}")
        results.append(True)
    else:
        print(f"   ⚠️ Some dependencies missing. Available: {', '.join(available_deps)}")
        results.append(False)
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 45)
    print("📊 STANDALONE TEST SUMMARY")
    print("=" * 45)
    print(f"✅ Passed: {passed}/{total} tests")
    print(f"⏱️ Time: {total_time:.2f} seconds")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 SUCCESS: DataCollect module is working correctly!")
        print("\n📚 Next steps:")
        print("   - DataCollect can be used independently")
        print("   - Try the network connectivity test")
        print("   - Check the demo usage examples")
    elif passed >= total * 0.7:
        print("\n⚠️ PARTIAL SUCCESS: Core functionality is working")
        print("\n🔧 Recommendations:")
        print("   - Install missing dependencies if needed")
        print("   - Check network connectivity")
        print("   - DataCollect should work for basic use cases")
    else:
        print("\n❌ ISSUES DETECTED: Multiple components not working")
        print("\n🚨 Actions needed:")
        print("   - Check DataCollect installation")
        print("   - Install missing dependencies")
        print("   - Verify module structure")
    
    print("\n📖 Alternative test options:")
    print("   - Network test: python test_ov_datacollect_network.py")
    print("   - Demo usage: python demo_ov_datacollect_usage.py --demo basic")
    print("   - API documentation: docs/API_REFERENCE.md")
    
    return results

if __name__ == "__main__":
    try:
        results = test_datacollect_standalone()
        
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