#!/usr/bin/env python3
"""
Test Suite Summary for OmicVerse DataCollect

This script provides an overview of all available test scripts and their
current status, helping users choose the right test for their needs.

Usage:
    python test_suite_summary.py [--run-available]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class TestSuiteSummary:
    """Summary and runner for DataCollect test suite."""
    
    def __init__(self):
        self.test_scripts = self._discover_test_scripts()
        self.status_results = {}
    
    def _discover_test_scripts(self) -> Dict[str, Dict]:
        """Discover available test scripts."""
        return {
            "test_ov_datacollect_quick.py": {
                "description": "Quick functionality validation (8 tests)",
                "purpose": "Fast verification of basic DataCollect functionality",
                "requirements": "OmicVerse with DataCollect module",
                "estimated_time": "~10 seconds",
                "tests": [
                    "OmicVerse import", "DataCollect module availability",
                    "Main collection functions", "API client availability", 
                    "Format converters", "Module information", "Function calls", "Dependencies"
                ]
            },
            
            "test_omicverse_datacollect_complete.py": {
                "description": "Comprehensive functionality testing",
                "purpose": "Complete validation of all DataCollect components",
                "requirements": "OmicVerse with DataCollect module",
                "estimated_time": "~60 seconds",
                "tests": [
                    "Main collection functions (3)", "API clients by category (29)", 
                    "Format converters (3)", "Utilities and collectors", "Integration tests"
                ]
            },
            
            "test_ov_datacollect_network.py": {
                "description": "Network connectivity testing (22+ APIs)",
                "purpose": "Test connectivity to biological database APIs",
                "requirements": "Internet connection",
                "estimated_time": "~30 seconds",
                "tests": [
                    "Protein APIs (UniProt, PDB, AlphaFold, etc.)",
                    "Genomics APIs (Ensembl, ClinVar, dbSNP, etc.)",
                    "Expression APIs (GEO, OpenTargets, etc.)",
                    "Pathway APIs (KEGG, Reactome, etc.)",
                    "Specialized APIs (BLAST, JASPAR, etc.)"
                ]
            },
            
            "demo_ov_datacollect_usage.py": {
                "description": "Interactive usage demonstrations",
                "purpose": "Learn DataCollect usage patterns and workflows",
                "requirements": "OmicVerse with DataCollect module",
                "estimated_time": "~30 seconds",
                "tests": [
                    "Basic data collection demos", "OmicVerse integration examples",
                    "Format conversion demos", "Complete workflow examples"
                ]
            },
            
            "test_datacollect_standalone.py": {
                "description": "Standalone module testing",
                "purpose": "Test DataCollect without full OmicVerse",
                "requirements": "DataCollect source code",
                "estimated_time": "~10 seconds",
                "tests": [
                    "Direct API imports", "Client instantiation", "Base functionality",
                    "Collectors", "Utilities", "Configuration", "Network", "Dependencies"
                ]
            }
        }
    
    def check_script_availability(self) -> Dict[str, bool]:
        """Check which test scripts are available."""
        availability = {}
        for script_name in self.test_scripts.keys():
            script_path = Path(script_name)
            availability[script_name] = script_path.exists()
        return availability
    
    def test_basic_requirements(self) -> Dict[str, bool]:
        """Test basic requirements for different test types."""
        requirements = {}
        
        # Test OmicVerse availability
        try:
            result = subprocess.run([sys.executable, "-c", "import omicverse"], 
                                  capture_output=True, text=True, timeout=10)
            requirements["omicverse_available"] = result.returncode == 0
            requirements["omicverse_error"] = result.stderr.strip() if result.returncode != 0 else None
        except Exception as e:
            requirements["omicverse_available"] = False
            requirements["omicverse_error"] = str(e)
        
        # Test network connectivity
        try:
            import requests
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            requirements["network_available"] = response.status_code == 200
        except Exception:
            requirements["network_available"] = False
        
        # Test core dependencies
        core_deps = ['pandas', 'numpy', 'requests']
        missing_deps = []
        for dep in core_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        requirements["core_dependencies"] = len(missing_deps) == 0
        requirements["missing_dependencies"] = missing_deps
        
        return requirements
    
    def get_recommendations(self, requirements: Dict[str, bool]) -> List[str]:
        """Get recommendations based on current environment."""
        recommendations = []
        
        if requirements["network_available"]:
            recommendations.append("✅ Network Test: test_ov_datacollect_network.py")
            recommendations.append("   → Tests API connectivity without OmicVerse")
        
        if requirements["omicverse_available"]:
            recommendations.append("✅ Quick Test: test_ov_datacollect_quick.py")
            recommendations.append("✅ Complete Test: test_omicverse_datacollect_complete.py")
            recommendations.append("✅ Demo: demo_ov_datacollect_usage.py")
        else:
            recommendations.append("❌ OmicVerse tests unavailable")
            if requirements["omicverse_error"]:
                recommendations.append(f"   Error: {requirements['omicverse_error']}")
            recommendations.append("   → Fix OmicVerse installation to run full tests")
        
        if not requirements["core_dependencies"]:
            recommendations.append("⚠️ Install missing dependencies:")
            recommendations.append(f"   pip install {' '.join(requirements['missing_dependencies'])}")
        
        return recommendations
    
    def run_available_tests(self):
        """Run tests that are likely to work in current environment."""
        requirements = self.test_basic_requirements()
        
        print("🏃 Running Available Tests...")
        print("=" * 50)
        
        # Always try network test first (most likely to work)
        if requirements["network_available"]:
            print("\n🌐 Running Network Connectivity Test...")
            try:
                result = subprocess.run([
                    sys.executable, "test_ov_datacollect_network.py", 
                    "--apis", "UniProt", "KEGG", "Ensembl", "--timeout", "10"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ Network test PASSED")
                    print("Summary: All tested APIs are accessible")
                else:
                    print("⚠️ Network test completed with issues")
                
            except Exception as e:
                print(f"❌ Network test failed: {e}")
        
        # Try OmicVerse tests if available
        if requirements["omicverse_available"]:
            print("\n🧪 Running Quick OmicVerse Test...")
            try:
                result = subprocess.run([
                    sys.executable, "test_ov_datacollect_quick.py"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("✅ OmicVerse quick test PASSED")
                else:
                    print("⚠️ OmicVerse quick test completed with issues")
                    
            except Exception as e:
                print(f"❌ OmicVerse test failed: {e}")
        
        print("\n📋 For detailed results, run individual test scripts")
    
    def show_summary(self):
        """Show comprehensive test suite summary."""
        print("🧪 OMICVERSE DATACOLLECT TEST SUITE SUMMARY")
        print("=" * 60)
        
        # Check availability
        availability = self.check_script_availability()
        requirements = self.test_basic_requirements()
        
        print("\n📋 AVAILABLE TEST SCRIPTS:")
        print("-" * 30)
        
        for script, details in self.test_scripts.items():
            status = "✅" if availability[script] else "❌"
            print(f"\n{status} {script}")
            print(f"   Purpose: {details['purpose']}")
            print(f"   Time: {details['estimated_time']}")
            print(f"   Tests: {len(details['tests'])} test categories")
            
            # Show first few test categories
            if len(details['tests']) <= 3:
                print(f"   Categories: {', '.join(details['tests'])}")
            else:
                print(f"   Categories: {', '.join(details['tests'][:2])}, +{len(details['tests'])-2} more")
        
        print(f"\n🔧 ENVIRONMENT STATUS:")
        print("-" * 25)
        print(f"✅ Network connectivity: {'Available' if requirements['network_available'] else 'Limited'}")
        print(f"{'✅' if requirements['omicverse_available'] else '❌'} OmicVerse: {'Available' if requirements['omicverse_available'] else 'Not available'}")
        if not requirements['omicverse_available'] and requirements['omicverse_error']:
            print(f"   Error: {requirements['omicverse_error']}")
        print(f"{'✅' if requirements['core_dependencies'] else '⚠️'} Core dependencies: {'Complete' if requirements['core_dependencies'] else 'Missing some'}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print("-" * 20)
        recommendations = self.get_recommendations(requirements)
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n🚀 QUICK START:")
        print("-" * 15)
        if requirements["network_available"]:
            print("   python test_ov_datacollect_network.py --apis UniProt KEGG")
        if requirements["omicverse_available"]:
            print("   python test_ov_datacollect_quick.py")
        else:
            print("   Fix OmicVerse installation first, then run tests")
        
        print(f"\n📚 DOCUMENTATION:")
        print("-" * 18)
        print("   - Test scripts include --help for detailed options")
        print("   - Network test: Verifies API connectivity")
        print("   - Quick test: Fast functionality check")
        print("   - Complete test: Comprehensive validation")
        print("   - Demo: Interactive learning examples")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="OmicVerse DataCollect Test Suite Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_suite_summary.py                    # Show test suite overview
  python test_suite_summary.py --run-available    # Run tests that should work
        """
    )
    
    parser.add_argument(
        '--run-available',
        action='store_true',
        help='Run tests that are likely to work in current environment'
    )
    
    args = parser.parse_args()
    
    summary = TestSuiteSummary()
    
    if args.run_available:
        summary.run_available_tests()
    else:
        summary.show_summary()

if __name__ == "__main__":
    main()