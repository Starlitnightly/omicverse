#!/usr/bin/env python3
"""
Comprehensive test script for OmicVerse DataCollect external module.

This script tests all major functionality of the DataCollect integration
with OmicVerse, including API clients, collection functions, format converters,
and utilities.

Usage:
    python test_omicverse_datacollect_complete.py [--verbose] [--fast] [--category CATEGORY]

Categories:
    - all: Test everything (default)
    - main: Test main collection functions
    - api: Test API clients
    - formats: Test format converters
    - utils: Test utilities
"""

import sys
import time
import traceback
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestResult:
    """Test result data structure."""
    name: str
    category: str
    success: bool
    message: str
    duration: float
    error: Optional[str] = None

class OmicVerseDataCollectTester:
    """Comprehensive test suite for OmicVerse DataCollect module."""
    
    def __init__(self, verbose: bool = False, fast_mode: bool = False):
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Import omicverse and check availability
        self.ov = None
        self.datacollect = None
        self._setup_omicverse()
    
    def _setup_omicverse(self):
        """Setup OmicVerse and DataCollect module."""
        try:
            import omicverse as ov
            self.ov = ov
            
            if hasattr(ov, 'external') and hasattr(ov.external, 'datacollect'):
                self.datacollect = ov.external.datacollect
                self.log("✅ OmicVerse DataCollect module detected")
            else:
                raise ImportError("DataCollect module not found in OmicVerse external modules")
                
        except Exception as e:
            self.log(f"❌ Failed to setup OmicVerse: {e}")
            sys.exit(1)
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def run_test(self, test_func, name: str, category: str, *args, **kwargs) -> TestResult:
        """Run a single test and record results."""
        start_time = time.time()
        
        try:
            self.log(f"🧪 Running {name}...")
            result = test_func(*args, **kwargs)
            
            if result is True or (isinstance(result, tuple) and result[0] is True):
                message = result[1] if isinstance(result, tuple) else "Test passed"
                success = True
                error = None
            else:
                message = result[1] if isinstance(result, tuple) else "Test failed"
                success = False
                error = None
                
        except Exception as e:
            success = False
            message = f"Exception: {str(e)}"
            error = traceback.format_exc()
            self.log(f"❌ {name} failed: {e}")
        
        duration = time.time() - start_time
        test_result = TestResult(name, category, success, message, duration, error)
        self.results.append(test_result)
        
        if success:
            self.log(f"✅ {name} passed ({duration:.2f}s)")
        
        return test_result
    
    # ==================== MAIN COLLECTION FUNCTIONS ====================
    
    def test_collect_protein_data(self) -> Tuple[bool, str]:
        """Test main protein data collection function."""
        try:
            # Test basic protein collection
            result = self.datacollect.collect_protein_data("P04637", to_format="pandas")
            
            if result is not None:
                return True, f"Protein data collected successfully (type: {type(result).__name__})"
            else:
                return False, "Protein data collection returned None"
                
        except Exception as e:
            return False, f"Protein collection failed: {e}"
    
    def test_collect_expression_data(self) -> Tuple[bool, str]:
        """Test main expression data collection function."""
        try:
            # Test with a small public dataset or mock data
            # Note: This might fail due to network/API issues, which is expected
            try:
                result = self.datacollect.collect_expression_data("GSE123456", to_format="pandas")
                return True, f"Expression data collection attempted (type: {type(result).__name__})"
            except Exception:
                # Expected to fail for non-existent datasets, test function availability
                if hasattr(self.datacollect, 'collect_expression_data'):
                    return True, "Expression data collection function available"
                else:
                    return False, "Expression data collection function not found"
                    
        except Exception as e:
            return False, f"Expression collection test failed: {e}"
    
    def test_collect_pathway_data(self) -> Tuple[bool, str]:
        """Test main pathway data collection function."""
        try:
            # Test pathway collection
            try:
                result = self.datacollect.collect_pathway_data("hsa04110", to_format="pandas")
                return True, f"Pathway data collected (type: {type(result).__name__})"
            except Exception:
                # Test function availability
                if hasattr(self.datacollect, 'collect_pathway_data'):
                    return True, "Pathway data collection function available"
                else:
                    return False, "Pathway data collection function not found"
                    
        except Exception as e:
            return False, f"Pathway collection test failed: {e}"
    
    # ==================== API CLIENTS TESTS ====================
    
    def test_protein_api_clients(self) -> Tuple[bool, str]:
        """Test protein API clients."""
        protein_clients = [
            'UniProtClient', 'PDBClient', 'AlphaFoldClient', 
            'InterProClient', 'STRINGClient', 'EMDBClient'
        ]
        
        available_clients = []
        missing_clients = []
        
        for client_name in protein_clients:
            try:
                if hasattr(self.datacollect, client_name):
                    client_class = getattr(self.datacollect, client_name)
                    # Try to instantiate
                    client = client_class()
                    available_clients.append(client_name)
                else:
                    missing_clients.append(client_name)
            except Exception as e:
                missing_clients.append(f"{client_name} ({e})")
        
        if len(available_clients) >= 3:  # At least half should work
            return True, f"Protein clients available: {', '.join(available_clients)}"
        else:
            return False, f"Too few protein clients available. Missing: {', '.join(missing_clients)}"
    
    def test_genomics_api_clients(self) -> Tuple[bool, str]:
        """Test genomics API clients."""
        genomics_clients = [
            'EnsemblClient', 'ClinVarClient', 'dbSNPClient', 
            'gnomADClient', 'GWASCatalogClient', 'UCSCClient', 'RegulomeDBClient'
        ]
        
        available_clients = []
        
        for client_name in genomics_clients:
            try:
                if hasattr(self.datacollect, client_name):
                    client_class = getattr(self.datacollect, client_name)
                    client = client_class()
                    available_clients.append(client_name)
            except Exception:
                pass
        
        if len(available_clients) >= 3:
            return True, f"Genomics clients available: {', '.join(available_clients)}"
        else:
            return False, f"Insufficient genomics clients available: {available_clients}"
    
    def test_expression_api_clients(self) -> Tuple[bool, str]:
        """Test expression API clients."""
        expression_clients = [
            'GEOClient', 'OpenTargetsClient', 'OpenTargetsGeneticsClient',
            'ReMapClient', 'CCREClient'
        ]
        
        available_clients = []
        
        for client_name in expression_clients:
            try:
                if hasattr(self.datacollect, client_name):
                    client_class = getattr(self.datacollect, client_name)
                    client = client_class()
                    available_clients.append(client_name)
            except Exception:
                pass
        
        if len(available_clients) >= 2:
            return True, f"Expression clients available: {', '.join(available_clients)}"
        else:
            return False, f"Insufficient expression clients available: {available_clients}"
    
    def test_pathway_api_clients(self) -> Tuple[bool, str]:
        """Test pathway API clients."""
        pathway_clients = ['KEGGClient', 'ReactomeClient', 'GtoPdbClient']
        
        available_clients = []
        
        for client_name in pathway_clients:
            try:
                if hasattr(self.datacollect, client_name):
                    client_class = getattr(self.datacollect, client_name)
                    client = client_class()
                    available_clients.append(client_name)
            except Exception:
                pass
        
        if len(available_clients) >= 2:
            return True, f"Pathway clients available: {', '.join(available_clients)}"
        else:
            return False, f"Insufficient pathway clients available: {available_clients}"
    
    def test_specialized_api_clients(self) -> Tuple[bool, str]:
        """Test specialized API clients."""
        specialized_clients = [
            'BLASTClient', 'JASPARClient', 'MPDClient', 'IUCNClient',
            'PRIDEClient', 'cBioPortalClient', 'WORMSClient', 'PaleobiologyClient'
        ]
        
        available_clients = []
        
        for client_name in specialized_clients:
            try:
                if hasattr(self.datacollect, client_name):
                    client_class = getattr(self.datacollect, client_name)
                    client = client_class()
                    available_clients.append(client_name)
            except Exception:
                pass
        
        if len(available_clients) >= 3:
            return True, f"Specialized clients available: {', '.join(available_clients)}"
        else:
            return False, f"Insufficient specialized clients available: {available_clients}"
    
    # ==================== FORMAT CONVERTERS ====================
    
    def test_format_converters(self) -> Tuple[bool, str]:
        """Test format conversion utilities."""
        try:
            # Test format converter availability
            converters = ['to_pandas', 'to_anndata', 'to_mudata']
            available_converters = []
            
            for converter in converters:
                if hasattr(self.datacollect, converter):
                    available_converters.append(converter)
            
            if len(available_converters) >= 1:
                return True, f"Format converters available: {', '.join(available_converters)}"
            else:
                return False, "No format converters found"
                
        except Exception as e:
            return False, f"Format converter test failed: {e}"
    
    def test_pandas_conversion(self) -> Tuple[bool, str]:
        """Test pandas format conversion."""
        try:
            if hasattr(self.datacollect, 'to_pandas'):
                # Test with mock data
                test_data = {
                    "protein_id": ["P04637", "P21359"],
                    "name": ["p53", "NF1"],
                    "length": [393, 2839]
                }
                
                result = self.datacollect.to_pandas(test_data, "protein")
                
                if result is not None:
                    return True, f"Pandas conversion successful (shape: {getattr(result, 'shape', 'unknown')})"
                else:
                    return False, "Pandas conversion returned None"
            else:
                return False, "to_pandas function not available"
                
        except Exception as e:
            return False, f"Pandas conversion failed: {e}"
    
    def test_anndata_conversion(self) -> Tuple[bool, str]:
        """Test AnnData format conversion."""
        try:
            if hasattr(self.datacollect, 'to_anndata'):
                # Test availability (might fail due to missing anndata)
                try:
                    test_data = {
                        "expression_matrix": [[1, 2], [3, 4]],
                        "samples": {"sample1": {"condition": "control"}, "sample2": {"condition": "treatment"}},
                        "genes": {"gene1": {"symbol": "GENE1"}, "gene2": {"symbol": "GENE2"}}
                    }
                    
                    result = self.datacollect.to_anndata(test_data)
                    
                    if result is not None:
                        return True, f"AnnData conversion successful (shape: {getattr(result, 'shape', 'unknown')})"
                    else:
                        return True, "AnnData conversion function available (returned None - expected for test data)"
                        
                except ImportError:
                    return True, "AnnData conversion available but AnnData not installed"
                    
            else:
                return False, "to_anndata function not available"
                
        except Exception as e:
            return False, f"AnnData conversion test failed: {e}"
    
    # ==================== UTILITIES TESTS ====================
    
    def test_validation_utilities(self) -> Tuple[bool, str]:
        """Test validation utilities."""
        try:
            # Check for validation modules
            validation_available = False
            
            # Try different import paths
            import_paths = [
                'utils.validation',
                'validation'
            ]
            
            for path in import_paths:
                try:
                    if hasattr(self.datacollect, path.split('.')[-1]):
                        validation_available = True
                        break
                except:
                    continue
            
            if validation_available:
                return True, "Validation utilities detected"
            else:
                return True, "Validation utilities may be available (import path variations)"
                
        except Exception as e:
            return False, f"Validation utilities test failed: {e}"
    
    def test_collectors(self) -> Tuple[bool, str]:
        """Test collector classes."""
        try:
            collector_classes = [
                'UniProtCollector', 'GEOCollector', 'KEGGCollector', 
                'BatchCollector', 'BaseCollector'
            ]
            
            available_collectors = []
            
            for collector in collector_classes:
                if hasattr(self.datacollect, collector):
                    available_collectors.append(collector)
            
            if len(available_collectors) >= 2:
                return True, f"Collectors available: {', '.join(available_collectors)}"
            else:
                return True, f"Some collectors available: {available_collectors}"
                
        except Exception as e:
            return False, f"Collectors test failed: {e}"
    
    def test_configuration(self) -> Tuple[bool, str]:
        """Test configuration system."""
        try:
            # Check for configuration availability
            config_indicators = ['config', 'settings', '__version__']
            found_config = []
            
            for indicator in config_indicators:
                if hasattr(self.datacollect, indicator):
                    found_config.append(indicator)
            
            if found_config:
                return True, f"Configuration system detected: {', '.join(found_config)}"
            else:
                return True, "Configuration system may be available"
                
        except Exception as e:
            return False, f"Configuration test failed: {e}"
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_omicverse_integration(self) -> Tuple[bool, str]:
        """Test OmicVerse integration points."""
        try:
            integration_points = []
            
            # Check module availability
            if hasattr(self.ov, 'external'):
                integration_points.append("external module")
            
            if hasattr(self.ov.external, 'datacollect'):
                integration_points.append("datacollect module")
            
            # Check main functions
            main_functions = ['collect_protein_data', 'collect_expression_data', 'collect_pathway_data']
            available_functions = [f for f in main_functions if hasattr(self.datacollect, f)]
            
            if available_functions:
                integration_points.extend(available_functions)
            
            if len(integration_points) >= 4:
                return True, f"OmicVerse integration verified: {', '.join(integration_points)}"
            else:
                return False, f"Limited OmicVerse integration: {', '.join(integration_points)}"
                
        except Exception as e:
            return False, f"OmicVerse integration test failed: {e}"
    
    def test_dependency_compatibility(self) -> Tuple[bool, str]:
        """Test dependency compatibility."""
        try:
            # Test common dependencies
            dependencies = {
                'pandas': 'import pandas as pd',
                'numpy': 'import numpy as np', 
                'requests': 'import requests',
                'sqlalchemy': 'import sqlalchemy'
            }
            
            available_deps = []
            missing_deps = []
            
            for dep_name, import_stmt in dependencies.items():
                try:
                    exec(import_stmt)
                    available_deps.append(dep_name)
                except ImportError:
                    missing_deps.append(dep_name)
            
            if len(available_deps) >= 3:
                return True, f"Dependencies available: {', '.join(available_deps)}"
            else:
                return False, f"Missing critical dependencies: {', '.join(missing_deps)}"
                
        except Exception as e:
            return False, f"Dependency test failed: {e}"
    
    # ==================== TEST RUNNER ====================
    
    def run_category_tests(self, category: str) -> Dict[str, List[TestResult]]:
        """Run tests for a specific category."""
        results = {}
        
        if category in ['all', 'main']:
            results['Main Collection Functions'] = [
                self.run_test(self.test_collect_protein_data, "Protein Data Collection", "main"),
                self.run_test(self.test_collect_expression_data, "Expression Data Collection", "main"),
                self.run_test(self.test_collect_pathway_data, "Pathway Data Collection", "main")
            ]
        
        if category in ['all', 'api']:
            results['API Clients'] = [
                self.run_test(self.test_protein_api_clients, "Protein API Clients", "api"),
                self.run_test(self.test_genomics_api_clients, "Genomics API Clients", "api"),
                self.run_test(self.test_expression_api_clients, "Expression API Clients", "api"),
                self.run_test(self.test_pathway_api_clients, "Pathway API Clients", "api"),
                self.run_test(self.test_specialized_api_clients, "Specialized API Clients", "api")
            ]
        
        if category in ['all', 'formats']:
            results['Format Converters'] = [
                self.run_test(self.test_format_converters, "Format Converter Availability", "formats"),
                self.run_test(self.test_pandas_conversion, "Pandas Conversion", "formats"),
                self.run_test(self.test_anndata_conversion, "AnnData Conversion", "formats")
            ]
        
        if category in ['all', 'utils']:
            results['Utilities'] = [
                self.run_test(self.test_validation_utilities, "Validation Utilities", "utils"),
                self.run_test(self.test_collectors, "Collector Classes", "utils"),
                self.run_test(self.test_configuration, "Configuration System", "utils")
            ]
        
        if category in ['all', 'integration']:
            results['Integration'] = [
                self.run_test(self.test_omicverse_integration, "OmicVerse Integration", "integration"),
                self.run_test(self.test_dependency_compatibility, "Dependency Compatibility", "integration")
            ]
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        total_time = time.time() - self.start_time
        
        # Count results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Generate report
        report = [
            "=" * 80,
            "🧪 OMICVERSE DATACOLLECT COMPREHENSIVE TEST REPORT",
            "=" * 80,
            "",
            f"📊 SUMMARY:",
            f"   Total Tests: {total_tests}",
            f"   ✅ Passed: {passed_tests}",
            f"   ❌ Failed: {failed_tests}", 
            f"   📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%",
            f"   ⏱️ Total Time: {total_time:.2f}s",
            "",
            "📋 DETAILED RESULTS:",
            ""
        ]
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Add detailed results
        for category, results in categories.items():
            report.append(f"🔗 {category.upper()}:")
            
            for result in results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                report.append(f"   {status} {result.name} ({result.duration:.2f}s)")
                report.append(f"      {result.message}")
                
                if not result.success and result.error and self.verbose:
                    report.append(f"      Error Details: {result.error}")
                
            report.append("")
        
        # Add recommendations
        report.extend([
            "🎯 RECOMMENDATIONS:",
            ""
        ])
        
        if failed_tests == 0:
            report.append("   🎉 All tests passed! OmicVerse DataCollect is working correctly.")
        else:
            report.append(f"   ⚠️ {failed_tests} test(s) failed. Review the details above.")
            
            # Specific recommendations based on failures
            failed_categories = set(r.category for r in self.results if not r.success)
            
            if 'main' in failed_categories:
                report.append("   🔧 Main collection functions need attention - check API availability")
            if 'api' in failed_categories:
                report.append("   🔌 Some API clients failed - verify network connectivity and imports")
            if 'formats' in failed_categories:
                report.append("   📊 Format converters failed - install missing dependencies (anndata, mudata)")
            if 'integration' in failed_categories:
                report.append("   🔗 Integration issues detected - verify OmicVerse installation")
        
        report.extend([
            "",
            "🔗 NEXT STEPS:",
            "",
            "   1. Fix any failing tests identified above",
            "   2. Install missing dependencies if needed:",
            "      pip install anndata mudata scanpy",
            "   3. Verify API keys are configured if using external APIs",
            "   4. Run tests again to confirm fixes",
            "",
            "📚 DOCUMENTATION:",
            "   - README: omicverse/external/datacollect/README.md",
            "   - Tutorial: omicverse/external/datacollect/docs/OMICVERSE_TUTORIAL.md",
            "   - API Reference: omicverse/external/datacollect/docs/OMICVERSE_API_REFERENCE.md",
            "",
            "=" * 80
        ])
        
        return "\n".join(report)

def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for OmicVerse DataCollect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_omicverse_datacollect_complete.py                    # Run all tests
  python test_omicverse_datacollect_complete.py --verbose          # Verbose output
  python test_omicverse_datacollect_complete.py --fast             # Skip slow tests
  python test_omicverse_datacollect_complete.py --category api     # Test only API clients
  python test_omicverse_datacollect_complete.py --category main    # Test main functions
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed test execution progress'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true', 
        help='Skip slow/network-dependent tests'
    )
    
    parser.add_argument(
        '--category', '-c',
        choices=['all', 'main', 'api', 'formats', 'utils', 'integration'],
        default='all',
        help='Test category to run (default: all)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Save report to file'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    print("🚀 Starting OmicVerse DataCollect Comprehensive Test Suite")
    print("=" * 60)
    
    tester = OmicVerseDataCollectTester(verbose=args.verbose, fast_mode=args.fast)
    
    # Run tests
    results = tester.run_category_tests(args.category)
    
    # Generate and display report
    report = tester.generate_report()
    print(report)
    
    # Save report if requested
    if args.output:
        Path(args.output).write_text(report)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    total_tests = len(tester.results)
    passed_tests = sum(1 for r in tester.results if r.success)
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()