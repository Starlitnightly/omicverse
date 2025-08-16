#!/usr/bin/env python3
"""
Network connectivity test script for OmicVerse DataCollect APIs.

This script tests actual network connectivity to the biological databases
to verify that API endpoints are accessible and responding correctly.

Usage:
    python test_ov_datacollect_network.py [--timeout SECONDS] [--apis API_LIST]
"""

import sys
import time
import requests
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class APIEndpoint:
    """API endpoint information."""
    name: str
    base_url: str
    test_endpoint: str
    expected_status: int = 200
    timeout: int = 10

class NetworkConnectivityTester:
    """Test network connectivity to biological database APIs."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.results = []
        
        # Define API endpoints to test
        self.api_endpoints = [
            # Protein & Structure APIs
            APIEndpoint("UniProt", "https://rest.uniprot.org", "/uniprotkb/P04637.json", 200),
            APIEndpoint("PDB", "https://data.rcsb.org", "/rest/v1/core/entry/1TUP", 200),
            APIEndpoint("AlphaFold", "https://alphafold.ebi.ac.uk", "/api/prediction/P04637", 200),
            APIEndpoint("InterPro", "https://www.ebi.ac.uk", "/interpro/api/protein/uniprot/P04637/", 200),
            APIEndpoint("STRING", "https://string-db.org", "/api/json/get_string_ids?identifiers=TP53&species=9606", 200),
            APIEndpoint("EMDB", "https://www.ebi.ac.uk", "/emdb/api/entry/EMD-1234", 404),  # May not exist
            
            # Genomics APIs
            APIEndpoint("Ensembl", "https://rest.ensembl.org", "/info/ping", 200),
            APIEndpoint("ClinVar", "https://eutils.ncbi.nlm.nih.gov", "/entrez/eutils/esearch.fcgi?db=clinvar&term=BRCA1", 200),
            APIEndpoint("dbSNP", "https://eutils.ncbi.nlm.nih.gov", "/entrez/eutils/esearch.fcgi?db=snp&term=rs7412", 200),
            APIEndpoint("gnomAD", "https://gnomad.broadinstitute.org", "/api/v4/graphql", 400),  # Needs POST with GraphQL
            APIEndpoint("GWAS Catalog", "https://www.ebi.ac.uk", "/gwas/rest/api", 200),
            APIEndpoint("UCSC", "https://api.genome.ucsc.edu", "/list/publicHubs", 200),
            
            # Expression & Regulation APIs
            APIEndpoint("GEO", "https://www.ncbi.nlm.nih.gov", "/geo/query/acc.cgi?acc=GSE1&targ=self&form=text&view=quick", 200),
            APIEndpoint("OpenTargets", "https://api.platform.opentargets.org", "/api/v4/graphql", 400),  # Needs POST
            APIEndpoint("ReMap", "http://remap.univ-amu.fr", "/", 200),
            
            # Pathway APIs
            APIEndpoint("KEGG", "https://rest.kegg.jp", "/info/kegg", 200),
            APIEndpoint("Reactome", "https://reactome.org", "/ContentService/data/database/version", 200),
            APIEndpoint("GtoPdb", "https://www.guidetopharmacology.org", "/services/targets", 200),
            
            # Specialized APIs
            APIEndpoint("NCBI BLAST", "https://www.ncbi.nlm.nih.gov", "/blast/", 200),
            APIEndpoint("JASPAR", "http://jaspar.genereg.net", "/api/v1/docs/", 200),
            APIEndpoint("IUCN", "https://www.iucnredlist.org", "/", 200),  # Main website (API needs key)
            APIEndpoint("PRIDE", "https://www.ebi.ac.uk", "/pride/ws/archive/v2/projects", 200),
        ]
    
    def test_single_endpoint(self, endpoint: APIEndpoint) -> Tuple[bool, str, float]:
        """Test a single API endpoint."""
        start_time = time.time()
        
        try:
            url = endpoint.base_url + endpoint.test_endpoint
            response = requests.get(url, timeout=self.timeout)
            duration = time.time() - start_time
            
            if response.status_code == endpoint.expected_status:
                return True, f"✅ {response.status_code} - {duration:.2f}s", duration
            elif response.status_code in [200, 201, 202]:
                return True, f"✅ {response.status_code} (unexpected but OK) - {duration:.2f}s", duration
            else:
                return False, f"❌ {response.status_code} - {duration:.2f}s", duration
                
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            return False, f"⏰ Timeout ({self.timeout}s)", duration
        except requests.exceptions.ConnectionError:
            duration = time.time() - start_time
            return False, f"🔌 Connection Error - {duration:.2f}s", duration
        except Exception as e:
            duration = time.time() - start_time
            return False, f"❌ Error: {str(e)[:50]} - {duration:.2f}s", duration
    
    def test_all_endpoints(self, selected_apis: Optional[List[str]] = None) -> Dict[str, Tuple[bool, str, float]]:
        """Test all API endpoints with threading."""
        print("🌐 Testing Network Connectivity to Biological Database APIs")
        print("=" * 60)
        
        # Filter APIs if specified
        endpoints_to_test = self.api_endpoints
        if selected_apis:
            endpoints_to_test = [ep for ep in self.api_endpoints if ep.name.lower() in [api.lower() for api in selected_apis]]
        
        results = {}
        
        print(f"Testing {len(endpoints_to_test)} API endpoints (timeout: {self.timeout}s)...\n")
        
        # Use ThreadPoolExecutor for concurrent testing
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tests
            future_to_endpoint = {
                executor.submit(self.test_single_endpoint, endpoint): endpoint 
                for endpoint in endpoints_to_test
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_endpoint):
                endpoint = future_to_endpoint[future]
                try:
                    success, message, duration = future.result()
                    results[endpoint.name] = (success, message, duration)
                    print(f"{endpoint.name:15} | {message}")
                except Exception as e:
                    results[endpoint.name] = (False, f"❌ Exception: {e}", 0.0)
                    print(f"{endpoint.name:15} | ❌ Exception: {e}")
        
        return results
    
    def generate_network_report(self, results: Dict[str, Tuple[bool, str, float]]) -> str:
        """Generate a network connectivity report."""
        total_apis = len(results)
        successful_apis = sum(1 for success, _, _ in results.values() if success)
        
        # Calculate average response time for successful requests
        successful_times = [duration for success, _, duration in results.values() if success]
        avg_response_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        report = [
            "\n" + "=" * 60,
            "📊 NETWORK CONNECTIVITY REPORT",
            "=" * 60,
            "",
            f"🌐 Total APIs Tested: {total_apis}",
            f"✅ Successful Connections: {successful_apis}",
            f"❌ Failed Connections: {total_apis - successful_apis}",
            f"📈 Success Rate: {(successful_apis/total_apis)*100:.1f}%",
            f"⚡ Average Response Time: {avg_response_time:.2f}s",
            "",
            "📋 DETAILED RESULTS BY CATEGORY:",
            ""
        ]
        
        # Group by category
        categories = {
            "Protein & Structure": ["UniProt", "PDB", "AlphaFold", "InterPro", "STRING", "EMDB"],
            "Genomics & Variants": ["Ensembl", "ClinVar", "dbSNP", "gnomAD", "GWAS Catalog", "UCSC"],
            "Expression & Regulation": ["GEO", "OpenTargets", "ReMap"],
            "Pathways": ["KEGG", "Reactome", "GtoPdb"],
            "Specialized": ["NCBI BLAST", "JASPAR", "IUCN", "PRIDE"]
        }
        
        for category, api_names in categories.items():
            report.append(f"🔗 {category}:")
            category_results = {name: results[name] for name in api_names if name in results}
            
            if category_results:
                category_success = sum(1 for success, _, _ in category_results.values() if success)
                report.append(f"   Success: {category_success}/{len(category_results)}")
                
                for api_name, (success, message, duration) in category_results.items():
                    status = "✅" if success else "❌"
                    report.append(f"   {status} {api_name}: {message}")
            else:
                report.append("   No APIs tested in this category")
            
            report.append("")
        
        # Add recommendations
        report.extend([
            "🎯 RECOMMENDATIONS:",
            ""
        ])
        
        if successful_apis == total_apis:
            report.append("🎉 All APIs are accessible! Network connectivity is excellent.")
        elif successful_apis >= total_apis * 0.8:
            report.append("✅ Most APIs are accessible. Some may be temporarily unavailable.")
        elif successful_apis >= total_apis * 0.5:
            report.append("⚠️ Mixed connectivity. Check your network connection and firewall settings.")
        else:
            report.append("🚨 Poor connectivity. Check network connection, firewall, and proxy settings.")
        
        # Specific recommendations based on failed APIs
        failed_apis = [name for name, (success, _, _) in results.items() if not success]
        if failed_apis:
            report.append(f"\n❌ Failed APIs ({len(failed_apis)}): {', '.join(failed_apis[:5])}")
            if len(failed_apis) > 5:
                report.append(f"   ... and {len(failed_apis) - 5} more")
            
            report.extend([
                "",
                "🔧 Troubleshooting steps:",
                "   1. Check your internet connection",
                "   2. Verify firewall/proxy settings allow HTTPS requests",
                "   3. Some APIs may require authentication (see documentation)",
                "   4. APIs may be temporarily down for maintenance",
                "   5. Consider using VPN if geographic restrictions apply"
            ])
        
        report.extend([
            "",
            "📚 NEXT STEPS:",
            "",
            "   • If most APIs work: Proceed with DataCollect usage",
            "   • If many APIs fail: Check network configuration",
            "   • For API-specific issues: Consult API documentation",
            "   • For persistent issues: Contact system administrator",
            "",
            "🔗 Related Documentation:",
            "   • Configuration: docs/CONFIGURATION.md",
            "   • Troubleshooting: docs/TROUBLESHOOTING.md",
            "   • API Reference: docs/OMICVERSE_API_REFERENCE.md",
            "",
            "=" * 60
        ])
        
        return "\n".join(report)

def main():
    """Main network test execution."""
    parser = argparse.ArgumentParser(
        description="Test network connectivity to biological database APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ov_datacollect_network.py                          # Test all APIs
  python test_ov_datacollect_network.py --timeout 30             # Longer timeout
  python test_ov_datacollect_network.py --apis UniProt PDB       # Test specific APIs
  python test_ov_datacollect_network.py --output network_report.txt
        """
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=10,
        help='Request timeout in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--apis',
        nargs='+',
        help='Specific APIs to test (space-separated list)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Save report to file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = NetworkConnectivityTester(timeout=args.timeout)
    
    # Run tests
    start_time = time.time()
    results = tester.test_all_endpoints(selected_apis=args.apis)
    total_time = time.time() - start_time
    
    # Generate and display report
    report = tester.generate_network_report(results)
    
    # Add timing info
    timing_info = f"\n⏱️ Total Test Time: {total_time:.2f} seconds"
    report += timing_info
    
    print(report)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    total_apis = len(results)
    successful_apis = sum(1 for success, _, _ in results.values() if success)
    
    if successful_apis == total_apis:
        print("\n🎉 All network tests passed!")
        sys.exit(0)
    elif successful_apis >= total_apis * 0.8:
        print(f"\n✅ Most network tests passed ({successful_apis}/{total_apis})")
        sys.exit(0)
    else:
        print(f"\n⚠️ Network connectivity issues detected ({successful_apis}/{total_apis})")
        sys.exit(1)

if __name__ == "__main__":
    main()