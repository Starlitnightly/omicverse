#!/usr/bin/env python3
"""
Demonstration script for OmicVerse DataCollect usage.

This script shows practical examples of how to use the DataCollect
external module within OmicVerse for real bioinformatics workflows.

Usage:
    python demo_ov_datacollect_usage.py [--demo DEMO_NAME]

Available demos:
    - basic: Basic data collection examples
    - integration: OmicVerse integration examples  
    - formats: Format conversion examples
    - workflow: Complete analysis workflow
    - all: Run all demos (default)
"""

import sys
import argparse
import time
from typing import Dict, Any, Optional

class OmicVerseDataCollectDemo:
    """Demonstration class for OmicVerse DataCollect usage."""
    
    def __init__(self):
        self.ov = None
        self.datacollect = None
        self._setup()
    
    def _setup(self):
        """Setup OmicVerse and DataCollect."""
        try:
            import omicverse as ov
            self.ov = ov
            
            if hasattr(ov, 'external') and hasattr(ov.external, 'datacollect'):
                self.datacollect = ov.external.datacollect
                print("✅ OmicVerse DataCollect module loaded successfully")
            else:
                raise ImportError("DataCollect module not found")
                
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print("\n💡 Make sure OmicVerse with DataCollect is properly installed")
            sys.exit(1)
    
    def demo_basic_collection(self):
        """Demonstrate basic data collection functionality."""
        print("\n" + "="*60)
        print("🧬 DEMO 1: Basic Data Collection")
        print("="*60)
        
        # Demo 1.1: Protein Data Collection
        print("\n1️⃣ Protein Data Collection")
        print("-" * 30)
        
        try:
            print("Collecting data for p53 protein (P04637)...")
            
            # Method 1: Using main collection function
            if hasattr(self.datacollect, 'collect_protein_data'):
                protein_data = self.datacollect.collect_protein_data("P04637", to_format="pandas")
                print(f"✅ Protein data collected: {type(protein_data).__name__}")
                if hasattr(protein_data, 'shape'):
                    print(f"   Data shape: {protein_data.shape}")
            else:
                print("⚠️ collect_protein_data function not available")
            
            # Method 2: Using direct API client
            if hasattr(self.datacollect, 'UniProtClient'):
                print("\nUsing UniProt API client directly...")
                client = self.datacollect.UniProtClient()
                # Note: This might fail due to network issues, which is expected in demo
                print("✅ UniProt client initialized")
            else:
                print("⚠️ UniProtClient not available")
                
        except Exception as e:
            print(f"ℹ️ Protein collection demo note: {e}")
            print("   (This is normal if network APIs are not accessible)")
        
        # Demo 1.2: Expression Data Collection
        print("\n2️⃣ Expression Data Collection")
        print("-" * 30)
        
        try:
            print("Testing expression data collection...")
            
            if hasattr(self.datacollect, 'collect_expression_data'):
                # Note: Using a non-existent dataset ID for demo purposes
                print("Expression data collection function available ✅")
                print("Example usage:")
                print("   adata = ov.external.datacollect.collect_expression_data('GSE123456', format='anndata')")
            else:
                print("⚠️ collect_expression_data function not available")
                
            if hasattr(self.datacollect, 'GEOClient'):
                print("GEO API client available ✅")
            else:
                print("⚠️ GEO client not available")
                
        except Exception as e:
            print(f"ℹ️ Expression collection demo note: {e}")
        
        # Demo 1.3: Pathway Data Collection  
        print("\n3️⃣ Pathway Data Collection")
        print("-" * 30)
        
        try:
            print("Testing pathway data collection...")
            
            if hasattr(self.datacollect, 'collect_pathway_data'):
                print("Pathway data collection function available ✅")
                print("Example usage:")
                print("   pathway = ov.external.datacollect.collect_pathway_data('hsa04110')")
            else:
                print("⚠️ collect_pathway_data function not available")
                
            pathway_clients = ['KEGGClient', 'ReactomeClient', 'GtoPdbClient']
            available_clients = [c for c in pathway_clients if hasattr(self.datacollect, c)]
            
            if available_clients:
                print(f"Pathway clients available: {', '.join(available_clients)} ✅")
            else:
                print("⚠️ No pathway clients available")
                
        except Exception as e:
            print(f"ℹ️ Pathway collection demo note: {e}")
    
    def demo_omicverse_integration(self):
        """Demonstrate OmicVerse integration features."""
        print("\n" + "="*60)
        print("🔗 DEMO 2: OmicVerse Integration")
        print("="*60)
        
        # Demo 2.1: Format Integration
        print("\n1️⃣ Format Integration with OmicVerse")
        print("-" * 40)
        
        try:
            print("Checking format conversion capabilities...")
            
            format_converters = ['to_pandas', 'to_anndata', 'to_mudata']
            available_formats = []
            
            for converter in format_converters:
                if hasattr(self.datacollect, converter):
                    available_formats.append(converter)
            
            if available_formats:
                print(f"✅ Format converters available: {', '.join(available_formats)}")
                
                # Demo pandas conversion
                if 'to_pandas' in available_formats:
                    print("\n📊 Pandas Conversion Demo:")
                    print("   Example: df = ov.external.datacollect.to_pandas(data, 'protein')")
                    
                    # Test with sample data
                    sample_data = {
                        "protein_id": ["P04637", "P21359"],
                        "name": ["p53", "NF1"],
                        "length": [393, 2839]
                    }
                    
                    try:
                        df = self.datacollect.to_pandas(sample_data, "protein")
                        print(f"   ✅ Sample conversion successful: {type(df).__name__}")
                        if hasattr(df, 'shape'):
                            print(f"      Shape: {df.shape}")
                    except Exception as e:
                        print(f"   ℹ️ Sample conversion note: {e}")
                
                # Demo AnnData conversion
                if 'to_anndata' in available_formats:
                    print("\n🧬 AnnData Conversion Demo:")
                    print("   Example: adata = ov.external.datacollect.to_anndata(expression_data)")
                    
                    try:
                        # Check if AnnData is available
                        import anndata
                        print("   ✅ AnnData library available")
                    except ImportError:
                        print("   ℹ️ AnnData library not installed (pip install anndata)")
            else:
                print("⚠️ No format converters found")
                
        except Exception as e:
            print(f"ℹ️ Format integration demo note: {e}")
        
        # Demo 2.2: Workflow Integration
        print("\n2️⃣ Workflow Integration")
        print("-" * 25)
        
        print("Example OmicVerse + DataCollect workflow:")
        print("""
        # Step 1: Collect expression data
        adata = ov.external.datacollect.collect_expression_data(
            "GSE123456", 
            format="anndata"
        )
        
        # Step 2: Use with OmicVerse analysis
        ov.pp.preprocess(adata)
        deg_results = ov.bulk.pyDEG(adata)
        
        # Step 3: Collect pathway data for enrichment
        pathway_data = ov.external.datacollect.collect_pathway_data("hsa04110")
        gsea_results = ov.bulk.pyGSEA(adata, pathway_data)
        """)
        
        print("✅ Workflow integration pattern demonstrated")
    
    def demo_format_conversion(self):
        """Demonstrate format conversion capabilities."""
        print("\n" + "="*60)
        print("📊 DEMO 3: Format Conversion")
        print("="*60)
        
        # Demo 3.1: Sample Data Preparation
        print("\n1️⃣ Sample Data Preparation")
        print("-" * 30)
        
        # Create sample datasets for demo
        sample_protein_data = {
            "entries": [
                {
                    "accession": "P04637",
                    "name": "Cellular tumor antigen p53",
                    "gene_name": "TP53",
                    "organism": "Homo sapiens",
                    "sequence": "MEEPQSDPSIEP...",  # Truncated for demo
                    "length": 393
                },
                {
                    "accession": "P21359", 
                    "name": "Neurofibromin",
                    "gene_name": "NF1",
                    "organism": "Homo sapiens", 
                    "sequence": "MAATAAAATS...",  # Truncated for demo
                    "length": 2839
                }
            ]
        }
        
        sample_expression_data = {
            "expression_matrix": [
                [1.5, 2.3, 0.8, 3.1],  # Sample 1
                [2.1, 1.9, 1.2, 2.8],  # Sample 2
                [0.9, 3.2, 2.1, 1.5]   # Sample 3
            ],
            "samples": {
                "sample1": {"condition": "control", "tissue": "brain"},
                "sample2": {"condition": "treatment", "tissue": "brain"},
                "sample3": {"condition": "control", "tissue": "liver"}
            },
            "genes": {
                "gene1": {"symbol": "TP53", "gene_id": "ENSG00000141510"},
                "gene2": {"symbol": "NF1", "gene_id": "ENSG00000196712"},
                "gene3": {"symbol": "EGFR", "gene_id": "ENSG00000146648"},
                "gene4": {"symbol": "MYC", "gene_id": "ENSG00000136997"}
            }
        }
        
        print("✅ Sample datasets prepared")
        
        # Demo 3.2: Pandas Conversion
        print("\n2️⃣ Pandas DataFrame Conversion")
        print("-" * 35)
        
        try:
            if hasattr(self.datacollect, 'to_pandas'):
                df = self.datacollect.to_pandas(sample_protein_data, "protein")
                print(f"✅ Protein data converted to pandas DataFrame")
                print(f"   Shape: {getattr(df, 'shape', 'Unknown')}")
                print(f"   Columns: {list(getattr(df, 'columns', []))}")
            else:
                print("⚠️ to_pandas function not available")
        except Exception as e:
            print(f"ℹ️ Pandas conversion note: {e}")
        
        # Demo 3.3: AnnData Conversion
        print("\n3️⃣ AnnData Conversion")
        print("-" * 22)
        
        try:
            if hasattr(self.datacollect, 'to_anndata'):
                # Check AnnData availability
                try:
                    import anndata
                    adata = self.datacollect.to_anndata(sample_expression_data)
                    
                    if adata is not None:
                        print(f"✅ Expression data converted to AnnData")
                        print(f"   Shape: {getattr(adata, 'shape', 'Unknown')}")
                        print(f"   Observations: {adata.n_obs if hasattr(adata, 'n_obs') else 'Unknown'}")
                        print(f"   Variables: {adata.n_vars if hasattr(adata, 'n_vars') else 'Unknown'}")
                    else:
                        print("ℹ️ AnnData conversion returned None (expected for sample data)")
                        
                except ImportError:
                    print("ℹ️ AnnData library not installed")
                    print("   Install with: pip install anndata")
            else:
                print("⚠️ to_anndata function not available")
        except Exception as e:
            print(f"ℹ️ AnnData conversion note: {e}")
        
        # Demo 3.4: MuData Conversion
        print("\n4️⃣ MuData Conversion")
        print("-" * 20)
        
        try:
            if hasattr(self.datacollect, 'to_mudata'):
                try:
                    import mudata
                    
                    multi_omics_data = {
                        'rna': sample_expression_data,
                        'protein': sample_protein_data
                    }
                    
                    mudata_obj = self.datacollect.to_mudata(multi_omics_data)
                    
                    if mudata_obj is not None:
                        print(f"✅ Multi-omics data converted to MuData")
                        print(f"   Modalities: {list(mudata_obj.mod.keys()) if hasattr(mudata_obj, 'mod') else 'Unknown'}")
                    else:
                        print("ℹ️ MuData conversion returned None (expected for sample data)")
                        
                except ImportError:
                    print("ℹ️ MuData library not installed")
                    print("   Install with: pip install mudata")
            else:
                print("⚠️ to_mudata function not available")
        except Exception as e:
            print(f"ℹ️ MuData conversion note: {e}")
    
    def demo_complete_workflow(self):
        """Demonstrate a complete analysis workflow."""
        print("\n" + "="*60)
        print("🔬 DEMO 4: Complete Analysis Workflow")
        print("="*60)
        
        print("\nThis demo shows how DataCollect integrates into a typical")
        print("bioinformatics analysis workflow with OmicVerse:")
        
        print("\n" + "🔹" * 50)
        print("WORKFLOW STEPS:")
        print("🔹" * 50)
        
        # Step 1: Data Collection
        print("\n📥 STEP 1: Data Collection")
        print("-" * 30)
        print("Goal: Collect multi-omics data for analysis")
        print()
        
        workflow_code = """
# Collect protein information
proteins_of_interest = ["P04637", "P21359", "P53_HUMAN"]
protein_data = ov.external.datacollect.collect_protein_data(
    proteins_of_interest, 
    source="uniprot",
    format="pandas"
)

# Collect gene expression data
expression_data = ov.external.datacollect.collect_expression_data(
    "GSE123456",  # Example GEO dataset
    format="anndata"  # OmicVerse standard format
)

# Collect pathway information
pathways = ["hsa04110", "hsa04151", "hsa04115"]  # Cell cycle related
pathway_data = []
for pathway_id in pathways:
    data = ov.external.datacollect.collect_pathway_data(
        pathway_id,
        source="kegg",
        format="pandas"
    )
    pathway_data.append(data)
"""
        
        print("Code example:")
        print(workflow_code)
        
        # Check if components are available
        available_components = []
        if hasattr(self.datacollect, 'collect_protein_data'):
            available_components.append("protein collection")
        if hasattr(self.datacollect, 'collect_expression_data'):
            available_components.append("expression collection") 
        if hasattr(self.datacollect, 'collect_pathway_data'):
            available_components.append("pathway collection")
        
        print(f"✅ Available components: {', '.join(available_components)}")
        
        # Step 2: Data Integration
        print("\n🔗 STEP 2: Data Integration with OmicVerse")
        print("-" * 45)
        print("Goal: Integrate collected data with OmicVerse analysis")
        
        integration_code = """
# Preprocess expression data with OmicVerse
ov.pp.preprocess(expression_data, mode='shiftlog|pearson', n_HVGs=2000)

# Quality control
ov.pp.qc(expression_data, tresh={'mito_perc': 0.2, 'n_UMIs': 500})

# Differential expression analysis
deg_results = ov.bulk.pyDEG(expression_data, group='condition')

# Pathway enrichment using collected pathways
gsea_results = ov.bulk.pyGSEA(expression_data, gene_sets=pathway_data)
"""
        
        print("Code example:")
        print(integration_code)
        print("✅ Integration pattern demonstrated")
        
        # Step 3: Advanced Analysis
        print("\n📊 STEP 3: Advanced Multi-Omics Analysis")
        print("-" * 40)
        print("Goal: Combine multiple data types for comprehensive analysis")
        
        advanced_code = """
# Create integrated dataset
integrated_data = {
    'expression': expression_data,
    'proteins': protein_data,
    'pathways': pathway_data
}

# Convert to MuData for multi-omics analysis
mudata_obj = ov.external.datacollect.to_mudata(integrated_data)

# Advanced OmicVerse analysis (conceptual)
# ov.tl.multi_omics_integration(mudata_obj)
# ov.pl.multi_omics_plot(mudata_obj)
"""
        
        print("Code example:")
        print(advanced_code)
        print("✅ Advanced analysis pattern demonstrated")
        
        # Step 4: Results and Validation
        print("\n📈 STEP 4: Results Validation")
        print("-" * 30)
        print("Goal: Validate findings with additional data")
        
        validation_code = """
# Validate DEGs with protein interaction data
for deg_gene in top_degs:
    interactions = ov.external.datacollect.collect_protein_data(
        deg_gene,
        source="string",
        format="pandas"
    )
    # Analyze interaction networks

# Cross-reference with clinical variants
for gene in candidate_genes:
    variants = ov.external.datacollect.collect_variant_data(
        gene,
        source="clinvar",
        format="pandas"
    )
    # Check for known clinical associations
"""
        
        print("Code example:")
        print(validation_code)
        print("✅ Validation workflow demonstrated")
        
        print("\n" + "🎯" * 50)
        print("WORKFLOW BENEFITS:")
        print("🎯" * 50)
        print("✅ Unified data access through single interface")
        print("✅ Automatic format conversion for OmicVerse compatibility")
        print("✅ Seamless integration with existing OmicVerse workflows")
        print("✅ Multi-omics data support (expression, proteins, pathways)")
        print("✅ Production-ready with error handling and validation")
    
    def run_demo(self, demo_name: str):
        """Run the specified demo."""
        if demo_name in ['all', 'basic']:
            self.demo_basic_collection()
        
        if demo_name in ['all', 'integration']:
            self.demo_omicverse_integration()
        
        if demo_name in ['all', 'formats']:
            self.demo_format_conversion()
        
        if demo_name in ['all', 'workflow']:
            self.demo_complete_workflow()
    
    def show_summary(self):
        """Show demo summary and next steps."""
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE")
        print("="*60)
        
        print("\n📚 What you've learned:")
        print("   ✅ How to access DataCollect through OmicVerse")
        print("   ✅ Basic data collection patterns")
        print("   ✅ Format conversion for OmicVerse compatibility")
        print("   ✅ Integration with OmicVerse analysis workflows")
        print("   ✅ Complete multi-omics analysis patterns")
        
        print("\n🚀 Next steps:")
        print("   1. Try the examples with real data")
        print("   2. Configure API keys for production use")
        print("   3. Explore the comprehensive documentation")
        print("   4. Run the full test suite for validation")
        
        print("\n📖 Documentation:")
        print("   • Main README: omicverse/external/datacollect/README.md")
        print("   • Tutorial: omicverse/external/datacollect/docs/OMICVERSE_TUTORIAL.md")
        print("   • API Reference: omicverse/external/datacollect/docs/OMICVERSE_API_REFERENCE.md")
        
        print("\n🧪 Testing:")
        print("   • Quick test: python test_ov_datacollect_quick.py")
        print("   • Full test suite: python test_omicverse_datacollect_complete.py")

def main():
    """Main demo execution."""
    parser = argparse.ArgumentParser(
        description="OmicVerse DataCollect Usage Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_ov_datacollect_usage.py                    # Run all demos
  python demo_ov_datacollect_usage.py --demo basic       # Basic usage only
  python demo_ov_datacollect_usage.py --demo integration # Integration examples
  python demo_ov_datacollect_usage.py --demo workflow    # Complete workflow
        """
    )
    
    parser.add_argument(
        '--demo', '-d',
        choices=['all', 'basic', 'integration', 'formats', 'workflow'],
        default='all',
        help='Demo to run (default: all)'
    )
    
    args = parser.parse_args()
    
    print("🚀 OmicVerse DataCollect Usage Demonstration")
    print("=" * 60)
    print("This demo shows practical examples of using DataCollect")
    print("with OmicVerse for bioinformatics data collection and analysis.")
    
    try:
        demo = OmicVerseDataCollectDemo()
        demo.run_demo(args.demo)
        demo.show_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Demo failed: {e}")
        print("\n💡 Make sure OmicVerse with DataCollect is properly installed")
        sys.exit(1)

if __name__ == "__main__":
    main()