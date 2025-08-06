#!/usr/bin/env python3
"""
UCE (Universal Cell Embeddings) Usage Examples
===============================================

This script demonstrates how to use UCE with external asset file paths
including token embeddings, protein embeddings, and species mappings.
"""

import numpy as np
import scanpy as sc
from omicverse.external.scllm import (
    SCLLMManager, 
    load_uce,
    get_embeddings_with_uce,
    end_to_end_uce_embedding,
    integrate_with_uce
)

# =============================================================================
# UCE Asset File Paths Configuration
# =============================================================================

# UCE model and asset file paths (replace with your actual paths)
UCE_MODEL_PATH = "/path/to/uce/model.torch"

# UCE asset files (these must be provided externally)
uce_asset_paths = {
    'token_file': '/path/to/token_embeddings.torch',
    'protein_embeddings_dir': '/path/to/protein_embeddings/',
    'spec_chrom_csv_path': '/path/to/species_chrom.csv',
    'offset_pkl_path': '/path/to/species_offsets.pkl'
}

# UCE model configuration
uce_config = {
    'species': 'human',
    'batch_size': 25,
    'nlayers': 4,
    'output_dim': 1280,
    'device': 'cpu'  # or 'cuda' if available
}

print("UCE Configuration:")
print("=" * 50)
for k, v in uce_config.items():
    print(f"  {k}: {v}")

print("\nUCE Asset Paths:")
print("=" * 50)
for k, v in uce_asset_paths.items():
    print(f"  {k}: {v}")

# =============================================================================
# Example 1: Using SCLLMManager with Asset Paths
# =============================================================================

def example_scllm_manager(adata):
    """Example using SCLLMManager with external asset paths."""
    print("\n" + "="*60)
    print("Example 1: SCLLMManager with Asset Paths")
    print("="*60)
    
    # Create manager with asset paths
    manager = SCLLMManager(
        model_type="uce",
        model_path=UCE_MODEL_PATH,
        device=uce_config['device'],
        # Pass asset file paths
        token_file=uce_asset_paths['token_file'],
        protein_embeddings_dir=uce_asset_paths['protein_embeddings_dir'],
        spec_chrom_csv_path=uce_asset_paths['spec_chrom_csv_path'],
        offset_pkl_path=uce_asset_paths['offset_pkl_path'],
        # Pass other config
        **uce_config
    )
    
    print("✓ SCLLMManager created with asset paths")
    
    # Extract embeddings
    embeddings = manager.get_embeddings(adata)
    print(f"✓ Embeddings extracted: {embeddings.shape}")
    
    # Batch integration
    if 'batch' in adata.obs:
        integration_results = manager.integrate(
            adata, 
            batch_key='batch', 
            correction_method='harmony'
        )
        print(f"✓ Batch integration completed: {integration_results['embeddings'].shape}")
    
    return manager


# =============================================================================
# Example 2: Using Load UCE Convenience Function
# =============================================================================

def example_load_uce(adata):
    """Example using load_uce convenience function."""
    print("\n" + "="*60)
    print("Example 2: load_uce Convenience Function")
    print("="*60)
    
    # Load UCE model with asset paths
    uce_model = load_uce(
        model_path=UCE_MODEL_PATH,
        device=uce_config['device'],
        species=uce_config['species'],
        # Asset paths
        token_file=uce_asset_paths['token_file'],
        protein_embeddings_dir=uce_asset_paths['protein_embeddings_dir'],
        spec_chrom_csv_path=uce_asset_paths['spec_chrom_csv_path'],
        offset_pkl_path=uce_asset_paths['offset_pkl_path'],
        # Other config
        **{k: v for k, v in uce_config.items() if k != 'species'}
    )
    
    print("✓ UCE model loaded with asset paths")
    
    # Get embeddings
    embeddings = uce_model.get_embeddings(adata)
    print(f"✓ Embeddings extracted: {embeddings.shape}")
    
    return uce_model


# =============================================================================
# Example 3: Direct Embedding Extraction
# =============================================================================

def example_direct_embeddings(adata):
    """Example using get_embeddings_with_uce function."""
    print("\n" + "="*60)
    print("Example 3: Direct Embedding Extraction")
    print("="*60)
    
    # Get embeddings directly with asset paths
    embeddings = get_embeddings_with_uce(
        adata=adata,
        model_path=UCE_MODEL_PATH,
        device=uce_config['device'],
        species=uce_config['species'],
        # Asset paths
        token_file=uce_asset_paths['token_file'],
        protein_embeddings_dir=uce_asset_paths['protein_embeddings_dir'],
        spec_chrom_csv_path=uce_asset_paths['spec_chrom_csv_path'],
        offset_pkl_path=uce_asset_paths['offset_pkl_path']
    )
    
    print(f"✓ Direct embeddings extracted: {embeddings.shape}")
    return embeddings


# =============================================================================
# Example 4: End-to-End Workflow
# =============================================================================

def example_end_to_end(adata):
    """Example using end-to-end workflow."""
    print("\n" + "="*60)
    print("Example 4: End-to-End Workflow")
    print("="*60)
    
    # Run complete workflow
    results = end_to_end_uce_embedding(
        adata=adata.copy(),
        model_path=UCE_MODEL_PATH,
        device=uce_config['device'],
        species=uce_config['species'],
        # Asset paths
        token_file=uce_asset_paths['token_file'],
        protein_embeddings_dir=uce_asset_paths['protein_embeddings_dir'],
        spec_chrom_csv_path=uce_asset_paths['spec_chrom_csv_path'],
        offset_pkl_path=uce_asset_paths['offset_pkl_path'],
        # Optional: save embeddings
        save_embeddings_path="uce_embeddings.npy"
    )
    
    print(f"✓ End-to-end workflow completed")
    print(f"  Results keys: {list(results.keys())}")
    return results


# =============================================================================
# Example 5: Batch Integration Workflow
# =============================================================================

def example_integration_workflow(adata):
    """Example using integration workflow."""
    print("\n" + "="*60)
    print("Example 5: Batch Integration Workflow")
    print("="*60)
    
    # Add batch information if not present
    if 'batch' not in adata.obs:
        n_cells = adata.n_obs
        adata.obs['batch'] = ['batch1'] * (n_cells // 2) + ['batch2'] * (n_cells - n_cells // 2)
    
    # Run integration workflow
    integration_results = integrate_with_uce(
        query_adata=adata.copy(),
        model_path=UCE_MODEL_PATH,
        batch_key='batch',
        correction_method='harmony',
        device=uce_config['device'],
        species=uce_config['species'],
        # Asset paths
        token_file=uce_asset_paths['token_file'],
        protein_embeddings_dir=uce_asset_paths['protein_embeddings_dir'],
        spec_chrom_csv_path=uce_asset_paths['spec_chrom_csv_path'],
        offset_pkl_path=uce_asset_paths['offset_pkl_path']
    )
    
    print(f"✓ Integration workflow completed")
    print(f"  Results keys: {list(integration_results.keys())}")
    return integration_results


# =============================================================================
# Main Demo Function
# =============================================================================

def main():
    """Main function to run all examples."""
    print("UCE Usage Examples with External Asset Paths")
    print("=" * 80)
    
    # Load sample data
    print("Loading sample data...")
    try:
        adata = sc.datasets.pbmc3k_processed()
        print(f"✓ Sample data loaded: {adata.shape}")
    except Exception as e:
        print(f"⚠️ Could not load sample data: {e}")
        print("Creating mock data...")
        # Create mock data
        n_cells, n_genes = 100, 500
        X = np.random.poisson(2, (n_cells, n_genes)).astype(np.float32)
        adata = sc.AnnData(X=X)
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        print(f"✓ Mock data created: {adata.shape}")
    
    # Run examples (Note: These will fail without actual UCE model files)
    try:
        print("\n🔧 Running UCE examples (will show expected interface)...")
        
        # Example 1: SCLLMManager
        manager = example_scllm_manager(adata)
        
        # Example 2: Load UCE
        model = example_load_uce(adata)
        
        # Example 3: Direct embeddings
        embeddings = example_direct_embeddings(adata)
        
        # Example 4: End-to-end
        results = example_end_to_end(adata)
        
        # Example 5: Integration
        integration = example_integration_workflow(adata)
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n⚠️ Examples failed (expected without actual UCE files): {e}")
        print("\n📋 This demonstrates the interface - replace paths with actual UCE files to run.")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: How to Use UCE with External Asset Paths")
    print("="*80)
    print("\n✅ All UCE functions now support external asset file paths:")
    print("  • token_file: Path to token embeddings (.torch)")
    print("  • protein_embeddings_dir: Directory with protein embeddings")
    print("  • spec_chrom_csv_path: Species chromosome mapping (.csv)")
    print("  • offset_pkl_path: Species offsets file (.pkl)")
    
    print("\n💡 Usage Tips:")
    print("  • Provide absolute paths to avoid file not found errors")
    print("  • Ensure all asset files match the model version")
    print("  • Check species compatibility between model and data")
    print("  • Use appropriate batch sizes based on available memory")
    
    print("\n🔗 Next Steps:")
    print("  • Download UCE model files from the official repository")
    print("  • Update file paths in this script with actual locations")
    print("  • Test with your own single-cell data")


if __name__ == "__main__":
    main()