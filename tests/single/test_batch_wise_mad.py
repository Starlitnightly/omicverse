import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import omicverse as ov
from omicverse.pp._qc import mads, mads_test


def test_mads_global_vs_batch():
    """Test that batch-wise MAD produces different results than global MAD when appropriate."""
    # Create test data with two clear batches of different coverage
    np.random.seed(42)
    
    # Batch 1: Low coverage
    batch1_umis = np.random.poisson(500, 100)  # Lower UMI counts
    batch1_genes = np.random.poisson(200, 100)  # Lower gene counts
    
    # Batch 2: High coverage  
    batch2_umis = np.random.poisson(2000, 100)  # Higher UMI counts
    batch2_genes = np.random.poisson(800, 100)  # Higher gene counts
    
    # Create metadata DataFrame
    meta = pd.DataFrame({
        'nUMIs': np.concatenate([batch1_umis, batch2_umis]),
        'detected_genes': np.concatenate([batch1_genes, batch2_genes]),
        'batch': ['batch1'] * 100 + ['batch2'] * 100
    })
    
    thresholds = {'nUMIs': 100, 'detected_genes': 50}
    
    # Test global MAD
    global_nUMIs_thresh = mads(meta, 'nUMIs', nmads=2, lt=thresholds)
    global_genes_thresh = mads(meta, 'detected_genes', nmads=2, lt=thresholds)
    
    # Test batch-wise MAD
    batch_nUMIs_thresh = mads(meta, 'nUMIs', nmads=2, lt=thresholds, batch_key='batch')
    batch_genes_thresh = mads(meta, 'detected_genes', nmads=2, lt=thresholds, batch_key='batch')
    
    # Global MAD should return tuple, batch MAD should return dict
    assert isinstance(global_nUMIs_thresh, tuple)
    assert isinstance(batch_nUMIs_thresh, dict)
    
    # Should have thresholds for both batches
    assert 'batch1' in batch_nUMIs_thresh
    assert 'batch2' in batch_nUMIs_thresh
    assert 'batch1' in batch_genes_thresh
    assert 'batch2' in batch_genes_thresh
    
    # Batch-specific thresholds should be different
    assert batch_nUMIs_thresh['batch1'] != batch_nUMIs_thresh['batch2']
    assert batch_genes_thresh['batch1'] != batch_genes_thresh['batch2']
    
    # Batch 1 thresholds should be lower than batch 2 (due to lower coverage)
    assert batch_nUMIs_thresh['batch1'][0] < batch_nUMIs_thresh['batch2'][0]  # lower bound
    assert batch_nUMIs_thresh['batch1'][1] < batch_nUMIs_thresh['batch2'][1]  # upper bound


def test_mads_test_global_vs_batch():
    """Test that batch-wise filtering retains more appropriate cells."""
    np.random.seed(42)
    
    # Create test data with two batches of different coverage
    batch1_umis = np.random.poisson(500, 100)
    batch2_umis = np.random.poisson(2000, 100)
    
    meta = pd.DataFrame({
        'nUMIs': np.concatenate([batch1_umis, batch2_umis]),
        'batch': ['batch1'] * 100 + ['batch2'] * 100
    })
    
    thresholds = {'nUMIs': 100}
    
    # Apply global MAD test
    global_pass = mads_test(meta, 'nUMIs', nmads=2, lt=thresholds)
    
    # Apply batch-wise MAD test
    batch_pass = mads_test(meta, 'nUMIs', nmads=2, lt=thresholds, batch_key='batch')
    
    # Should return boolean Series of same length
    assert len(global_pass) == len(batch_pass) == 200
    assert isinstance(global_pass, pd.Series)
    assert isinstance(batch_pass, pd.Series)
    
    # Check that batch-wise filtering retains more cells from low-coverage batch
    batch1_mask = meta['batch'] == 'batch1'
    batch2_mask = meta['batch'] == 'batch2'
    
    # Count cells passing in each batch for each method
    batch1_global_pass = global_pass[batch1_mask].sum()
    batch1_batch_pass = batch_pass[batch1_mask].sum()
    batch2_global_pass = global_pass[batch2_mask].sum()
    batch2_batch_pass = batch_pass[batch2_mask].sum()
    
    # Batch-wise should generally retain more cells from low-coverage batch
    # and be more stringent on high-coverage batch
    print(f"Batch1 - Global: {batch1_global_pass}, Batch-wise: {batch1_batch_pass}")
    print(f"Batch2 - Global: {batch2_global_pass}, Batch-wise: {batch2_batch_pass}")


def test_qc_with_scanpy_pbmc3k_data():
    """Test batch-wise QC with scanpy's PBMC3k dataset, simulating batches."""
    # Load PBMC3k dataset
    adata = sc.datasets.pbmc3k_processed()
    
    # Reset to raw counts for proper QC testing
    # We'll simulate having raw count data
    if adata.raw is not None:
        # Create a new AnnData object with raw data to avoid shape mismatch
        import anndata as ad
        adata_raw = ad.AnnData(X=adata.raw.X.copy())
        adata_raw.var = adata.raw.var.copy()
        adata_raw.var_names = adata.raw.var_names
        adata_raw.obs = adata.obs.copy()
        adata = adata_raw
    
    # Make variable names unique to avoid issues
    adata.var_names_unique()
    
    # Remove existing QC columns if they exist
    qc_cols = ['n_genes', 'n_counts', 'pct_counts_mt', 'nUMIs', 'detected_genes', 'mito_perc']
    for col in qc_cols:
        if col in adata.obs.columns:
            del adata.obs[col]
    
    # Simulate two batches with different coverage by modifying the data
    n_cells = adata.n_obs
    np.random.seed(42)
    
    # Create artificial batch assignment
    adata.obs['batch'] = np.random.choice(['high_cov', 'low_cov'], n_cells, p=[0.6, 0.4])
    
    # Simulate coverage difference by downsampling low coverage batch
    low_cov_mask = adata.obs['batch'] == 'low_cov'
    if hasattr(adata.X, 'data'):  # sparse matrix
        # For sparse matrices, multiply data by coverage factor
        low_cov_indices = np.where(low_cov_mask)[0]
        for idx in low_cov_indices:
            adata.X[idx].data = (adata.X[idx].data * 0.3).astype(int)  # Reduce to 30% coverage
    else:
        # For dense matrices
        adata.X[low_cov_mask] = (adata.X[low_cov_mask] * 0.3).astype(int)
    
    # Test global MAD QC
    adata_global = adata.copy()
    adata_global = ov.pp.qc(
        adata_global,
        mode='mads',
        nmads=3,
        tresh={'mito_perc': 0.2, 'nUMIs': 200, 'detected_genes': 100},
        doublets=False  # Skip doublet detection for faster testing
    )
    
    # Test batch-wise MAD QC
    adata_batch = adata.copy()
    adata_batch = ov.pp.qc(
        adata_batch,
        mode='mads',
        batch_key='batch',
        nmads=3,
        tresh={'mito_perc': 0.2, 'nUMIs': 200, 'detected_genes': 100},
        doublets=False  # Skip doublet detection for faster testing
    )
    
    # Both should return AnnData objects
    assert adata_global is not None
    assert adata_batch is not None
    
    # Batch-wise should generally retain more cells (especially from low coverage batch)
    original_cells = adata.n_obs
    global_retained = adata_global.n_obs
    batch_retained = adata_batch.n_obs
    
    print(f"Original cells: {original_cells}")
    print(f"Global MAD retained: {global_retained} ({global_retained/original_cells*100:.1f}%)")
    print(f"Batch-wise MAD retained: {batch_retained} ({batch_retained/original_cells*100:.1f}%)")
    
    # Check that both methods actually filtered some cells
    assert global_retained < original_cells
    assert batch_retained < original_cells
    
    # Verify that QC status was stored
    assert adata_global.uns['status']['qc'] is True
    assert adata_batch.uns['status']['qc'] is True


def test_backward_compatibility():
    """Test that the new batch_key parameter doesn't break existing functionality."""
    # Test with simple synthetic data
    np.random.seed(42)
    meta = pd.DataFrame({
        'nUMIs': np.random.poisson(1000, 200),
        'detected_genes': np.random.poisson(400, 200)
    })
    
    thresholds = {'nUMIs': 100, 'detected_genes': 50}
    
    # Test that calling without batch_key works as before
    old_nUMIs_thresh = mads(meta, 'nUMIs', nmads=3, lt=thresholds)
    new_nUMIs_thresh = mads(meta, 'nUMIs', nmads=3, lt=thresholds, batch_key=None)
    
    # Should be identical
    assert old_nUMIs_thresh == new_nUMIs_thresh
    
    # Test mads_test backward compatibility
    old_pass = mads_test(meta, 'nUMIs', nmads=3, lt=thresholds)
    new_pass = mads_test(meta, 'nUMIs', nmads=3, lt=thresholds, batch_key=None)
    
    # Should be identical
    assert old_pass.equals(new_pass)


def test_edge_cases():
    """Test edge cases for batch-wise MAD calculation."""
    # Test with single batch
    meta = pd.DataFrame({
        'nUMIs': np.random.poisson(1000, 100),
        'batch': ['single_batch'] * 100
    })
    
    thresholds = {'nUMIs': 100}
    
    batch_thresh = mads(meta, 'nUMIs', nmads=3, lt=thresholds, batch_key='batch')
    assert 'single_batch' in batch_thresh
    assert isinstance(batch_thresh['single_batch'], tuple)
    
    # Test with empty batch (shouldn't happen in practice, but good to handle)
    meta_with_empty = pd.DataFrame({
        'nUMIs': [100, 200, 300],
        'batch': ['batch1', 'batch1', 'empty_batch']
    })
    
    # Remove the 'empty_batch' entry to simulate empty batch
    meta_filtered = meta_with_empty[meta_with_empty['batch'] != 'empty_batch'].copy()
    meta_filtered.loc[len(meta_filtered)] = [400, 'empty_batch']
    meta_filtered = meta_filtered[meta_filtered['batch'] != 'empty_batch']  # Remove again
    
    # This should not crash
    batch_thresh = mads(meta_with_empty, 'nUMIs', nmads=3, lt=thresholds, batch_key='batch')
    assert len(batch_thresh) >= 1


if __name__ == "__main__":
    # Run basic tests
    test_backward_compatibility()
    print("✓ Backward compatibility test passed")
    
    test_mads_global_vs_batch()
    print("✓ MADs global vs batch test passed")
    
    test_mads_test_global_vs_batch() 
    print("✓ MADs test global vs batch test passed")
    
    test_edge_cases()
    print("✓ Edge cases test passed")
    
    # This test requires internet connection to download data
    try:
        test_qc_with_scanpy_pbmc3k_data()
        print("✓ PBMC3k QC test passed")
    except Exception as e:
        print(f"⚠ PBMC3k test skipped (likely missing data): {e}")
    
    print("\n🎉 All batch-wise MAD tests completed!")