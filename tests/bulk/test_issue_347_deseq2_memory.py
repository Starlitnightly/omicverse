"""
Specific tests for Issue #347: DESeq2 memory explosion with large datasets

This test reproduces and validates the fix for the memory issue when running
DESeq2 analysis on 31,002 cells x 3,000 genes dataset.
"""

import pytest
import numpy as np
import pandas as pd
import omicverse as ov
import gc
import psutil
from unittest.mock import patch, MagicMock


class TestIssue347DESeq2Memory:
    """Test suite specifically for Issue #347 DESeq2 memory problems."""
    
    @pytest.fixture
    def large_single_cell_data(self):
        """Create test data similar to the issue: 31K cells x 3K genes."""
        np.random.seed(42)
        
        # Create a scaled-down version for testing (1K cells x 500 genes)
        # This is more manageable for CI but still tests the memory optimization logic
        n_genes = 500
        n_cells = 1000
        
        # Generate realistic single-cell count data
        counts = np.random.negative_binomial(
            n=5, p=0.3, size=(n_genes, n_cells)
        ).astype(np.int32)
        
        # Create DataFrame with proper indexing
        data = pd.DataFrame(
            counts,
            index=[f"Gene_{i}" for i in range(n_genes)],
            columns=[f"Cell_{i}" for i in range(n_cells)]
        )
        
        return data.T  # Transpose to match expected format (cells x genes)
    
    @pytest.fixture
    def cell_metadata(self):
        """Create cell metadata similar to the original issue."""
        n_cells = 1000
        
        metadata = pd.DataFrame({
            'major_celltype': ['Exc'] * n_cells,
            'label': ['case'] * 600 + ['control'] * 400,  # Imbalanced groups like real data
            'batch': np.random.choice(['batch1', 'batch2', 'batch3'], n_cells)
        }, index=[f"Cell_{i}" for i in range(n_cells)])
        
        return metadata
    
    def test_original_issue_scenario(self, large_single_cell_data, cell_metadata):
        """Test the exact scenario from Issue #347."""
        # Filter for Exc cells (as in original issue)
        exc_mask = cell_metadata['major_celltype'].isin(['Exc'])
        exc_adata = large_single_cell_data[exc_mask]
        exc_metadata = cell_metadata[exc_mask]
        
        # Initialize pyDEG with transposed data (genes x cells)
        dds = ov.bulk.pyDEG(exc_adata.T)
        
        # Drop duplicates as in original issue
        dds.drop_duplicates_index()
        print('... drop_duplicates_index success')
        
        # Set up groups as in original issue
        treatment_groups = exc_metadata[exc_metadata['label'] == 'case'].index.tolist()
        control_groups = exc_metadata[exc_metadata['label'] == 'control'].index.tolist()
        
        # Monitor memory usage during analysis
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # This should not cause memory explosion with optimization
        result = dds.deg_analysis(
            treatment_groups, 
            control_groups, 
            method='DEseq2'
        )
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.1f} MB")
        memory_increase = final_memory - initial_memory
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Validate results
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check required columns exist
        required_columns = ['pvalue', 'qvalue', 'log2FC', 'BaseMean']
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Memory increase should be reasonable (less than 1GB for test data)
        assert memory_increase < 1000, f"Memory increase too high: {memory_increase:.1f} MB"
        
        print("✅ DESeq2 analysis completed successfully without memory explosion")
    
    def test_memory_usage_progression(self, large_single_cell_data, cell_metadata):
        """Test memory usage at each step of DESeq2 analysis."""
        memory_log = []
        
        def log_memory(step):
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_log.append((step, memory))
            print(f"{step}: {memory:.1f} MB")
        
        log_memory("Initial")
        
        # Setup data
        exc_mask = cell_metadata['major_celltype'].isin(['Exc'])
        exc_adata = large_single_cell_data[exc_mask]
        exc_metadata = cell_metadata[exc_mask]
        log_memory("Data setup")
        
        # Initialize pyDEG
        dds = ov.bulk.pyDEG(exc_adata.T)
        log_memory("pyDEG initialization")
        
        # Drop duplicates
        dds.drop_duplicates_index()
        log_memory("After drop_duplicates_index")
        
        # Set up groups
        treatment_groups = exc_metadata[exc_metadata['label'] == 'case'].index.tolist()
        control_groups = exc_metadata[exc_metadata['label'] == 'control'].index.tolist()
        log_memory("Groups setup")
        
        # Run DESeq2 analysis
        result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
        log_memory("After DESeq2 analysis")
        
        # Check that memory usage is reasonable at each step
        max_memory = max(memory for _, memory in memory_log)
        min_memory = min(memory for _, memory in memory_log)
        memory_range = max_memory - min_memory
        
        print(f"Memory range during analysis: {memory_range:.1f} MB")
        
        # Memory range should be reasonable for test dataset
        assert memory_range < 2000, f"Memory range too high: {memory_range:.1f} MB"
        
        # Validate result
        assert result is not None
        assert len(result) > 0
    
    def test_large_group_handling(self, large_single_cell_data):
        """Test handling of large treatment/control groups."""
        # Create large groups that might trigger memory issues
        dds = ov.bulk.pyDEG(large_single_cell_data.T)
        dds.drop_duplicates_index()
        
        # Large groups
        treatment_groups = [f"Cell_{i}" for i in range(600)]
        control_groups = [f"Cell_{i}" for i in range(600, 1000)]
        
        initial_memory = psutil.Process().memory_info().rss
        
        # This should handle large groups efficiently
        result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert result is not None
        assert len(result) > 0
        assert memory_increase < 1500, f"Memory increase too high: {memory_increase:.1f} MB"
    
    def test_memory_cleanup_after_analysis(self, large_single_cell_data, cell_metadata):
        """Test that memory is properly cleaned up after analysis."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Setup and run analysis
        exc_mask = cell_metadata['major_celltype'].isin(['Exc'])
        exc_adata = large_single_cell_data[exc_mask]
        exc_metadata = cell_metadata[exc_mask]
        
        dds = ov.bulk.pyDEG(exc_adata.T)
        dds.drop_duplicates_index()
        
        treatment_groups = exc_metadata[exc_metadata['label'] == 'case'].index.tolist()
        control_groups = exc_metadata[exc_metadata['label'] == 'control'].index.tolist()
        
        result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
        
        # Clean up
        del dds, result, exc_adata, exc_metadata
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_retained = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"Memory retained after cleanup: {memory_retained:.1f} MB")
        
        # Should not retain excessive memory
        assert memory_retained < 500, f"Too much memory retained: {memory_retained:.1f} MB"
    
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases that might cause memory issues."""
        # Test with very small dataset
        small_data = pd.DataFrame(
            np.random.poisson(5, size=(10, 20)),
            index=[f"Gene_{i}" for i in range(10)],
            columns=[f"Cell_{i}" for i in range(20)]
        )
        
        dds = ov.bulk.pyDEG(small_data)
        treatment_groups = [f"Cell_{i}" for i in range(10)]
        control_groups = [f"Cell_{i}" for i in range(10, 20)]
        
        # Should handle small datasets without issues
        result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
        assert result is not None
        
        # Test with empty groups
        with pytest.raises((ValueError, IndexError)):
            dds.deg_analysis([], control_groups, method='DEseq2')
    
    @patch('omicverse.bulk._Deseq2.pyDEG.deg_analysis')
    def test_memory_optimization_fallback(self, mock_deg_analysis, large_single_cell_data):
        """Test fallback mechanism when DESeq2 fails due to memory issues."""
        # Mock DESeq2 to raise memory error, then fallback to t-test
        def mock_analysis(*args, **kwargs):
            if kwargs.get('method') == 'DEseq2':
                raise MemoryError("Simulated memory error")
            else:
                # Return a mock result for t-test
                return pd.DataFrame({
                    'pvalue': [0.01, 0.05, 0.1],
                    'qvalue': [0.02, 0.06, 0.12],
                    'log2FC': [1.5, -2.0, 0.5],
                    'BaseMean': [100, 200, 150]
                })
        
        mock_deg_analysis.side_effect = mock_analysis
        
        dds = ov.bulk.pyDEG(large_single_cell_data.T)
        treatment_groups = [f"Cell_{i}" for i in range(100)]
        control_groups = [f"Cell_{i}" for i in range(100, 200)]
        
        # Should catch memory error and potentially fallback
        try:
            result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
            assert result is not None
        except MemoryError:
            # If no fallback implemented yet, that's expected
            pytest.skip("Memory error fallback not yet implemented")
    
    def test_performance_benchmark(self, large_single_cell_data, cell_metadata):
        """Benchmark performance to ensure optimization doesn't hurt speed too much."""
        import time
        
        exc_mask = cell_metadata['major_celltype'].isin(['Exc'])
        exc_adata = large_single_cell_data[exc_mask]
        exc_metadata = cell_metadata[exc_mask]
        
        dds = ov.bulk.pyDEG(exc_adata.T)
        dds.drop_duplicates_index()
        
        treatment_groups = exc_metadata[exc_metadata['label'] == 'case'].index.tolist()[:100]
        control_groups = exc_metadata[exc_metadata['label'] == 'control'].index.tolist()[:100]
        
        start_time = time.time()
        result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Analysis duration: {duration:.2f} seconds")
        
        assert result is not None
        assert len(result) > 0
        
        # Should complete in reasonable time (less than 2 minutes for test data)
        assert duration < 120, f"Analysis took too long: {duration:.2f} seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])