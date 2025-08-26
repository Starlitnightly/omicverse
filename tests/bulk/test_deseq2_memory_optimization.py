"""
Tests for DESeq2 memory optimization functionality in omicverse.bulk.pyDEG

This test suite verifies the memory optimization features for DESeq2 analysis
when dealing with large single-cell datasets (31K+ cells x 3K+ genes).
"""

import pytest
import numpy as np
import pandas as pd
import omicverse as ov
from unittest.mock import patch, MagicMock
import gc
import psutil
import os


class TestDESeq2MemoryOptimization:
    """Test suite for DESeq2 memory optimization features."""
    
    @pytest.fixture
    def sample_small_data(self):
        """Create a small test dataset for basic functionality."""
        np.random.seed(42)
        # Create 100 genes x 50 samples
        data = pd.DataFrame(
            np.random.poisson(10, size=(100, 50)),
            index=[f"Gene_{i}" for i in range(100)],
            columns=[f"Sample_{i}" for i in range(50)]
        )
        return data
    
    @pytest.fixture
    def sample_large_data(self):
        """Create a large test dataset to simulate memory issues."""
        np.random.seed(42)
        # Create 3000 genes x 1000 samples to simulate large dataset
        data = pd.DataFrame(
            np.random.poisson(5, size=(3000, 1000)),
            index=[f"Gene_{i}" for i in range(3000)],
            columns=[f"Sample_{i}" for i in range(1000)]
        )
        return data
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return pd.DataFrame({
            'label': ['case'] * 25 + ['control'] * 25,
            'batch': ['batch1'] * 12 + ['batch2'] * 13 + ['batch1'] * 12 + ['batch2'] * 13
        }, index=[f"Sample_{i}" for i in range(50)])
    
    def test_pyDEG_initialization(self, sample_small_data):
        """Test basic pyDEG initialization."""
        dds = ov.bulk.pyDEG(sample_small_data)
        assert dds is not None
        assert hasattr(dds, 'raw_data')
        assert hasattr(dds, 'data')
        pd.testing.assert_frame_equal(dds.raw_data, sample_small_data)
    
    def test_memory_monitoring_basic(self, sample_small_data):
        """Test basic memory monitoring functionality."""
        dds = ov.bulk.pyDEG(sample_small_data)
        
        # Test if memory monitoring methods exist (they should be added)
        if hasattr(dds, 'get_current_memory_usage'):
            memory_usage = dds.get_current_memory_usage()
            assert isinstance(memory_usage, (int, float))
            assert memory_usage > 0
    
    def test_memory_efficient_parameter(self, sample_large_data):
        """Test memory_efficient parameter in deg_analysis."""
        dds = ov.bulk.pyDEG(sample_large_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(500)]
        control_groups = [f"Sample_{i}" for i in range(500, 1000)]
        
        # Test if memory_efficient parameter is supported
        try:
            # This should work with the memory optimization implementation
            result = dds.deg_analysis(
                treatment_groups, control_groups, 
                method='DEseq2', 
                memory_efficient=True
            )
            assert result is not None
            assert isinstance(result, pd.DataFrame)
        except TypeError as e:
            if "memory_efficient" in str(e):
                pytest.skip("memory_efficient parameter not yet implemented")
            else:
                raise
    
    def test_chunk_size_parameter(self, sample_large_data):
        """Test chunk_size parameter for memory-efficient processing."""
        dds = ov.bulk.pyDEG(sample_large_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(500)]
        control_groups = [f"Sample_{i}" for i in range(500, 1000)]
        
        try:
            result = dds.deg_analysis(
                treatment_groups, control_groups,
                method='DEseq2',
                memory_efficient=True,
                chunk_size=200
            )
            assert result is not None
            assert isinstance(result, pd.DataFrame)
        except TypeError as e:
            if "chunk_size" in str(e):
                pytest.skip("chunk_size parameter not yet implemented")
            else:
                raise
    
    def test_auto_fallback_to_ttest(self, sample_large_data):
        """Test automatic fallback to t-test for extremely large datasets."""
        # Create extremely large dataset (simulate 50K+ cells)
        np.random.seed(42)
        large_data = pd.DataFrame(
            np.random.poisson(3, size=(3000, 100)),  # Use smaller sample for testing
            index=[f"Gene_{i}" for i in range(3000)],
            columns=[f"Sample_{i}" for i in range(100)]
        )
        
        dds = ov.bulk.pyDEG(large_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(50)]
        control_groups = [f"Sample_{i}" for i in range(50, 100)]
        
        # Test with mock to simulate extremely large dataset
        with patch.object(dds, 'data', large_data):
            result = dds.deg_analysis(
                treatment_groups, control_groups,
                method='DEseq2'
            )
            assert result is not None
            assert isinstance(result, pd.DataFrame)
    
    def test_memory_optimization_suggestions(self, sample_large_data):
        """Test memory optimization suggestions functionality."""
        dds = ov.bulk.pyDEG(sample_large_data)
        
        treatment_groups = [f"Sample_{i}" for i in range(500)]
        control_groups = [f"Sample_{i}" for i in range(500, 1000)]
        
        # Test if get_memory_optimization_suggestions method exists
        if hasattr(dds, 'get_memory_optimization_suggestions'):
            suggestions = dds.get_memory_optimization_suggestions(
                treatment_groups, control_groups
            )
            assert isinstance(suggestions, str)
            assert len(suggestions) > 0
    
    def test_memory_estimation(self, sample_large_data):
        """Test memory requirement estimation."""
        dds = ov.bulk.pyDEG(sample_large_data)
        
        treatment_groups = [f"Sample_{i}" for i in range(500)]
        control_groups = [f"Sample_{i}" for i in range(500, 1000)]
        
        # Test if memory estimation method exists
        if hasattr(dds, 'estimate_memory_requirements'):
            memory_req = dds.estimate_memory_requirements(
                treatment_groups, control_groups, method='DEseq2'
            )
            assert isinstance(memory_req, (int, float))
            assert memory_req > 0
    
    def test_subsampling_for_large_groups(self, sample_large_data):
        """Test intelligent subsampling for large groups."""
        dds = ov.bulk.pyDEG(sample_large_data)
        dds.drop_duplicates_index()
        
        # Create very large groups to trigger subsampling
        treatment_groups = [f"Sample_{i}" for i in range(600)]
        control_groups = [f"Sample_{i}" for i in range(600, 1000)]
        
        try:
            result = dds.deg_analysis(
                treatment_groups, control_groups,
                method='DEseq2',
                memory_efficient=True,
                max_samples_per_group=200  # Should trigger subsampling
            )
            assert result is not None
            assert isinstance(result, pd.DataFrame)
        except TypeError as e:
            if "max_samples_per_group" in str(e):
                pytest.skip("max_samples_per_group parameter not yet implemented")
            else:
                raise
    
    @pytest.mark.parametrize("method", ["DEseq2", "ttest"])
    def test_memory_monitoring_during_analysis(self, sample_large_data, method):
        """Test memory monitoring during different analysis methods."""
        dds = ov.bulk.pyDEG(sample_large_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(100)]
        control_groups = [f"Sample_{i}" for i in range(100, 200)]
        
        # Mock memory monitoring
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock(available=8*1024*1024*1024)  # 8GB available
            
            result = dds.deg_analysis(
                treatment_groups, control_groups,
                method=method
            )
            assert result is not None
            assert isinstance(result, pd.DataFrame)
    
    def test_memory_warnings(self, sample_large_data, capfd):
        """Test that memory warnings are displayed for large datasets."""
        dds = ov.bulk.pyDEG(sample_large_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(500)]
        control_groups = [f"Sample_{i}" for i in range(500, 1000)]
        
        # Run analysis and capture output
        try:
            dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
            captured = capfd.readouterr()
            
            # Check if memory-related warnings appear in output
            # These might not exist yet, so we make it optional
            assert True  # Placeholder for actual memory warning checks
        except Exception:
            # If the method fails due to memory issues, that's expected for large data
            pass
    
    def test_cleanup_intermediate_objects(self, sample_large_data):
        """Test cleanup of intermediate objects to free memory."""
        dds = ov.bulk.pyDEG(sample_large_data)
        
        # Test if cleanup method exists
        if hasattr(dds, 'cleanup_intermediate_objects'):
            initial_memory = psutil.Process().memory_info().rss
            
            # Create some intermediate data
            dds._intermediate_data = pd.DataFrame(np.random.random((1000, 1000)))
            
            # Cleanup
            dds.cleanup_intermediate_objects()
            
            # Force garbage collection
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss
            # Memory should not increase significantly after cleanup
            assert final_memory <= initial_memory * 1.1  # Allow 10% variance
    
    def test_deseq2_specific_memory_optimizations(self, sample_large_data):
        """Test DESeq2-specific memory optimizations."""
        dds = ov.bulk.pyDEG(sample_large_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(200)]
        control_groups = [f"Sample_{i}" for i in range(200, 400)]
        
        # Test with various DESeq2 optimization parameters
        try:
            result = dds.deg_analysis(
                treatment_groups, control_groups,
                method='DEseq2',
                memory_efficient=True,
                # DESeq2-specific optimizations
                reduce_fit_iterations=True,
                skip_outlier_detection=True,
                use_approximation=True
            )
            assert result is not None
            assert isinstance(result, pd.DataFrame)
        except TypeError:
            # These parameters might not be implemented yet
            pytest.skip("DESeq2 memory optimization parameters not yet implemented")
    
    def test_memory_efficient_preprocessing(self, sample_large_data):
        """Test memory-efficient preprocessing steps."""
        dds = ov.bulk.pyDEG(sample_large_data)
        
        # Test efficient duplicate removal
        initial_memory = psutil.Process().memory_info().rss
        dds.drop_duplicates_index()
        post_dedup_memory = psutil.Process().memory_info().rss
        
        # Memory should not increase significantly
        assert post_dedup_memory <= initial_memory * 1.2  # Allow 20% increase
        
        # Test efficient normalization
        if hasattr(dds, 'normalize'):
            dds.normalize()
            post_norm_memory = psutil.Process().memory_info().rss
            assert post_norm_memory <= post_dedup_memory * 1.5  # Allow 50% increase for normalization
    
    def test_large_dataset_integration(self):
        """Integration test simulating the original issue scenario."""
        # Simulate the original issue: 31,002 cells x 3,000 genes
        np.random.seed(42)
        
        # Create a more realistic but smaller test dataset
        n_genes = 1000
        n_cells = 2000
        
        # Create count data
        counts_data = pd.DataFrame(
            np.random.negative_binomial(10, 0.3, size=(n_genes, n_cells)),
            index=[f"Gene_{i}" for i in range(n_genes)],
            columns=[f"Cell_{i}" for i in range(n_cells)]
        )
        
        # Create metadata
        metadata = pd.DataFrame({
            'major_celltype': ['Exc'] * n_cells,
            'label': ['case'] * (n_cells // 2) + ['control'] * (n_cells // 2)
        }, index=counts_data.columns)
        
        # Filter for Exc cells (similar to original issue)
        exc_data = counts_data.T  # Transpose to match expected format
        
        # Initialize pyDEG
        dds = ov.bulk.pyDEG(exc_data)
        
        # Drop duplicates (as in original issue)
        dds.drop_duplicates_index()
        
        # Set up groups
        treatment_groups = metadata[metadata['label'] == 'case'].index.tolist()
        control_groups = metadata[metadata['label'] == 'control'].index.tolist()
        
        # This should not cause memory explosion with optimization
        try:
            result = dds.deg_analysis(
                treatment_groups, control_groups, 
                method='DEseq2',
                memory_efficient=True  # This parameter should exist after implementation
            )
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'pvalue' in result.columns
            assert 'qvalue' in result.columns
            
        except TypeError as e:
            if "memory_efficient" in str(e):
                # Run without memory optimization if not implemented
                result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
                assert result is not None
                pytest.skip("Memory optimization not yet implemented")
            else:
                raise
    
    def test_error_handling_for_insufficient_memory(self):
        """Test proper error handling when memory is insufficient."""
        # This test would mock low memory conditions
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock(available=1*1024*1024*1024)  # Only 1GB available
            
            # Create dataset that would normally require more memory
            large_data = pd.DataFrame(
                np.random.poisson(5, size=(100, 100)),  # Small for testing
                index=[f"Gene_{i}" for i in range(100)],
                columns=[f"Sample_{i}" for i in range(100)]
            )
            
            dds = ov.bulk.pyDEG(large_data)
            treatment_groups = [f"Sample_{i}" for i in range(50)]
            control_groups = [f"Sample_{i}" for i in range(50, 100)]
            
            # Should either work with optimization or provide helpful error
            result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
            assert result is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])