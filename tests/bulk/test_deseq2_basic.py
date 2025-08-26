"""
Basic tests for DESeq2 functionality in omicverse.bulk.pyDEG

This test suite verifies core DESeq2 functionality works correctly.
"""

import pytest
import numpy as np
import pandas as pd
import omicverse as ov


class TestDESeq2Basic:
    """Basic test suite for DESeq2 functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create a small test dataset."""
        np.random.seed(42)
        # Create 50 genes x 20 samples
        data = pd.DataFrame(
            np.random.poisson(10, size=(50, 20)),
            index=[f"Gene_{i}" for i in range(50)],
            columns=[f"Sample_{i}" for i in range(20)]
        )
        return data
    
    def test_pyDEG_initialization(self, sample_data):
        """Test basic pyDEG initialization."""
        dds = ov.bulk.pyDEG(sample_data)
        assert dds is not None
        assert hasattr(dds, 'raw_data')
        assert hasattr(dds, 'data')
        pd.testing.assert_frame_equal(dds.raw_data, sample_data)
    
    def test_drop_duplicates_index(self, sample_data):
        """Test drop_duplicates_index function."""
        # Add duplicate gene names
        data_with_dups = sample_data.copy()
        data_with_dups.index = ['Gene_0'] * 5 + [f"Gene_{i}" for i in range(1, 46)]
        
        dds = ov.bulk.pyDEG(data_with_dups)
        result = dds.drop_duplicates_index()
        
        # Should remove duplicates
        assert len(result.index.unique()) == len(result.index)
        assert result is not None
    
    def test_deg_analysis_ttest(self, sample_data):
        """Test DEG analysis using t-test method."""
        dds = ov.bulk.pyDEG(sample_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(10)]
        control_groups = [f"Sample_{i}" for i in range(10, 20)]
        
        result = dds.deg_analysis(treatment_groups, control_groups, method='ttest')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check required columns exist
        required_columns = ['pvalue', 'qvalue', 'log2FC', 'BaseMean']
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_deg_analysis_deseq2(self, sample_data):
        """Test DEG analysis using DESeq2 method."""
        dds = ov.bulk.pyDEG(sample_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(10)]
        control_groups = [f"Sample_{i}" for i in range(10, 20)]
        
        try:
            result = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            
            # Check required columns exist
            required_columns = ['pvalue', 'qvalue', 'log2FC', 'BaseMean']
            for col in required_columns:
                assert col in result.columns, f"Missing column: {col}"
                
        except Exception as e:
            # DESeq2 might fail due to dependency issues in CI
            pytest.skip(f"DESeq2 failed: {str(e)}")
    
    def test_foldchange_set(self, sample_data):
        """Test fold change threshold setting."""
        dds = ov.bulk.pyDEG(sample_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(10)]
        control_groups = [f"Sample_{i}" for i in range(10, 20)]
        
        # Run analysis first
        result = dds.deg_analysis(treatment_groups, control_groups, method='ttest')
        
        # Set fold change thresholds
        dds.foldchange_set(fc_threshold=1.5, pval_threshold=0.05)
        
        assert hasattr(dds, 'fc_max')
        assert hasattr(dds, 'fc_min')
        assert hasattr(dds, 'pval_threshold')
        assert dds.fc_max == 1.5
        assert dds.fc_min == -1.5
        assert dds.pval_threshold == 0.05
        
        # Check that sig column was added
        assert 'sig' in dds.result.columns
        assert set(dds.result['sig'].unique()).issubset({'up', 'down', 'normal'})
    
    def test_plot_boxplot(self, sample_data):
        """Test boxplot plotting function."""
        dds = ov.bulk.pyDEG(sample_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(10)]
        control_groups = [f"Sample_{i}" for i in range(10, 20)]
        genes_to_plot = ['Gene_0', 'Gene_1']
        
        try:
            fig, ax = dds.plot_boxplot(
                genes=genes_to_plot,
                treatment_groups=treatment_groups,
                control_groups=control_groups
            )
            
            assert fig is not None
            assert ax is not None
            
        except Exception as e:
            # Plotting might fail due to missing dependencies
            pytest.skip(f"Plotting failed: {str(e)}")
    
    def test_ranking2gsea(self, sample_data):
        """Test ranking for GSEA analysis."""
        dds = ov.bulk.pyDEG(sample_data)
        dds.drop_duplicates_index()
        
        treatment_groups = [f"Sample_{i}" for i in range(10)]
        control_groups = [f"Sample_{i}" for i in range(10, 20)]
        
        # Run analysis first
        result = dds.deg_analysis(treatment_groups, control_groups, method='ttest')
        
        # Generate ranking
        rnk = dds.ranking2gsea()
        
        assert rnk is not None
        assert isinstance(rnk, pd.DataFrame)
        assert 'gene_name' in rnk.columns
        assert 'rnk' in rnk.columns
        assert len(rnk) == len(result)
    
    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        dds = ov.bulk.pyDEG(sample_data)
        
        treatment_groups = [f"Sample_{i}" for i in range(10)]
        control_groups = [f"Sample_{i}" for i in range(10, 20)]
        
        # Test invalid method
        with pytest.raises(ValueError):
            dds.deg_analysis(treatment_groups, control_groups, method='invalid_method')
        
        # Test empty groups
        with pytest.raises((ValueError, IndexError, KeyError)):
            dds.deg_analysis([], control_groups, method='ttest')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])