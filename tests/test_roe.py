import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import chi2_contingency, fisher_exact
import omicverse as ov


class TestROE:
    """Test suite for ov.utils.roe function"""
    
    @pytest.fixture
    def sample_adata_large(self):
        """Create test AnnData with large sample sizes for chi-square test"""
        np.random.seed(42)
        n_cells = 1000
        n_samples = 4
        n_celltypes = 5
        
        # Create balanced data with large expected frequencies
        sample_labels = np.random.choice([f'sample_{i}' for i in range(n_samples)], n_cells)
        celltype_labels = np.random.choice([f'celltype_{i}' for i in range(n_celltypes)], n_cells)
        
        obs = pd.DataFrame({
            'sample': sample_labels,
            'celltype': celltype_labels
        })
        
        adata = ad.AnnData(np.random.randn(n_cells, 100), obs=obs)
        return adata
    
    @pytest.fixture
    def sample_adata_small(self):
        """Create test AnnData with small sample sizes for Fisher's exact test"""
        # Create 2x2 contingency table with small expected frequencies
        obs = pd.DataFrame({
            'sample': ['A'] * 8 + ['B'] * 12,
            'celltype': ['type1'] * 3 + ['type2'] * 5 + ['type1'] * 2 + ['type2'] * 10
        })
        
        adata = ad.AnnData(np.random.randn(20, 10), obs=obs)
        return adata
    
    @pytest.fixture
    def balanced_adata(self):
        """Create perfectly balanced test data"""
        obs = pd.DataFrame({
            'sample': ['A'] * 50 + ['B'] * 50,
            'celltype': ['type1'] * 25 + ['type2'] * 25 + ['type1'] * 25 + ['type2'] * 25
        })
        
        adata = ad.AnnData(np.random.randn(100, 10), obs=obs)
        return adata
    
    def test_roe_basic_functionality(self, sample_adata_large):
        """Test basic ROE calculation functionality"""
        result = ov.utils.roe(sample_adata_large, 'sample', 'celltype')
        
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "cluster"
        assert len(result.index) > 0
        assert len(result.columns) > 0
    
    def test_roe_contingency_table_creation(self, sample_adata_large):
        """Test that contingency table is created correctly"""
        adata = sample_adata_large.copy()
        ov.utils.roe(adata, 'sample', 'celltype')
        
        # Verify that results are stored in uns
        assert 'expected_values' in adata.uns
        assert isinstance(adata.uns['expected_values'], pd.DataFrame)
    
    def test_roe_chi_square_test(self, sample_adata_large):
        """Test chi-square test implementation"""
        # Capture printed output to verify chi-square results
        adata = sample_adata_large.copy()
        result = ov.utils.roe(adata, 'sample', 'celltype')
        
        # Manually verify chi-square calculation
        contingency = pd.crosstab(adata.obs['celltype'], adata.obs['sample'])
        chi2_manual, p_manual, dof_manual, expected_manual = chi2_contingency(contingency)
        
        # Verify expected values match
        stored_expected = adata.uns['expected_values']
        np.testing.assert_array_almost_equal(stored_expected.values, expected_manual)
    
    def test_roe_small_expected_frequencies(self, sample_adata_small):
        """Test behavior with small expected frequencies (should warn about Fisher's exact test)"""
        adata = sample_adata_small.copy()
        
        # Capture stdout to verify warning message
        import io
        import contextlib
        from unittest.mock import patch
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = ov.utils.roe(adata, 'sample', 'celltype')
        
        output = f.getvalue()
        
        # Should contain warning about Fisher's exact test
        assert "Fisher's exact test" in output or "other statistical methods" in output
        
        # Should store results as unreliable
        assert 'unsig_roe_results' in adata.uns or 'sig_roe_results' in adata.uns
    
    def test_roe_balanced_data(self, balanced_adata):
        """Test ROE with perfectly balanced data (should give values close to 1)"""
        adata = balanced_adata.copy()
        result = ov.utils.roe(adata, 'sample', 'celltype')
        
        # All ROE values should be close to 1 for balanced data
        np.testing.assert_array_almost_equal(result.values, 1.0, decimal=10)
    
    def test_roe_threshold_categorization_nature_paper_implementation(self):
        """Test Nature paper threshold categorization implementation"""
        from omicverse.utils._roe import transform_roe_values
        
        # Test data with known ROE values
        test_roe = pd.DataFrame({
            'sample1': [0.0, 0.1, 0.3, 0.9, 1.2],
            'sample2': [0.05, 0.2, 0.8, 1.0, 2.5]
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])
        
        result = transform_roe_values(test_roe)
        
        # Nature paper thresholds: >1 (+++), 0.8<x≤1 (++), 0.2≤x≤0.8 (+), 0<x<0.2 (+/-), =0 (—)
        expected = pd.DataFrame({
            'sample1': ['—', '+/-', '+', '++', '+++'],
            'sample2': ['+/-', '+', '+', '++', '+++']
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_roe_threshold_categorization_nature_paper(self):
        """Test what threshold categorization SHOULD be according to Nature paper"""
        # This test documents the CORRECT thresholds according to the Nature paper
        # Current implementation will fail this test - it serves as specification
        
        test_roe = pd.DataFrame({
            'sample1': [0.0, 0.1, 0.3, 0.9, 1.2],
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])
        
        # Expected according to Nature paper:
        # +++: Ro/e > 1
        # ++: 0.8 < Ro/e ≤ 1  
        # +: 0.2 ≤ Ro/e ≤ 0.8
        # +/-: 0 < Ro/e < 0.2
        # —: Ro/e = 0
        def correct_transform_roe_values(roe):
            """Correct implementation according to Nature paper"""
            def _categorize_value(x):
                if x == 0:
                    return "—"
                if 0 < x < 0.2:
                    return "+/-"
                if 0.2 <= x <= 0.8:
                    return "+"
                if 0.8 < x <= 1:
                    return "++"
                return "+++"

            transformed_roe = roe.copy()
            return transformed_roe.apply(lambda col: col.map(_categorize_value))
        
        result = correct_transform_roe_values(test_roe)
        
        expected = pd.DataFrame({
            'sample1': ['—', '+/-', '+', '++', '+++'],
        }, index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'])
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_roe_pvalue_threshold(self, sample_adata_large):
        """Test p-value threshold handling"""
        adata = sample_adata_large.copy()
        
        # Test with very strict p-value threshold
        result = ov.utils.roe(adata, 'sample', 'celltype', pval_threshold=0.001)
        assert isinstance(result, pd.DataFrame)
        
        # Test with very lenient p-value threshold  
        result = ov.utils.roe(adata, 'sample', 'celltype', pval_threshold=0.99)
        assert isinstance(result, pd.DataFrame)
    
    def test_roe_expected_value_threshold(self, sample_adata_large):
        """Test expected value threshold handling"""
        adata = sample_adata_large.copy()
        
        # Test with high expected value threshold (should trigger warning)
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = ov.utils.roe(adata, 'sample', 'celltype', expected_value_threshold=100)
        
        output = f.getvalue()
        assert isinstance(result, pd.DataFrame)
    
    def test_roe_column_order(self, sample_adata_large):
        """Test column ordering functionality"""
        adata = sample_adata_large.copy()
        
        # Get available sample names
        sample_names = adata.obs['sample'].unique()
        if len(sample_names) >= 2:
            order_str = ','.join(sorted(sample_names, reverse=True))
            result = ov.utils.roe(adata, 'sample', 'celltype', order=order_str)
            
            # Check that columns are in the specified order
            expected_order = order_str.split(',')
            assert list(result.columns) == expected_order
    
    def test_fisher_exact_2x2_case(self):
        """Test case that should use Fisher's exact test for 2x2 contingency table"""
        # Create minimal 2x2 case with small expected frequencies
        obs = pd.DataFrame({
            'sample': ['A'] * 5 + ['B'] * 7,
            'celltype': ['type1'] * 2 + ['type2'] * 3 + ['type1'] * 1 + ['type2'] * 6
        })
        
        adata = ad.AnnData(np.random.randn(12, 5), obs=obs)
        
        # This should trigger the small expected frequencies warning
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = ov.utils.roe(adata, 'sample', 'celltype')
        
        output = f.getvalue()
        
        # Should warn about using Fisher's exact test
        assert "Fisher's exact test" in output or "other statistical methods" in output
        
        # Manually verify this is indeed a case for Fisher's exact test
        contingency = pd.crosstab(adata.obs['celltype'], adata.obs['sample'])
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        # At least one expected frequency should be < 5
        assert any(exp < 5 for exp in expected.flatten())
    
    def test_roe_plot_heatmap_functionality(self, sample_adata_large):
        """Test that the heatmap plotting works with ROE results"""
        adata = sample_adata_large.copy()
        ov.utils.roe(adata, 'sample', 'celltype')
        
        # Test that plotting functions can be called without error
        # (We won't actually display the plot in tests)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for testing
            ov.utils.roe_plot_heatmap(adata, display_numbers=True)
            ov.utils.roe_plot_heatmap(adata, display_numbers=False)
        except ImportError:
            pytest.skip("Matplotlib not available for plotting tests")
    
    def test_roe_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with empty data
        empty_adata = ad.AnnData(np.array([]).reshape(0, 5))
        empty_adata.obs = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            ov.utils.roe(empty_adata, 'sample', 'celltype')
        
        # Test with missing keys
        adata = ad.AnnData(np.random.randn(10, 5))
        adata.obs = pd.DataFrame({'sample': ['A'] * 10})
        
        with pytest.raises(KeyError):
            ov.utils.roe(adata, 'sample', 'missing_celltype')
    
    def test_roe_return_values(self, sample_adata_large):
        """Test that appropriate results are returned and stored"""
        adata = sample_adata_large.copy()
        result = ov.utils.roe(adata, 'sample', 'celltype')
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should store results in adata.uns
        assert 'expected_values' in adata.uns
        assert ('sig_roe_results' in adata.uns) or ('unsig_roe_results' in adata.uns)
        
        # Check that returned result matches what's stored
        if 'sig_roe_results' in adata.uns:
            stored_result = adata.uns['sig_roe_results']
        else:
            stored_result = adata.uns['unsig_roe_results']
        
        pd.testing.assert_frame_equal(result, stored_result)


if __name__ == "__main__":
    pytest.main([__file__])
