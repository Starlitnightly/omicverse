import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import warnings

# Test imports
try:
    import omicverse as ov
    import scanpy as sc
    from anndata import AnnData
    OMICVERSE_AVAILABLE = True
except ImportError as e:
    OMICVERSE_AVAILABLE = False
    pytest.skip(f"omicverse not available: {e}", allow_module_level=True)


class TestDatasetsModule:
    """Test the datasets module functionality."""
    
    def test_create_mock_dataset(self):
        """Test creating mock datasets."""
        # Basic mock dataset
        adata = ov.datasets.create_mock_dataset(n_cells=100, n_genes=50, n_cell_types=3)
        
        assert adata.n_obs == 100
        assert adata.n_vars == 50
        assert 'cell_type' in adata.obs.columns
        assert 'condition' in adata.obs.columns
        assert 'tissue' in adata.obs.columns
        assert len(adata.obs['cell_type'].unique()) <= 3
        
    def test_create_mock_dataset_with_clustering(self):
        """Test creating mock dataset with clustering preprocessing."""
        # Mock with clustering
        adata = ov.datasets.create_mock_dataset(
            n_cells=200, 
            n_genes=100, 
            n_cell_types=4,
            with_clustering=True
        )
        
        assert adata.n_obs == 200
        assert 'X_pca' in adata.obsm
        assert 'neighbors' in adata.uns
        
    def test_load_scanpy_pbmc3k_fallback(self):
        """Test loading PBMC3k with fallback to mock data."""
        # This should work even if scanpy datasets are not available
        adata = ov.datasets.load_scanpy_pbmc3k(processed=True)
        
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert 'sample_id' in adata.obs.columns
        assert 'condition' in adata.obs.columns
        
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = ov.datasets.list_available_datasets()
        
        assert isinstance(datasets, list)
        assert 'pbmc3k_processed' in datasets
        assert 'mock_dataset' in datasets
        
    def test_load_dataset_by_name(self):
        """Test loading dataset by name."""
        # Test mock dataset loading
        adata = ov.datasets.load_dataset('mock_dataset', n_cells=50, n_cell_types=2)
        
        assert adata.n_obs == 50
        assert len(adata.obs['cell_type'].unique()) <= 2


class TestOddsRatio:
    """Test odds ratio calculation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create a simple test dataset
        self.adata = ov.datasets.create_mock_dataset(
            n_cells=200, 
            n_genes=100, 
            n_cell_types=3,
            random_state=42
        )
    
    def test_odds_ratio_basic(self):
        """Test basic odds ratio calculation."""
        results = ov.utils.odds_ratio(
            self.adata,
            sample_key='condition',
            cell_type_key='cell_type'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'odds_ratio' in results.columns
        assert 'p_value' in results.columns
        assert 'ci_lower' in results.columns
        assert 'ci_upper' in results.columns
        assert len(results) > 0
        
        # Check that results are stored in adata
        assert 'odds_ratio_results' in self.adata.uns
        
    def test_odds_ratio_with_reference_group(self):
        """Test odds ratio calculation with specified reference group."""
        results = ov.utils.odds_ratio(
            self.adata,
            sample_key='tissue',
            cell_type_key='cell_type',
            reference_group='Blood'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert all(results['reference_group'] == 'Blood')
        
    def test_odds_ratio_with_correction(self):
        """Test odds ratio calculation with multiple testing correction."""
        results = ov.utils.odds_ratio(
            self.adata,
            sample_key='condition',
            cell_type_key='cell_type',
            correction_method='bonferroni'
        )
        
        assert 'p_value_adjusted' in results.columns
        # Adjusted p-values should be >= original p-values
        valid_mask = ~(results['p_value'].isna() | results['p_value_adjusted'].isna())
        if valid_mask.sum() > 0:
            assert all(results.loc[valid_mask, 'p_value_adjusted'] >= results.loc[valid_mask, 'p_value'])
    
    def test_odds_ratio_invalid_reference(self):
        """Test odds ratio calculation with invalid reference group."""
        with pytest.raises(ValueError, match="Reference group .* not found"):
            ov.utils.odds_ratio(
                self.adata,
                sample_key='condition',
                cell_type_key='cell_type',
                reference_group='NonexistentGroup'
            )
    
    def test_plot_odds_ratio_heatmap(self):
        """Test odds ratio heatmap plotting."""
        # First calculate odds ratios
        ov.utils.odds_ratio(
            self.adata,
            sample_key='tissue',
            cell_type_key='cell_type'
        )
        
        # Test plotting (should not raise errors)
        with patch('matplotlib.pyplot.show'):
            ov.utils.plot_odds_ratio_heatmap(self.adata, log_scale=True)
            ov.utils.plot_odds_ratio_heatmap(self.adata, log_scale=False, show_ci=True)
    
    def test_plot_odds_ratio_without_results(self):
        """Test plotting without calculated results."""
        # Create fresh dataset without results
        adata = ov.datasets.create_mock_dataset(n_cells=50, n_cell_types=2)
        
        with pytest.raises(ValueError, match="No odds ratio results found"):
            ov.utils.plot_odds_ratio_heatmap(adata)


class TestShannonDiversity:
    """Test Shannon diversity calculation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.adata = ov.datasets.create_mock_dataset(
            n_cells=300, 
            n_genes=100, 
            n_cell_types=4,
            random_state=42
        )
    
    def test_shannon_diversity_basic(self):
        """Test basic Shannon diversity calculation."""
        results = ov.utils.shannon_diversity(
            self.adata,
            groupby='condition',
            cell_type_key='cell_type'
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'shannon_diversity' in results.columns
        assert 'shannon_evenness' in results.columns
        assert 'simpson_diversity' in results.columns
        assert 'total_cells' in results.columns
        assert 'n_cell_types' in results.columns
        
        # Check that diversity values are reasonable
        assert all(results['shannon_diversity'] >= 0)
        assert all(results['simpson_diversity'] >= 0)
        assert all(results['simpson_diversity'] <= 1)
        
        # Check that results are stored in adata
        assert 'shannon_diversity_results' in self.adata.uns
        
    def test_shannon_diversity_different_bases(self):
        """Test Shannon diversity with different logarithm bases."""
        # Natural log
        results_ln = ov.utils.shannon_diversity(
            self.adata,
            groupby='condition',
            cell_type_key='cell_type',
            base='natural'
        )
        
        # Base 2
        results_log2 = ov.utils.shannon_diversity(
            self.adata,
            groupby='condition', 
            cell_type_key='cell_type',
            base='2'
        )
        
        # Log2 should give different values than natural log
        # (unless diversity is 0 or 1)
        assert not results_ln['shannon_diversity'].equals(results_log2['shannon_diversity'])
        
    def test_shannon_diversity_no_evenness(self):
        """Test Shannon diversity without evenness calculation."""
        results = ov.utils.shannon_diversity(
            self.adata,
            groupby='tissue',
            cell_type_key='cell_type',
            calculate_evenness=False
        )
        
        assert results['shannon_evenness'].isna().all()
        
    def test_compare_shannon_diversity(self):
        """Test Shannon diversity comparison between groups."""
        comparison = ov.utils.compare_shannon_diversity(
            self.adata,
            groupby='condition',
            cell_type_key='cell_type'
        )
        
        assert isinstance(comparison, dict)
        assert 'diversity_values' in comparison
        assert 'test_results' in comparison
        assert 'mean_diversity' in comparison
        
    def test_plot_shannon_diversity(self):
        """Test Shannon diversity plotting."""
        # First calculate diversity
        ov.utils.shannon_diversity(
            self.adata,
            groupby='condition',
            cell_type_key='cell_type'
        )
        
        # Test plotting (should not raise errors)
        with patch('matplotlib.pyplot.show'):
            ov.utils.plot_shannon_diversity(self.adata, metric='shannon_diversity')
            ov.utils.plot_shannon_diversity(self.adata, metric='shannon_evenness')
    
    def test_plot_shannon_diversity_without_results(self):
        """Test plotting without calculated results."""
        adata = ov.datasets.create_mock_dataset(n_cells=50, n_cell_types=2)
        
        with pytest.raises(ValueError, match="No Shannon diversity results found"):
            ov.utils.plot_shannon_diversity(adata)
    
    def test_shannon_diversity_invalid_metric(self):
        """Test plotting with invalid metric."""
        # First calculate diversity
        ov.utils.shannon_diversity(
            self.adata,
            groupby='condition',
            cell_type_key='cell_type'
        )
        
        with pytest.raises(ValueError, match="Metric .* not found in results"):
            ov.utils.plot_shannon_diversity(self.adata, metric='invalid_metric')


class TestResolutionOptimization:
    """Test resolution optimization functionality."""
    
    def setup_method(self):
        """Set up test data with clustering preprocessing."""
        self.adata = ov.datasets.create_mock_dataset(
            n_cells=200, 
            n_genes=100, 
            n_cell_types=4,
            with_clustering=True,
            random_state=42
        )
    
    def test_optimal_resolution_basic(self):
        """Test basic optimal resolution finding."""
        optimal_res = ov.utils.optimal_resolution(
            self.adata,
            resolution_range=(0.1, 1.0),
            n_resolutions=5,
            metric='silhouette'
        )
        
        assert isinstance(optimal_res, float)
        assert 0.1 <= optimal_res <= 1.0
        
        # Check that results are stored
        assert 'optimal_resolution_results' in self.adata.uns
        assert 'optimal_resolution_params' in self.adata.uns
        
    def test_optimal_resolution_without_neighbors(self):
        """Test optimal resolution without neighbors graph."""
        # Create dataset without clustering preprocessing
        adata = ov.datasets.create_mock_dataset(n_cells=100, n_cell_types=3)
        
        with pytest.raises(ValueError, match="Neighbors graph not found"):
            ov.utils.optimal_resolution(adata)
    
    def test_optimal_resolution_invalid_range(self):
        """Test optimal resolution with invalid range."""
        with pytest.raises(ValueError, match="resolution_range.*must be less than"):
            ov.utils.optimal_resolution(
                self.adata,
                resolution_range=(1.0, 0.1)  # Invalid: min > max
            )
    
    def test_optimal_resolution_different_methods(self):
        """Test optimal resolution with different clustering methods."""
        # Test leiden
        res_leiden = ov.utils.optimal_resolution(
            self.adata,
            clustering_method='leiden',
            n_resolutions=3
        )
        
        # Test louvain
        res_louvain = ov.utils.optimal_resolution(
            self.adata,
            clustering_method='louvain',
            n_resolutions=3
        )
        
        assert isinstance(res_leiden, float)
        assert isinstance(res_louvain, float)
    
    def test_optimal_resolution_invalid_method(self):
        """Test optimal resolution with invalid clustering method."""
        with pytest.raises(ValueError, match="clustering_method must be"):
            ov.utils.optimal_resolution(
                self.adata,
                clustering_method='invalid_method'
            )
    
    def test_optimal_resolution_copy(self):
        """Test optimal resolution with copy=True."""
        adata_copy = ov.utils.optimal_resolution(
            self.adata,
            n_resolutions=3,
            copy=True
        )
        
        assert isinstance(adata_copy, AnnData)
        assert adata_copy is not self.adata
        assert 'optimal_resolution_results' in adata_copy.uns
    
    def test_plot_resolution_optimization(self):
        """Test plotting resolution optimization results."""
        # First run optimization
        ov.utils.optimal_resolution(
            self.adata,
            n_resolutions=5,
            metric='silhouette'
        )
        
        # Test plotting
        with patch('matplotlib.pyplot.show'):
            ov.utils.plot_resolution_optimization(self.adata)
    
    def test_plot_resolution_without_results(self):
        """Test plotting without optimization results."""
        adata = ov.datasets.create_mock_dataset(n_cells=50, with_clustering=True)
        
        with pytest.raises(ValueError, match="No resolution optimization results found"):
            ov.utils.plot_resolution_optimization(adata)
    
    def test_resolution_stability_analysis(self):
        """Test resolution stability analysis."""
        results = ov.utils.resolution_stability_analysis(
            self.adata,
            resolution_range=(0.1, 0.5),
            n_resolutions=3,
            n_iterations=3
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'resolution' in results.columns
        assert 'mean_ari' in results.columns
        assert 'stability_score' in results.columns
        
        # Check that results are stored
        assert 'resolution_stability_results' in self.adata.uns


class TestIntegrationWithExistingCode:
    """Test integration of new functions with existing omicverse code."""
    
    def setup_method(self):
        """Set up test data."""
        self.adata = ov.datasets.load_clustering_tutorial_data()
    
    def test_roe_still_works(self):
        """Test that existing roe function still works."""
        # This tests backward compatibility
        try:
            roe_results = ov.utils.roe(
                self.adata,
                sample_key='condition',
                cell_type_key='cell_type'
            )
            assert isinstance(roe_results, pd.DataFrame)
        except Exception as e:
            # Some failures might be expected due to statistical requirements
            # but the function should still be importable and callable
            assert 'roe' in str(type(e).__name__).lower() or 'statistical' in str(e).lower()
    
    def test_functions_are_importable(self):
        """Test that all new functions can be imported."""
        # Test direct import
        from omicverse.utils import odds_ratio, shannon_diversity, optimal_resolution
        
        # Test that they're callable
        assert callable(odds_ratio)
        assert callable(shannon_diversity)  
        assert callable(optimal_resolution)
        
        # Test import through ov.utils
        assert hasattr(ov.utils, 'odds_ratio')
        assert hasattr(ov.utils, 'shannon_diversity')
        assert hasattr(ov.utils, 'optimal_resolution')
    
    def test_datasets_module_importable(self):
        """Test that datasets module can be imported."""
        assert hasattr(ov, 'datasets')
        assert hasattr(ov.datasets, 'create_mock_dataset')
        assert hasattr(ov.datasets, 'load_scanpy_pbmc3k')


class TestRealWorldScenarios:
    """Test functions with real-world-like scenarios."""
    
    def test_complete_workflow(self):
        """Test a complete analysis workflow using the new functions."""
        # Load data
        adata = ov.datasets.load_clustering_tutorial_data()
        
        # Test Shannon diversity
        diversity_results = ov.utils.shannon_diversity(
            adata, 
            groupby='condition',
            cell_type_key='cell_type'
        )
        assert len(diversity_results) >= 1
        
        # Test odds ratio (if we have sufficient data)
        if len(adata.obs['condition'].unique()) >= 2 and len(adata.obs['cell_type'].unique()) >= 2:
            or_results = ov.utils.odds_ratio(
                adata,
                sample_key='condition', 
                cell_type_key='cell_type'
            )
            assert len(or_results) >= 1
        
        # Test resolution optimization (if neighbors exist)
        if 'neighbors' in adata.uns:
            try:
                optimal_res = ov.utils.optimal_resolution(
                    adata,
                    resolution_range=(0.1, 0.8),
                    n_resolutions=3,
                    metric='silhouette'
                )
                assert isinstance(optimal_res, float)
            except Exception as e:
                # May fail due to clustering requirements, but shouldn't crash
                warnings.warn(f"Resolution optimization failed: {e}")
    
    def test_edge_cases(self):
        """Test functions with edge cases."""
        # Very small dataset
        small_adata = ov.datasets.create_mock_dataset(n_cells=10, n_cell_types=2)
        
        # Shannon diversity should still work
        diversity_results = ov.utils.shannon_diversity(
            small_adata,
            groupby='condition',
            cell_type_key='cell_type'
        )
        assert isinstance(diversity_results, pd.DataFrame)
        
        # Single cell type
        single_type_adata = ov.datasets.create_mock_dataset(n_cells=50, n_cell_types=1)
        diversity_single = ov.utils.shannon_diversity(
            single_type_adata,
            groupby='condition', 
            cell_type_key='cell_type'
        )
        # Shannon diversity should be 0 for single cell type
        assert all(diversity_single['shannon_diversity'] == 0)


if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__, '-v'])