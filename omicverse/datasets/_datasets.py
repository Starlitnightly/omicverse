import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from anndata import AnnData
import scanpy as sc
import urllib.request
import warnings


def get_cache_dir() -> Path:
    """Get the cache directory for omicverse datasets."""
    cache_dir = Path.home() / '.cache' / 'omicverse' / 'datasets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_and_cache(
    url: str, 
    filename: str, 
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False
) -> Path:
    """
    Download a file and cache it locally.
    
    Arguments:
        url: URL to download from.
        filename: Local filename to save as.
        cache_dir: Directory to cache files. If None, uses default cache.
        force_download: Whether to force re-download if file exists.
    
    Returns:
        Path to the cached file.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = cache_dir / filename
    
    if not file_path.exists() or force_download:
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded to {file_path}")
        except Exception as e:
            if file_path.exists():
                file_path.unlink()  # Remove partial download
            raise Exception(f"Failed to download {url}: {e}")
    else:
        print(f"Using cached file: {file_path}")
    
    return file_path


def load_scanpy_pbmc3k(processed: bool = True) -> AnnData:
    """
    Load the PBMC 3k dataset from scanpy.
    
    This dataset contains 2700 single cells from peripheral blood mononuclear cells (PBMCs)
    sequenced with 10x Genomics. It's commonly used for clustering tutorials.
    
    Arguments:
        processed: Whether to return the preprocessed version with clustering.
    
    Returns:
        AnnData object with PBMC 3k data.
        
    Examples:
        >>> import omicverse as ov
        >>> 
        >>> # Load processed data with clustering
        >>> adata = ov.datasets.load_scanpy_pbmc3k(processed=True)
        >>> print(adata.obs.columns)
        >>> 
        >>> # Test statistical functions
        >>> adata.obs['condition'] = np.random.choice(['A', 'B'], adata.n_obs)
        >>> diversity_results = ov.utils.shannon_diversity(
        ...     adata, 'condition', 'louvain'
        ... )
    """
    
    try:
        if processed:
            adata = sc.datasets.pbmc3k_processed()
            print("Loaded PBMC 3k processed dataset (2700 cells)")
        else:
            adata = sc.datasets.pbmc3k()
            print("Loaded PBMC 3k raw dataset (2700 cells)")
            
        # Add some mock metadata for testing statistical functions
        np.random.seed(42)  # For reproducibility
        n_samples = 3
        sample_labels = np.random.choice([f'Sample_{i+1}' for i in range(n_samples)], adata.n_obs)
        adata.obs['sample_id'] = sample_labels
        
        # Add mock conditions
        conditions = np.random.choice(['Control', 'Treatment'], adata.n_obs)
        adata.obs['condition'] = conditions
        
        # Add mock tissue types for more complex analyses
        tissues = np.random.choice(['Blood', 'Normal', 'Tumor'], adata.n_obs, p=[0.4, 0.3, 0.3])
        adata.obs['tissue'] = tissues
        
        return adata
        
    except Exception as e:
        warnings.warn(f"Failed to load PBMC 3k dataset from scanpy: {e}")
        # Return mock dataset as fallback
        return create_mock_dataset(n_cells=2700, n_genes=2000, n_cell_types=8)


def load_scanpy_pbmc68k(filtered: bool = True) -> AnnData:
    """
    Load the PBMC 68k dataset from scanpy.
    
    Arguments:
        filtered: Whether to return the filtered version.
    
    Returns:
        AnnData object with PBMC 68k data.
    """
    
    try:
        adata = sc.datasets.pbmc68k_reduced()
        print("Loaded PBMC 68k reduced dataset")
        
        # Add mock metadata
        np.random.seed(42)
        n_samples = 5
        sample_labels = np.random.choice([f'Donor_{i+1}' for i in range(n_samples)], adata.n_obs)
        adata.obs['donor_id'] = sample_labels
        
        conditions = np.random.choice(['Healthy', 'Diseased'], adata.n_obs, p=[0.6, 0.4])
        adata.obs['condition'] = conditions
        
        return adata
        
    except Exception as e:
        warnings.warn(f"Failed to load PBMC 68k dataset from scanpy: {e}")
        return create_mock_dataset(n_cells=10000, n_genes=3000, n_cell_types=12)


def load_clustering_tutorial_data() -> AnnData:
    """
    Load data suitable for clustering tutorials, similar to scanpy's clustering tutorial.
    
    This function attempts to load the dataset used in scanpy's clustering tutorial.
    If unavailable, it creates a mock dataset with similar characteristics.
    
    Returns:
        AnnData object suitable for clustering analysis.
        
    Examples:
        >>> import omicverse as ov
        >>> import scanpy as sc
        >>> 
        >>> # Load tutorial data
        >>> adata = ov.datasets.load_clustering_tutorial_data()
        >>> 
        >>> # Preprocess for clustering
        >>> sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        >>> 
        >>> # Test resolution optimization
        >>> optimal_res = ov.utils.optimal_resolution(adata, metric='silhouette')
        >>> print(f"Optimal resolution: {optimal_res}")
    """
    
    try:
        # Try to load a standard dataset
        adata = load_scanpy_pbmc3k(processed=True)
        
        # Ensure it has the necessary preprocessing for clustering
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata, svd_solver='arpack')
            
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            
        print("Loaded clustering tutorial data based on PBMC 3k")
        return adata
        
    except Exception as e:
        warnings.warn(f"Failed to load tutorial data: {e}")
        return create_mock_dataset(n_cells=2000, n_genes=1500, n_cell_types=6, with_clustering=True)


def create_mock_dataset(
    n_cells: int = 2000,
    n_genes: int = 1500, 
    n_cell_types: int = 6,
    with_clustering: bool = False,
    random_state: int = 42
) -> AnnData:
    """
    Create a mock single-cell dataset for testing statistical functions.
    
    Arguments:
        n_cells: Number of cells to simulate.
        n_genes: Number of genes to simulate.
        n_cell_types: Number of cell types to simulate.
        with_clustering: Whether to include clustering preprocessing.
        random_state: Random seed for reproducibility.
    
    Returns:
        AnnData object with mock single-cell data.
        
    Examples:
        >>> import omicverse as ov
        >>> 
        >>> # Create mock data
        >>> adata = ov.datasets.create_mock_dataset(n_cells=1000, n_cell_types=5)
        >>> 
        >>> # Test statistical functions
        >>> or_results = ov.utils.odds_ratio(adata, 'condition', 'cell_type')
        >>> diversity_results = ov.utils.shannon_diversity(adata, 'condition', 'cell_type')
    """
    
    np.random.seed(random_state)
    
    # Generate mock expression data
    # Create some structure with different expression patterns per cell type
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    
    # Add some structure: different cell types have different expression patterns
    cell_type_labels = np.random.choice(range(n_cell_types), n_cells)
    
    for ct in range(n_cell_types):
        ct_mask = cell_type_labels == ct
        if np.sum(ct_mask) > 0:
            # Make some genes more highly expressed in this cell type
            high_genes = np.random.choice(n_genes, size=100, replace=False)
            X[ct_mask][:, high_genes] *= np.random.uniform(2, 5)
    
    # Create gene names
    gene_names = [f"Gene_{i+1:04d}" for i in range(n_genes)]
    
    # Create cell names  
    cell_names = [f"Cell_{i+1:04d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add mock metadata
    adata.obs['cell_type'] = [f'CellType_{i+1}' for i in cell_type_labels]
    
    # Add sample information
    n_samples = max(2, n_cell_types // 2)
    sample_labels = np.random.choice([f'Sample_{i+1}' for i in range(n_samples)], n_cells)
    adata.obs['sample_id'] = sample_labels
    
    # Add conditions
    conditions = np.random.choice(['Control', 'Treatment'], n_cells, p=[0.5, 0.5])
    adata.obs['condition'] = conditions
    
    # Add tissue types for odds ratio testing
    tissues = np.random.choice(['Blood', 'Normal', 'Tumor'], n_cells, p=[0.33, 0.33, 0.34])
    adata.obs['tissue'] = tissues
    
    # Add some basic gene information
    adata.var['gene_symbols'] = gene_names
    adata.var['highly_variable'] = np.random.choice([True, False], n_genes, p=[0.2, 0.8])
    
    # Add clustering preprocessing if requested
    if with_clustering:
        try:
            import scanpy as sc
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata.raw = adata
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            sc.tl.umap(adata)
            
            # Add some clustering results
            sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
            
        except ImportError:
            warnings.warn("Scanpy not available for preprocessing mock dataset")
    
    print(f"Created mock dataset: {n_cells} cells, {n_genes} genes, {n_cell_types} cell types")
    return adata


def list_available_datasets() -> List[str]:
    """
    List all available datasets that can be loaded.
    
    Returns:
        List of dataset names.
    """
    
    datasets = [
        'pbmc3k_processed',
        'pbmc3k_raw', 
        'pbmc68k_reduced',
        'clustering_tutorial',
        'mock_dataset'
    ]
    
    return datasets


# Convenience function for backward compatibility
def load_dataset(name: str, **kwargs) -> AnnData:
    """
    Load a dataset by name.
    
    Arguments:
        name: Name of the dataset to load.
        **kwargs: Additional arguments passed to the specific loader function.
    
    Returns:
        AnnData object with the requested dataset.
        
    Examples:
        >>> import omicverse as ov
        >>> 
        >>> # Load by name
        >>> adata = ov.datasets.load_dataset('pbmc3k_processed')
        >>> adata = ov.datasets.load_dataset('mock_dataset', n_cells=500)
    """
    
    if name == 'pbmc3k_processed':
        return load_scanpy_pbmc3k(processed=True)
    elif name == 'pbmc3k_raw':
        return load_scanpy_pbmc3k(processed=False)
    elif name == 'pbmc68k_reduced':
        return load_scanpy_pbmc68k()
    elif name == 'clustering_tutorial':
        return load_clustering_tutorial_data()
    elif name == 'mock_dataset':
        return create_mock_dataset(**kwargs)
    else:
        available = list_available_datasets()
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")