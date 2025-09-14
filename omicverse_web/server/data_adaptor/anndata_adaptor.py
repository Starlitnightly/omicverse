"""
High-performance AnnData adaptor with FlatBuffers serialization
Based on CellxGene's data handling architecture
"""

import warnings
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
from typing import Optional, Dict, List, Any, Tuple

from ..common.fbs.matrix import encode_matrix_fbs, decode_matrix_fbs

# Suppress warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HighPerformanceAnndataAdaptor:
    """
    High-performance adaptor for AnnData objects with optimized data access
    and FlatBuffers serialization for fast client-server communication
    """

    def __init__(self, data_path: str, chunk_size: int = 50000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.adata = None
        self.n_obs = 0
        self.n_vars = 0

        # Cached data for performance
        self._obs_cache = {}
        self._var_cache = {}
        self._expression_cache = {}
        self._embedding_cache = {}

        # Load data
        self._load_data()
        self._build_indexes()

    def _load_data(self):
        """Load AnnData with optimizations"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.adata = sc.read_h5ad(self.data_path)
            self.n_obs = self.adata.n_obs
            self.n_vars = self.adata.n_vars
            logger.info(f"Loaded data: {self.n_obs} cells, {self.n_vars} genes")

            # Ensure data is properly formatted
            self._optimize_data_types()

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _optimize_data_types(self):
        """Optimize data types for memory efficiency and performance"""
        # Convert categorical columns to save memory
        for col in self.adata.obs.columns:
            if self.adata.obs[col].dtype == 'object':
                try:
                    self.adata.obs[col] = self.adata.obs[col].astype('category')
                except:
                    pass

        # Ensure sparse matrices are in CSR format for fast row access
        if sparse.issparse(self.adata.X):
            if not sparse.isspmatrix_csr(self.adata.X):
                self.adata.X = self.adata.X.tocsr()

    def _build_indexes(self):
        """Build indexes for fast data access"""
        logger.info("Building data indexes...")

        # Pre-compute embeddings info
        self.available_embeddings = list(self.adata.obsm.keys())

        # Pre-compute obs/var column info
        self.obs_columns = {
            'categorical': [],
            'continuous': []
        }

        for col in self.adata.obs.columns:
            if pd.api.types.is_categorical_dtype(self.adata.obs[col]) or self.adata.obs[col].dtype == 'object':
                self.obs_columns['categorical'].append(col)
            elif pd.api.types.is_numeric_dtype(self.adata.obs[col]):
                self.obs_columns['continuous'].append(col)

        logger.info("Index building complete")

    def get_schema(self) -> Dict[str, Any]:
        """Get data schema for client"""
        return {
            'dataframe': {
                'nObs': self.n_obs,
                'nVars': self.n_vars,
                'type': 'float32'
            },
            'annotations': {
                'obs': {
                    'index': self.adata.obs.index.name or 'cell_id',
                    'columns': [
                        {
                            'name': col,
                            'type': 'categorical' if col in self.obs_columns['categorical'] else 'continuous',
                            'categories': list(self.adata.obs[col].cat.categories) if col in self.obs_columns['categorical'] else None
                        }
                        for col in self.adata.obs.columns
                    ]
                },
                'var': {
                    'index': self.adata.var.index.name or 'gene_id',
                    'columns': [
                        {
                            'name': col,
                            'type': 'categorical' if pd.api.types.is_categorical_dtype(self.adata.var[col]) else 'continuous'
                        }
                        for col in self.adata.var.columns
                    ]
                }
            },
            'layout': {
                'obs': [
                    {
                        'name': emb.replace('X_', ''),
                        'type': 'float32',
                        'dims': ['x', 'y'] if self.adata.obsm[emb].shape[1] >= 2 else ['x']
                    }
                    for emb in self.available_embeddings
                    if self.adata.obsm[emb].shape[1] >= 2
                ]
            }
        }

    def get_obs_fbs(self, columns: Optional[List[str]] = None) -> bytes:
        """Get observation annotations as FlatBuffers"""
        if columns is None:
            df = self.adata.obs
        else:
            df = self.adata.obs[columns]

        return encode_matrix_fbs(df, col_idx=df.columns)

    def get_var_fbs(self, columns: Optional[List[str]] = None) -> bytes:
        """Get variable annotations as FlatBuffers"""
        if columns is None:
            df = self.adata.var
        else:
            df = self.adata.var[columns]

        return encode_matrix_fbs(df, col_idx=df.columns)

    def get_embedding_fbs(self, embedding_name: str) -> bytes:
        """Get embedding coordinates as FlatBuffers"""
        # Support synthetic random embedding when no embedding available or explicitly requested
        if embedding_name == 'random' or embedding_name == 'X_random':
            coords = self._get_random_embedding()
        else:
            embedding_key = f'X_{embedding_name}' if not embedding_name.startswith('X_') else embedding_name
            if embedding_key not in self.adata.obsm:
                raise KeyError(f"Embedding '{embedding_name}' not found")
            coords = self.adata.obsm[embedding_key]

        # Convert to DataFrame for consistent serialization
        df = pd.DataFrame(
            coords[:, :2],  # Take first 2 dimensions
            columns=['x', 'y'],
            index=self.adata.obs.index
        )

        return encode_matrix_fbs(df, col_idx=df.columns)

    def get_expression_fbs(self, gene_names: List[str], cell_indices: Optional[List[int]] = None) -> bytes:
        """Get gene expression data as FlatBuffers"""
        # Find gene indices
        gene_indices = []
        valid_genes = []

        for gene in gene_names:
            if gene in self.adata.var_names:
                gene_indices.append(self.adata.var_names.get_loc(gene))
                valid_genes.append(gene)
            else:
                logger.warning(f"Gene '{gene}' not found")

        if not gene_indices:
            raise ValueError("No valid genes found")

        # Get cell indices
        if cell_indices is None:
            cell_indices = list(range(self.n_obs))

        # Extract expression data
        if sparse.issparse(self.adata.X):
            expression_data = self.adata.X[np.ix_(cell_indices, gene_indices)].toarray()
        else:
            expression_data = self.adata.X[np.ix_(cell_indices, gene_indices)]

        # Convert to DataFrame
        df = pd.DataFrame(
            expression_data,
            columns=valid_genes,
            index=self.adata.obs.index[cell_indices]
        )

        return encode_matrix_fbs(df, col_idx=df.columns)

    def filter_cells(self, filters: Dict[str, Any]) -> List[int]:
        """Filter cells based on criteria and return indices"""
        indices = np.arange(self.n_obs)

        for column, criteria in filters.items():
            if column not in self.adata.obs.columns:
                continue

            if 'values' in criteria:
                # Categorical filter
                mask = self.adata.obs[column].isin(criteria['values'])
                indices = indices[mask[indices]]
            elif 'min' in criteria or 'max' in criteria:
                # Range filter
                values = self.adata.obs[column].values[indices]
                if 'min' in criteria:
                    indices = indices[values >= criteria['min']]
                if 'max' in criteria:
                    values = self.adata.obs[column].values[indices]
                    indices = indices[values <= criteria['max']]

        return indices.tolist()

    def get_differential_expression(self, group1_indices: List[int], group2_indices: List[int],
                                    method: str = 'wilcoxon', n_genes: int = 100) -> Dict[str, Any]:
        """Compute differential expression between two groups"""
        try:
            # Create temporary grouping
            groups = np.full(self.n_obs, 'other', dtype='<U10')
            groups[group1_indices] = 'group1'
            groups[group2_indices] = 'group2'

            # Add to obs temporarily
            self.adata.obs['temp_groups'] = pd.Categorical(groups)

            # Run differential expression
            if method == 'wilcoxon':
                sc.tl.rank_genes_groups(self.adata, 'temp_groups', groups=['group1'], reference='group2', method='wilcoxon')
            else:
                sc.tl.rank_genes_groups(self.adata, 'temp_groups', groups=['group1'], reference='group2', method='t-test')

            # Extract results
            result = self.adata.uns['rank_genes_groups']

            differential_genes = {
                'names': result['names']['group1'][:n_genes].tolist(),
                'scores': result['scores']['group1'][:n_genes].tolist(),
                'pvals': result['pvals']['group1'][:n_genes].tolist(),
                'pvals_adj': result['pvals_adj']['group1'][:n_genes].tolist(),
                'logfoldchanges': result['logfoldchanges']['group1'][:n_genes].tolist()
            }

            # Clean up
            del self.adata.obs['temp_groups']

            return differential_genes

        except Exception as e:
            logger.error(f"Differential expression failed: {e}")
            return {'error': str(e)}

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset"""
        return {
            'n_obs': self.n_obs,
            'n_vars': self.n_vars,
            'embeddings': [emb.replace('X_', '') for emb in self.available_embeddings],
            'obs_columns': self.obs_columns,
            'memory_usage': {
                'X_memory_mb': self.adata.X.data.nbytes / 1024 / 1024 if sparse.issparse(self.adata.X) else self.adata.X.nbytes / 1024 / 1024,
                'obs_memory_mb': self.adata.obs.memory_usage(deep=True).sum() / 1024 / 1024,
                'var_memory_mb': self.adata.var.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }

    def get_chunked_data(self, data_type: str, chunk_index: int, **kwargs) -> bytes:
        """Get data in chunks for large datasets"""
        chunk_start = chunk_index * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.n_obs)

        if data_type == 'embedding':
            embedding_name = kwargs.get('embedding_name')
            if embedding_name == 'random' or embedding_name == 'X_random':
                coords_full = self._get_random_embedding()
                coords = coords_full[chunk_start:chunk_end]
            else:
                embedding_key = f'X_{embedding_name}' if not embedding_name.startswith('X_') else embedding_name
                coords = self.adata.obsm[embedding_key][chunk_start:chunk_end]

            df = pd.DataFrame(
                coords[:, :2],
                columns=['x', 'y'],
                index=self.adata.obs.index[chunk_start:chunk_end]
            )

            return encode_matrix_fbs(df, col_idx=df.columns)

        elif data_type == 'obs':
            columns = kwargs.get('columns', None)
            if columns:
                df = self.adata.obs[columns].iloc[chunk_start:chunk_end]
            else:
                df = self.adata.obs.iloc[chunk_start:chunk_end]

            return encode_matrix_fbs(df, col_idx=df.columns)

        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about data chunking"""
        total_chunks = (self.n_obs + self.chunk_size - 1) // self.chunk_size

        return {
            'total_chunks': total_chunks,
            'chunk_size': self.chunk_size,
            'total_cells': self.n_obs
        }

    def close(self):
        """Clean up resources"""
        self._obs_cache.clear()
        self._var_cache.clear()
        self._expression_cache.clear()
        self._embedding_cache.clear()
        logger.info("Data adaptor closed")

    def _get_random_embedding(self):
        """Generate or return cached random 2D embedding for fallback visualization."""
        if 'X_random' in self._embedding_cache:
            return self._embedding_cache['X_random']
        # Seed based on dataset path for determinism
        seed = abs(hash(self.data_path)) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        coords = rng.rand(self.n_obs, 2)
        self._embedding_cache['X_random'] = coords
        return coords
