"""Simplified preprocessing utilities for testing."""

import scanpy as sc

__all__ = ["qc", "preprocess", "scale"]


def qc(adata, tresh=None):
    """Placeholder quality-control function."""
    return adata


def preprocess(adata, mode="scanpy", target_sum=1e4, n_HVGs=2000):
    """Minimal preprocessing: normalization and HVG selection."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_HVGs, subset=False)
    return adata


def scale(adata, max_value=10, layers_add="scaled"):
    """Scale data and store result in ``layers_add``."""
    sc.pp.scale(adata, max_value=max_value)
    adata.layers[layers_add] = adata.X.copy()
    return adata
