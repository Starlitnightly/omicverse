"""
Functions for preprocessing of single cell RNA-seq counts
"""
import numpy as np
import anndata
from scipy.sparse import issparse


def filter_counts_data(data, cell_min_molecules=1000, genes_min_cells=10):
    """Remove low molecule count cells and low detection genes

    :param data: Counts matrix: Cells x Genes
    :param cell_min_molecules: Minimum number of molecules per cell
    :param genes_min_cells: Minimum number of cells in which a gene is detected
    :return: Filtered counts matrix
    """

    # Molecule and cell counts
    ms = data.sum(axis=1)
    cs = data.sum()

    # Filter
    return data.loc[ms.index[ms > cell_min_molecules], cs.index[cs > genes_min_cells]]


def normalize_counts(data):
    """Correct the counts for molecule count variability

    :param data: Counts matrix: Cells x Genes
    :return: Normalized matrix
    """
    ms = data.sum(axis=1)
    norm_df = data.div(ms, axis=0).mul(np.median(ms), axis=0)
    return norm_df


def log_transform(data, pseudo_count=0.1):
    """Log transform the matrix

    :param data: Counts matrix: Cells x Genes or Anndata object
    :return: Log transformed matrix
    """
    if isinstance(data, anndata.AnnData):
        if issparse(data.X):
            data.X.data = np.log2(data.X.data + pseudo_count) - np.log2(pseudo_count)
        else:
            data.X = np.log2(data.X + pseudo_count) - np.log2(pseudo_count)
    else:
        return np.log2(data + pseudo_count)
