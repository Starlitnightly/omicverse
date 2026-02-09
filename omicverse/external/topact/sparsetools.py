"""Tools for manipulating sparse matrices."""
import itertools
from typing import Any, Collection, Iterator, Sequence

import numpy as np
from scipy import sparse


def rescale(matrix, factors: Sequence[float], axis: int = 0):
    diagonal = sparse.csr_array(sparse.diags(factors))
    match axis:
        case 0:
            return diagonal @ matrix
        case 1:
            return matrix @ diagonal
        case _:
            raise ValueError(f"axis must be 0 or 1, not {axis}.")


def rescale_rows(matrix, factors: Sequence[float]):
    return rescale(matrix, factors, axis=0)


def rescale_columns(matrix, factors: Sequence[float]):
    return rescale(matrix, factors, axis=1)


def iterate_sparse(matrix) -> Iterator[tuple[int, int, Any]]:
    """Yields all entries in the matrix along with their indices.

    Args:
        matrix: A sparse array.

    Yields:
        The tuples (i, j, v = matrix[i,j]) for all non-zero entries v.

    Raises:
        NotImplementedError: If the matrix is not in csc or csr format.
    """
    height, width = matrix.shape
    match matrix.format:
        case "csc":
            ptr = matrix.indptr
            zipped = zip(matrix.indices, matrix.data)
            for j in range(width):
                to_pull = ptr[j+1] - ptr[j]
                for i, value in itertools.islice(zipped, to_pull):
                    yield i, j, value
        case "csr":
            ptr = matrix.indptr
            zipped = zip(matrix.indices, matrix.data)
            for i in range(height):
                to_pull = ptr[i+1] - ptr[i]
                for j, value in itertools.islice(zipped, to_pull):
                    yield i, j, value
        case _:
            raise NotImplementedError(f"Cannot iterate type {type(matrix)}")
    yield from ()


def filter_cols(matrix, keep_indices: Collection[int]):
    """Returns the input matrix with only columns from the given indices.

    Args:
        matrix: A sparse array.
        keep_indices: Column indices.

    Returns:
        A matrix with the same entries as the input, but with only the
        specified columns.
    """
    keep_indices = np.unique(np.array(keep_indices))
    keep_indices.sort()
    coo = matrix.tocoo()
    keep = np.isin(coo.col, keep_indices)
    coo.data, coo.row, coo.col = coo.data[keep], coo.row[keep], coo.col[keep]
    coo.col = np.searchsorted(keep_indices, coo.col)
    coo._shape = coo.shape[0], len(keep_indices)  # pylint: disable=W0212
    coo.getnnz()
    return coo.tocsr()
