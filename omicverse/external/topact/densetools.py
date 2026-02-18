"""Tools for manipulating dense matrices."""
import itertools

import numpy as np
import numpy.typing as npt
#import utils_find_1st as utf1st
from skimage import morphology


def _get_pad(shift: int) -> tuple[int, int]:
    return (0, -shift) if shift < 0 else (shift, 0)


def _get_slice(shift: int) -> slice:
    if shift < 0:
        return slice(-shift, None)
    if shift > 0:
        return slice(0, -shift)
    return slice(None)


def first_nonzero_1d(vector: npt.NDArray) -> int:
    """Returns the inedx of the first nonzero element of a 1D array.

    Args:
        vector: a 1-dimensional numpy array

    Returns:
        The smallest i >= 0 such that vector[i] != 0. If no such i exists,
        then -1.
    """
    try:
        import utils_find_1st as utf1st
        return utf1st.find_1st(vector, 0, utf1st.cmp_not_equal)
    except Exception:
        nz = np.flatnonzero(vector)
        return int(nz[0]) if nz.size > 0 else -1


def first_nonzero_2d(array: npt.NDArray, axis: int) -> npt.NDArray:
    return np.apply_along_axis(first_nonzero_1d, axis, array)


def last_nonzero_2d(array: npt.NDArray, axis: int) -> npt.NDArray:
    flipped = np.flip(array, axis=axis)
    return array.shape[axis] - first_nonzero_2d(flipped, axis) - 1


def translate(matrix: npt.NDArray,
              x_shift: int,
              y_shift: int
              ) -> npt.NDArray:
    """Translates the matrix by the given shift, filling in with 0s.

    If matrix' is the output, it has the same shape (M,N) as the input
    matrix with entires matrix'[i,j] = matrix[i-y, j-x] whenever
    0 <= i-y < M and 0 <= j-x < N, and 0 otherwise.

    Args:
        matrix: A 2D numpy array
        x_shift: The signed number of places the matrix is shifted horizontally
        y_shift: The signed number of places the matrix is shifted vertically

    Returns:
        A matrix of the same shape as the input matrix, but with entries
        shifted by the given values in each direction. Missing values are set
        to 0.
    """

    height, width = matrix.shape
    if abs(x_shift) >= width or abs(y_shift) >= height:
        return np.zeros_like(matrix)

    x_pad = _get_pad(x_shift)
    y_pad = _get_pad(y_shift)
    padded = np.pad(matrix, (y_pad, x_pad))  # pyright: ignore

    return padded[_get_slice(y_shift)][:, _get_slice(x_shift)]


def pool(matrix: npt.NDArray, radius: int) -> npt.NDArray:
    """Replaces each entry by the sum of all neighbouring entries."""
    radius = min(max(matrix.shape) - 1, radius)
    pooled = np.zeros_like(matrix)
    iterable = range(-radius, radius+1)
    for i, j in itertools.product(iterable, iterable):
        pooled += translate(matrix, i, j)
    return pooled


def density(matrix: npt.NDArray, radius: int) -> npt.NDArray:
    """Computes the density at each entry in the array at the given radius."""
    ones = np.zeros_like(matrix) + 1
    num_entries = pool(ones, radius)
    return pool(matrix, radius) / num_entries


def density_hull(matrix: npt.NDArray,
                 radius: int,
                 threshold: float
                 ) -> npt.NDArray:
    density_matrix = density(matrix, radius)
    markers = density_matrix >= threshold
    hull = morphology.convex_hull_image(markers)
    return hull
