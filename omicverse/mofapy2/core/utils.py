from __future__ import division
from time import sleep

import numpy as np
import pandas as pd
import numpy.ma as ma
import os
import h5py

from sys import platform
from sklearn.utils.validation import check_array
from scipy import linalg

def _impose_f_order(X):
    """Helper Function"""
    # important to access flags instead of calling np.isfortran,
    # this catches corner cases.
    if X.flags.c_contiguous:
        return check_array(X.T, copy=False, order='F'), True
    else:
        return check_array(X, copy=False, order='F'), False


def fast_dot(A, B):
    """Compute fast dot products directly calling BLAS.

    This function calls BLAS directly while warranting Fortran contiguity.
    This helps avoiding extra copies `np.dot` would have created.
    For details see section `Linear Algebra on large Arrays`:
    http://wiki.scipy.org/PerformanceTips

    Parameters
    ----------
    A, B: instance of np.ndarray
        input matrices.
    """
    if A.dtype != B.dtype:
        raise ValueError('A and B must be of the same type.')
    if A.dtype not in (np.float32, np.float64):
        raise ValueError('Data must be single or double precision float.')

    dot = linalg.get_blas_funcs('gemm', (A, B))
    A, trans_a = _impose_f_order(A)
    B, trans_b = _impose_f_order(B)
    return dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)

def dotd(A, B, out=None):
    """Diagonal of :math:`\mathrm A\mathrm B^\intercal`.
    If ``A`` is :math:`n\times p` and ``B`` is :math:`p\times n`, it is done in :math:`O(pn)`.
    Args:
        A (array_like): Left matrix.
        B (array_like): Right matrix.
        out (:class:`numpy.ndarray`, optional): copy result to.
    Returns:
        :class:`numpy.ndarray`: Resulting diagonal.
    """
    A = ma.asarray(A, float)
    B = ma.asarray(B, float)
    if A.ndim == 1 and B.ndim == 1:
        if out is None:
            return ma.dot(A, B)
        return ma.dot(A, B, out)

    if out is None:
        out = ma.empty((A.shape[0], ), float)

    out[:] = ma.sum(A * B.T, axis=1)
    return out

def logdet(X):
    return np.log(np.linalg.det(X))
    # UC = np.linalg.cholesky(X)
    # return 2*sum(np.log(np.diag(UC)))

def sigmoid(X):
    """ Method to compute sigmoid function """
    return np.divide(1.,1.+np.exp(-X))
    # return 1./(1.+np.exp(-X))

def ddot(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array
      left: is the diagonal matrix on the left or on the right of the product?

    Output:
      ddot(d, mts, left=True) == dot(diag(d), mtx)
      ddot(d, mts, left=False) == dot(mtx, diag(d))
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

def lambdafn(X):
    return np.tanh(X/2.)/(4.*X)

def nans(shape, dtype=float):
    """ Method to create an array filled with missing values """
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def corr(A,B):
    """ Method to efficiently compute correlation coefficients between two matrices

    PARAMETERS
    ---------
    A: np array
    B: np array
    """

    # Rowwise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


def infer_platform():
    if platform == "linux" or platform == "linux2":
        return 1e3
    elif platform == "darwin":
        return 1e6
    else:
        print("Platform not recognised")
        exit()
