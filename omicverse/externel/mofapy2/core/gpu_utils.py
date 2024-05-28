import sys
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass

# --------------------------------------------------------
# Defalut is to use numpy, not cupy
# --------------------------------------------------------
gpu_mode = False

# --------------------------------------------------------
# operations on single matrices
# --------------------------------------------------------
def log(mat):
    if gpu_mode:
        return cp.log(mat)
    else:
        return np.log(mat)

def square(mat):
    if gpu_mode:
        return cp.square(mat)
    else:
        return np.square(mat)

def exp(mat):
    if gpu_mode:
        return cp.exp(mat)
    else:
        return np.exp(mat)

def sum(mat):
    if gpu_mode:
        return cp.sum(mat)
    else:
        return np.sum(mat)

# --------------------------------------------------------
# operations on matrices pairs
# --------------------------------------------------------
def dot(mat1, mat2):
    if gpu_mode:
        return cp.dot(mat1, mat2)
    else:
        return np.dot(mat1, mat2)


def divide(mat1, mat2):
    if gpu_mode:
        return cp.divide(mat1, mat2)
    else:
        return np.divide(mat1, mat2)


# --------------------------------------------------------
# initialiser
# --------------------------------------------------------
def zeros(dim):
    if gpu_mode:
        return cp.zeros(dim)
    else:
        return np.zeros(dim)

# --------------------------------------------------------
# loading on and from the GPU
# --------------------------------------------------------
def array(mat):
    if gpu_mode:
        return cp.array(mat)
    else:
        return mat

def asnumpy(mat):
    if gpu_mode:
        return cp.asnumpy(mat)
    else:
        return mat
