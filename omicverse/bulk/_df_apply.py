#import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool


def gen_row(ndarray):
    r"""Generate rows from numpy array.
    
    Arguments:
        ndarray: Numpy array to iterate through
        
    Yields:
        Row vectors from the input array
    """
    for i in range(ndarray.shape[0]):
        yield ndarray[i,:]


def gen_col(ndarray):
    r"""Generate columns from numpy array.
    
    Arguments:
        ndarray: Numpy array to iterate through
        
    Yields:
        Column vectors from the input array
    """
    for i in range(ndarray.shape[1]):
        yield ndarray[:,i]

def apply(func, df, axis=0, ncores=None, p=None, **kwargs):
    r"""Apply function to rows or columns of numpy array with multiprocessing support.
    
    Arguments:
        func: Function to apply to each row/column
        df: Numpy array to process
        axis: Apply function along rows (0) or columns (1) (default: 0)
        ncores: Number of CPU cores to use for parallel processing (default: None)
        p: Pre-existing multiprocessing Pool object (default: None)
        **kwargs: Additional keyword arguments passed to func
        
    Returns:
        results: Numpy array containing function results
    """
    # Check axis input is 0 or 1
    if axis not in (0,1):
        raise IndexError("axis must equal 0 or 1")
    
    #check if p is provided or needs to be created
    if p == None and ncores != None:
        p = Pool(ncores)
    
    #create function and pass kwargs
    g = partial(func, **kwargs)
    
    #if axis is 0 apply function to rows
    if axis == 0:
        #conduct multiprocessed version or not
        if p != None:
            iter_results = p.map(g, gen_row(df))
        else:
            iter_results = map(g, gen_row(df))
    #if axis is 1 apply function to columns
    elif axis == 1:
        #conduct multiprocessed version or not
        if p != None:
            iter_results = p.map(g, gen_col(df))
        else:
            iter_results = map(g, gen_col(df))
    
    #close Pool if it wasn't provided
    if ncores != None:
        p.close()
        p.join()
    
    #create DataFrame for output
    results = np.array(list(iter_results))
    #if applied to comluns output transposed
    #results(to retain shape of df input)
    if axis == 1:
        results = np.transpose(results)
        
    return(results)
    