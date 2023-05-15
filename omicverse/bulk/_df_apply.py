#import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool


#iterate through rows of DataFrame
def gen_row(ndarray):
    
    for i in range(ndarray.shape[0]):
        yield ndarray[i,:]


#iterate through columns of DataFrame
def gen_col(ndarray):
    
    for i in range(ndarray.shape[1]):
        yield ndarray[:,i]

#apply a function to each row or columns of a DataFrame
def apply(func, df, axis=0, ncores=None, p=None, **kwargs):
    
    #check axis input is 0 or 1
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
    