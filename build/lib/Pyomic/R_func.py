import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import to_tree


def sign(value):
    #python version of R's sign
    
    if value > 0:
        return(1)
    elif value < 0:
        return(-1)
    else:
        return(0)

def paste(string, n, sep=""):
    #python version of R's paste
    
    results = []
    for i in range(n):
        results.append(string + sep + str(i))
    
    return(results)


def get_heights(Z):
    #python verison of R's dendro$height
    #height = np.zeros(len(dendro["dcoord"]))
    
    #for i, d in enumerate(dendro["dcoord"]):
        #height[i] = d[1]
    
    clusternode = to_tree(Z, True)
    #height = np.array([c.dist for c in clusternode[1]])
    height = np.array([c.dist for c in clusternode[1] if c.is_leaf() != True])
        
    #height.sort()
    
    return(height)
    

def get_merges(z):
    #python version of R's dendro$merge
    n = z.shape[0]
    merges = np.zeros((z.shape[0], 2), dtype=int)
        
    for i in range(z.shape[0]):
        for j in range(2):
            if z[i][j] <= n:
                merges[i][j] = -(z[i][j] + 1)
            else:
                cluster = z[i][j] - n
                merges[i][j] = cluster
                
    return(merges)
    
    
def factor(vector):
    return(vector)


def nlevels(vector):
    #python version of R's nlevels
    return(len(np.unique(vector)))


def levels(vector):
    #python version of R's levels
    return(np.unique(vector))


def tapply(vector, index, function): #can add **args, **kwargs
    #python version of R's tapply
    
    factors = np.unique(index)
    
    #results = pd.Series(np.repeat(np.nan, len(factors)))
    results = np.repeat(np.nan, len(factors))
    #results.index = factors
    
    for i, k in enumerate(factors):
        subset = vector[index == k]
        #results.iloc[i] = function(subset)
        results[i] = function(subset)
    
    return(results)


def tapply_df(df, index, function, axis=0): #can add **args, **kwargs
    #python version of R's tapply
    
    factors = np.unique(index)
    
    if axis == 1:
        #results = pd.DataFrame(np.zeros((len(factors), df.shape[1])))
        results = np.zeros((len(factors), df.shape[1]))
    else:
        #results = pd.DataFrame(np.zeros((df.shape[0], len(factors))))
        results = np.zeros((df.shape[0], len(factors)))
    
    #results.index = factors
    
    if axis == 1:
        for j in range(df.shape[1]):
            for i, k in enumerate(factors):
                subset = df[index == k, j]
                #results.iloc[i, j] = function(subset)
                results[i, j] = function(subset)
    else:
        for i in range(df.shape[0]):
            for j, k in enumerate(factors):
                subset = df[i, index == k]
                #results.iloc[i, j] = function(subset)
                results[i, j] = function(subset)
    
    return(results)


def table(vector):
    
    factors = np.unique(vector)
    results = pd.Series(np.zeros(len(factors), dtype=int))
    results.index = factors
    
    for i, k in enumerate(factors):
        results.iloc[i] = np.sum(vector == k)
        
    return(results)

