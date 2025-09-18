import scanpy as sc
import scvelo as scv
import numpy as np
import scipy as scp
import anndata as ad

from scipy.interpolate import splrep, BSpline, UnivariateSpline, splder

def check_velocity(adata, vskey = 'velo_s', vukey = 'velo_u', genes = [], s = 100):
    
    if vskey not in adata.layers or vukey not in adata.layers:
        print('Check vskey and vukey')
        return 0
    
    
    val_scores = {}
    val_error = {}
    pred = {}
    x_list = {}
    for gene in genes:
        
        x = adata.layers['spliced'][:,np.where(adata.var.index.values==gene)[0][0]]
        y = adata.layers['unspliced'][:,np.where(adata.var.index.values==gene)[0][0]]
        
        idx = np.argsort(x) #x
        x = x[idx]
        y = y[idx]

        spl = splrep(x, y, s=s)
        
        yhat = BSpline(*spl)(x)
        deriv = splder(BSpline(*spl))(x)

        velocity = np.concatenate((adata.layers[vskey][:,[np.where(adata.var.index.values==gene)[0][0]]], adata.layers[vukey][:,[np.where(adata.var.index.values==gene)[0][0]]]), axis=-1)[idx]
        
        tangent = np.concatenate([velocity[:,[0]], deriv[:,None]], axis=1)/np.sqrt(velocity[:,0]**2 + deriv**2)[:,None]
        
        cosine = np.abs(np.sum(velocity * tangent, axis=1)/(np.linalg.norm(velocity, axis=-1)*np.linalg.norm(tangent, axis=-1)))
        error = np.sum((yhat - y)**2, axis=-1)
        
        val_scores[gene] = cosine
        val_error[gene] = error
        pred[gene] = yhat
        x_list[gene] = x

    return val_scores, val_error, x_list, pred
