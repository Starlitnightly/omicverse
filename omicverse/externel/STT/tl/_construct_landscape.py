import numpy as np
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture

def construct_landscape(sc_object,thresh_cal_cov = 0.3, scale_axis = 1.0, 
                        scale_land = 1.1, N_grid = 100, coord_key = 'X_umap'):
    """
    Function to construct the landscape of the multi-stable attractors
    
    Parameters  
    ----------  
    sc_object: AnnData object
        Single cell data object
    thresh_cal_cov: float
        Threshold to calculate the covariance matrix
    scale_axis: float
        Scaling factor for the axis
    scale_land: float
        Scaling factor for the landscape
    N_grid: int 
        Number of grid points for the landscape
    coord_key: str 
        Key of the coordinates in the sc_object.obsm
    
    Returns     
    -------
    None, but updates the sc_object.uns with the following keys:   
    land_out: dict
        Dictionary of landscape values and grid points
    
    """
       
    mu_hat = sc_object.uns['da_out']['mu_hat']
    rho = sc_object.obsm['rho']
    projection = sc_object.obsm[coord_key][:,0:2]
    p_hat=sc_object.uns['da_out']['P_hat']
    
    
    labels = np.argmax(rho,axis = 1)
    K = max(labels)+1

    while K<rho.shape[1]:
        rho=rho[:,:-1]
        mu_hat=mu_hat[:-1]
        mu_hat = mu_hat / np.sum(mu_hat)
        p_hat=p_hat[:-1,:-1]
        p_hat=(p_hat.T/np.sum(p_hat, axis=1)).T

    sc_object.obsm['rho']=rho
    sc_object.uns['da_out']['mu_hat']=mu_hat
    sc_object.uns['da_out']['P_hat']=p_hat
    

    
    centers = []
    for i in range(K):
        index = labels==i
        p = np.mean(projection[index], axis=0)
        centers.append(p)
    centers = np.array(centers)
    centers = np.nan_to_num(centers, nan=0.0)
        
    trans_coord = np.matmul(rho,centers)
    
    
    centers = []
    for i in range(K):
        index = labels==i
        p = np.mean(trans_coord[index], axis=0)
        centers.append(p)
    centers = np.array(centers)
    
    mu = np.zeros((K,2))
    precision = np.zeros((K,2,2))    
    
    for i in range(K):
        member_id = np.array(labels == i) 
        stable_id = rho[:,i]>thresh_cal_cov
        select_id = np.logical_or(member_id,stable_id)
        coord_select = trans_coord[select_id,]
        mu[i,:] = np.mean(coord_select,axis = 0)
        precision[i,:,:] = inv(np.cov(coord_select.T))
    

    gmm = GaussianMixture(n_components = K, weights_init=mu_hat, 
                          means_init = mu,precisions_init=precision, max_iter = 20,reg_covar = 1e-03)
    gmm.fit(trans_coord)
    land_cell = -gmm.score_samples(trans_coord)

    coord_min,coord_max = trans_coord.min(axis = 0),trans_coord.max(axis = 0)
    x_grid = scale_axis* np.linspace(coord_min[0],coord_max[0], N_grid)
    y_grid = scale_axis* np.linspace(coord_min[1],coord_max[1], N_grid)
    xv,yv = np.meshgrid(x_grid,y_grid)

    pos = np.empty(xv.shape + (2,))
    pos[:, :, 0] = xv; pos[:, :, 1] = yv
    land_value = -gmm.score_samples(pos.reshape(-1,2)).reshape(N_grid,N_grid)
    land_max_thresh = scale_land*np.max(land_cell)
    land_value[land_value>land_max_thresh] = np.nan

    
    sc_object.uns['land_out'] = {}
    sc_object.uns['land_out']['land_value'] = land_value
    sc_object.uns['land_out']['grid_x'] = xv
    sc_object.uns['land_out']['grid_y'] = yv
    sc_object.uns['land_out']['trans_coord'] = trans_coord
    sc_object.uns['land_out']['cluster_centers'] = centers
    sc_object.obs['land_cell'] = land_cell