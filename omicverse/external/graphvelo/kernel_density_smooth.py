import math
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde


def dnorm(x, u=0, sig=1):
    """
    Generate the gaussian kernel function.
    """
    return np.exp(-((x - u) ** 2) / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

def bandwidth_nrd(x):
    """
    Determine the prefered bandwidth h (smooth factor) for kernel density estimation
    """
    
    x = pd.Series(x)
    h = (x.quantile([0.75]).values - x.quantile([0.25]).values) / 1.34

    return 4 * 1.06 * min(math.sqrt(np.var(x, ddof=1)), h) * (len(x) ** (-1 / 5))

def rep(x, length):
    len_x = len(x)
    n = int(length / len_x)
    r = length % len_x
    re = []
    for i in range(0, n):
        re = re + x
    for i in range(0, r):
        re = re + [x[i]]
    return re

def kde2d_scipy(x, y, n=25, lims=None):
    """2D Kernel Density Estimation using scipy with adaptive bandwidth."""
    nx = len(x)
    if not lims:
        lims = [min(x), max(x), min(y), max(y)]
    if len(y) != nx:
        raise ValueError("data vectors must be the same length")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(lims).all():
        raise ValueError("data and limits must be finite")

    n = rep(n, length=2) if isinstance(n, list) else rep([n], length=2)
    gx = np.linspace(lims[0], lims[1], n[0])
    gy = np.linspace(lims[2], lims[3], n[1])

    # Estimate the marginal density P(X)
    kde_x = gaussian_kde(x, bw_method='scott')
    p_x = kde_x(gx)

    # Estimate the joint density P(X, Y)
    kde_xy = gaussian_kde([x, y], bw_method='scott')
    positions = np.vstack([np.repeat(gx, len(gy)), np.tile(gy, len(gx))])
    p_xy = kde_xy(positions).reshape(len(gx), len(gy))

    return gx, gy, p_xy

def kde2d_statsmodels(x, y, n=25, lims=None):
    """2D Kernel Density Estimation using statsmodels with adaptive bandwidth."""
    try:
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
    except:
        raise ImportError("If you want to use `statsmodels` as KDE backend, you need to"
                           "install `statsmodels` first")
    nx = len(x)
    if not lims:
        lims = [min(x), max(x), min(y), max(y)]
    if len(y) != nx:
        raise ValueError("data vectors must be the same length")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(lims).all():
        raise ValueError("data and limits must be finite")

    n = rep(n, length=2) if isinstance(n, list) else rep([n], length=2)
    gx = np.linspace(lims[0], lims[1], n[0])
    gy = np.linspace(lims[2], lims[3], n[1])

    # Estimate the joint density P(X, Y) using statsmodels
    kde_xy = KDEMultivariate(data=[x, y], var_type='cc', bw='normal_reference')
    positions = np.vstack([np.repeat(gx, len(gy)), np.tile(gy, len(gx))]).T
    p_xy = kde_xy.pdf(positions).reshape(len(gx), len(gy))

    return gx, gy, p_xy

def kde2d_fixbdw(x, y, n=25, lims=None):
    nx = len(x)
    if not lims:
        lims = [min(x), max(x), min(y), max(y)]
    if len(y) != nx:
        raise Exception("data vectors must be the same length")
    elif (False in np.isfinite(x)) or (False in np.isfinite(y)):
        raise Exception("missing or infinite values in the data are not allowed")
    elif False in np.isfinite(lims):
        raise Exception("only finite values are allowed in 'lims'")
    else:
        n = rep(n, length=2) if isinstance(n, list) else rep([n], length=2)
        gx = np.linspace(lims[0], lims[1], n[0])
        gy = np.linspace(lims[2], lims[3], n[1])
        
        h = [bandwidth_nrd(x), bandwidth_nrd(y)]
        if 0 in h:
            max_vec = [max(x), max(y)]
            h[h == 0] = max_vec[h == 0] / n
        h = [item[0] if isinstance(item, np.ndarray) else item for item in h]
        h = np.array(rep(h, length=2))
            
        if h[0] <= 0 or h[1] <= 0:
            raise Exception("bandwidths must be strictly positive")
        else:
            h /= 4
            ax = pd.DataFrame()
            ay = pd.DataFrame()
            for i in range(len(x)):
                ax[i] = (gx - x[i]) / h[0]
            for i in range(len(y)):
                ay[i] = (gy - y[i]) / h[1]
            
            z = (np.matrix(dnorm(ax)) * np.matrix(dnorm(ay).T)) / (nx * h[0] * h[1])
            
    return gx, gy, z


def kde2d(x, y, n=25, lims=None, backend='scipy'):
    if backend == 'scipy':
        return kde2d_scipy(x, y, n, lims)
    elif backend == 'statsmodels':
        return kde2d_statsmodels(x, y, n, lims)
    elif backend == 'fixbdw':
        return kde2d_fixbdw(x, y, n, lims)
    else:
        raise ValueError("Backend not recognized. Choose 'scipy', 'statsmodels' or 'fixbdw'.")


def kde3d(x, y, z, h=None, n=25, lims=None):
    """
    Three-dimensional kernel density estimation with an axis-aligned
    bivariate normal kernel, evaluated on a cubic grid.

    Arguments
    ---------
        x:  `List`
            x coordinate of data
        y:  `List`
            y coordinate of data
        z:  'List'
            z coordinate of data
        
        h:  `List` (Default: None)
            vector of bandwidths for :math:`x`, :math:`y` directions and :math:'z' directions.  
            Defaults to normal reference bandwidth (see `bandwidth.nrd`). A scalar value will 
            be taken to apply to each directions.
        n: `int` (Default: 25)
            Number of grid points in each direction.  Can be scalar or a length-3 integer list.
        lims: `List` (Default: None)
            The limits of the cubic covered by the grid .

    Returns
    ---------
        A list of three components
        gx, gy, gz: `List`
            The x,y and z coordinates of the grid points, lists of length `n`.
        z:  `List`
            An :math:`n[1]` by :math:`n[2]` by :math: 'n[3]' matrix of the estimated density: first dimension corresponds 
            to the value of :math:`x`, second dimension corresponds to the value of :math:`y`, third dimension corresponds
            to the value of :math:'z'.
    """
    nx = len(x)
    ny = len(y)
    if not lims:
        lims = [min(x), max(x), min(y), max(y), min(z), max(z)]
    if not nx == ny == len(z):
        raise Exception("data vectors must be the same length")
    
    elif (False in np.isfinite(x)) or (False in np.isfinite(y)):
        raise Exception("missing or infinite values in the data are not allowed")
    elif False in np.isfinite(lims):
        raise Exception("only finite values are allowed in 'lims'")
    else:
        n = rep(n, length=3) if isinstance(n, list) else rep([n], length=3)
        gx = np.linspace(lims[0], lims[1], n[0])
        gy = np.linspace(lims[2], lims[3], n[1])
        gz = np.linspace(lims[4], lims[5], n[1])
        if h is None:
            h = [bandwidth_nrd(x), bandwidth_nrd(y), bandwidth_nrd(z)]
        else:
            h = np.array(rep(h, length=3))

        if h[0] <= 0 or h[1] <= 0 or h[2] <= 0:
            raise Exception("bandwidths must be strictly positive")
        else:
            h /= 4
            ax = pd.DataFrame()
            ay = pd.DataFrame()
            az = pd.DataFrame()
            for i in range(len(x)):
                ax[i] = (gx - x[i]) / h[0]
            for i in range(len(y)):
                ay[i] = (gy - y[i]) / h[1]
            for i in range(len(z)):
                az[i] = (gz - z[i]) / h[2]
                
            z = np.einsum('ai, bi, ci -> abc',np.matrix(dnorm(ax)),np.matrix(dnorm(ay)),np.matrix(dnorm(az)))/ (nx* ny * h[0] * h[1]*h[2])
            
    return gx, gy, gz, z

def rescale_density(dens):
    """
    Rescale the preliminary density from the kernel density estimation methods. 
    This function can be applied to (n,n) or (n,n,n) dimensional density matrix.

    Arguments
    ---------
        dens: 'numpy.array'
                numpy.array object with dimension of (n,n) or (n,n,n)

    Returns
    ---------
        rescaled_dens: 'numpy.array'
                numpy.array object with dimension of (n,n) or (n,n,n)
    """
    ndim = dens.ndim
    dens_shape = dens.shape[0]
    dens = np.array(dens)
    
    if ndim == 2:
        
        den_x = np.sum(dens, axis = 1) #condition on each input x, sum over y
        rescaled_dens = np.zeros([dens_shape, dens_shape])
        
        for i in range(dens_shape):
            tmp = dens[i] / den_x[i]  #condition on each input x, normalize over y
            max_val = max(tmp)
            min_val = min(tmp)
            rescaled_val = (tmp - min_val) / (max_val - min_val)
            rescaled_dens[:,i] = rescaled_val
            
    elif ndim == 3:
        
        den_xy = np.sum(dens, axis=2)  # condition on each input x,y, sum over z
        rescaled_dens = np.zeros([dens_shape, dens_shape, dens_shape])
        
        for i in range(dens_shape):
            for j in range(dens_shape): 
                tmp = dens[i,j] / den_xy[i,j]  # condition on each input x, normalize over y
                max_val = max(tmp)
                min_val = min(tmp)
                rescaled_val = (tmp - min_val) / (max_val - min_val)
                rescaled_dens[i,j,:] =  rescaled_val
    
    else:
        print('We have not defined the kde method for more than 3-dimensional data.')
    
    return rescaled_dens

def kde2d_to_mean_and_sigma(gx, gy, dens):
    """
    Use density as weight to smooth data on y-axis. gx, gy are meshgrid matrix, 
    dens should be the density matrix after rescaling.

    Arguments
    ---------
        gx: 'numpy.array'
            numpy.array object, with dimension of (n,n)
        gy: 'numpy.array'
            numpy.array object, with dimension of (n,n)
        dens:'numpy.array'
            numpy.array object, with dimension of (n,n)
            
    Returns
    ---------
        x_grid: 'numpy.array'
            numpy.array object, with dimension of (n,)
        y_mean: 'numpy.array'
            numpy.array object, with dimension of (n,)
        y_sigm: 'numpy.array'
            numpy.array object, with dimension of (n,)      
    """
    
    x_grid = np.unique(gx)
    y_mean = np.zeros(len(x_grid))
    y_sigm = np.zeros(len(x_grid))
    for i, x in enumerate(x_grid):
        mask = gx == x
        den = dens[mask]
        Y_ = gy[mask]
        mean = np.average(Y_, weights = den)
        sigm = np.sqrt(np.average((Y_ - mean) ** 2, weights=den))
        y_mean[i] = mean
        y_sigm[i] = sigm
    return x_grid, y_mean, y_sigm

def kde3d_to_mean_and_sigma(gx, gy, gz, dens):
    
    """
    Use density as weight to smooth data on z-axis. gx, gy, gz are meshgrid matrix, 
    dens should be the density matrix after rescaling.

    Arguments
    ---------
        gx: 'numpy.array'
            numpy.array object, with dimension of (n,n)
        gy: 'numpy.array'
            numpy.array object, with dimension of (n,n)
        gz: 'numpy.array'
            numpy.array object, with dimension of (n,n)
        dens:'numpy.array'
            numpy.array object, with dimension of (n,n,n)
            
    Returns
    ---------
        x_grid: 'numpy.array'
            numpy.array object, with dimension of (n,)
        y_grid: 'numpy.array'
            numpy.array object, with dimension of (n,)
        z_mean: 'numpy.array'
            numpy.array object, with dimension of (n,n)
        z_sigm: 'numpy.array'
            numpy.array object, with dimension of (n,n)      
    """
    
    x_grid = np.unique(gx)
    y_grid = np.unique(gy)
    
    z_mean = np.zeros(len(x_grid)*len(y_grid)).reshape(len(x_grid),len(y_grid))
    z_sigm = np.zeros(len(x_grid)*len(y_grid)).reshape(len(x_grid),len(y_grid))
    
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            
            mask1 = gx == x
            mask2 = gy == y
            mask  = mask1 & mask2
            den = dens[mask]
            z_ = gz[mask]
            mean = np.average(z_, weights=den)
            sigm = np.sqrt(np.average((z_ - mean) ** 2, weights=den))
            z_mean[i,j] = mean
            z_sigm[i,j] = sigm

    return x_grid, y_grid, z_mean, z_sigm


def velocity_kde(adata, regulator, coregulator, effector, axis_layer = 'M_t', drop_zero_cells = True, remove_outlier = True):
    
    """
    Calculate the velocity value of specified effector, 
    then perform the kernel density estimation. Return 
    the grid points' coordinates along each axis and the 
    density at each grid points.
    
    Arguments
    ---------
        adata:       'AnnData object'
        regultor:    `String`
                      Name of the regulator gene.
        
        coregultor:    `String`
                      Name of the regulator gene.

        effector:    'String'
                      Name of the effector gene.

        axis_layer:  'String'
                     Specify the layer.

        drop_zero_cells: 'bool'
                        If True, drop out all the zero expression data points.
            
    Returns
    ---------
        gx: 'numpy.array'
            The grid points' x axis coordinates | Dimension (n,).
            
        gy: 'numpy.array'
            The grid points' y axis coordinates | Dimension (n,).
            
        gz: 'numpy.array'
            The grid points' z axis coordinates | Dimension (n,).
            
        dens:  'numpy.array'
            The densities on each grid point | Dimension (n,n,n).
            
        x_axis: 'numpy.array'
            The regulator's expression array
        
        y_axis: 'numpy.array'
            The coregulator's expression array
            
        z_velocity: 'numpy.array'
            The corresponding velocity value of effector gene


    """
    # Get the effector gene(z)'s index within adata's data frame
    z_gene_index = [i for i, g in enumerate(adata.var_names) if g in effector]

    
    # Get the gene expression profile for regulator(x), coregulator(y) and effector(z)
    x_axis = adata[:, regulator].layers[axis_layer].A.flatten()
    y_axis = adata[:, coregulator].layers[axis_layer].A.flatten()
    z_axis = adata[:, effector].layers[axis_layer].A.flatten()
    
    # Calculate the velocity (model 2)
    v = adata.layers['velocity_hill']
    z_velocity = v[:,z_gene_index]

    # Drop zero expression cells
    if drop_zero_cells:
        finite = np.isfinite(x_axis + y_axis + z_axis)
        nonzero = np.abs(x_axis) + np.abs(y_axis)+np.abs(z_axis) > 0
        valid_ids = np.logical_and(finite, nonzero)
        x_axis = x_axis[valid_ids]
        y_axis = y_axis[valid_ids]
        z_axis = z_axis[valid_ids]
        z_velocity = z_velocity[valid_ids].toarray().flatten()
    
    # Remove outlier 
    if remove_outlier:
        points = np.column_stack((x_axis, y_axis, z_velocity))
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps= 0.9, min_samples=10) # How to choose appropriate hyperparameters?
        labels = dbscan.fit_predict(points)
        x_axis = points[labels == 0, 0]
        y_axis = points[labels == 0, 1]
        z_velocity = points[labels == 0, 2]
        
        # x_axis_outlier = points[labels == -1, 0]
        # y_axis_outlier = points[labels == -1, 1]
        # z_velocity_outlier = points[labels == -1, 2]

    # Setting appropriate bandwidth for kde 
    bandwidth = [bandwidth_nrd(x_axis), bandwidth_nrd(y_axis), bandwidth_nrd(z_velocity)]
    bandwidth  = [item.flatten()[0] if isinstance(item, np.ndarray) else item for item in bandwidth]
    print(f"bandwith shape:{len(bandwidth)}")
    print(f"bandwith:{bandwidth}")
    
    # Setting limits of grid points for kde
    lims=[min(x_axis), max(x_axis), min(y_axis), max(y_axis), min(z_velocity), max(z_velocity)]
    
    #Applying kde3d function 
    gx, gy, gz, dens=kde3d(x_axis, y_axis, z_velocity, h= bandwidth,lims = lims)
    
    return gx, gy, gz, dens, x_axis, y_axis, z_velocity

def jacobian_kde(adata, regulator, effector, axis_layer = 'M_t', drop_zero_cells =True):
    """
    Calculate the jacobian value of specified regulator and effector,
    then perform the kernel density estimation. Return the grid points' coordinates
    along each axis and the density at each grid points.

    Arguments
    ---------
        regultor:    `String`
                      Name of the regulator gene.

        effector:    'String'
                      Name of the effector gene.

        axis_layer:  'String'
                     Specify the layer.

        drop_zero_cells: 'bool'
                        If True, drop out all the zero expression data points.

    Returns
    -------
        gx: 'numpy.array'
            The grid points' x axis coordinates | Dimension (n,).
            
        gy: 'numpy.array'
            The grid points' y axis coordinates | Dimension (n,).
            
        dens:  'numpy.array'
            The densities on each grid point | Dimension (n,n).
            
        x_axis: 'numpy.array'
            The regulator's expression array
            
        y_jacobian: 'numpy.array'
            The corresponding jacobian value of (/partial effector) / (/partial regulator)

    """
    try:
        from dynamo import dyn
    except ImportError:
        raise ImportError(
            "If you want to show jacobian analysis in plotting function, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    #Get the gene expression profile for regulator(x) and effector(y) 
    x_axis = adata[:, regulator].layers[axis_layer].A.flatten().reshape(-1,1)
    y_axis = adata[:, effector].layers[axis_layer].A.flatten().reshape(-1,1)

    # #Get the cell_type information
    # keys_to_check = ['cell_type', 'celltype', 'dyn_phase', 'cell_cycle_phase']
    # cell_type_key = next((key for key in keys_to_check if key in adata.obs.columns), None)
    # cell_type = adata.obs[cell_type_key]

    # color_mapper = get_mapper()
    # color_series = cell_type.map(color_mapper)
    # cell_colors = np.array(color_series).reshape((-1, 1))
    # cell_types = np.array(cell_type).reshape((-1, 1))

    #Calculate the Jacobian
    dyn.vf.jacobian(adata, regulators = [regulator, effector], effectors=[regulator, effector])

    #Specify the gene index within the jacobiann result
    x_gene_index = [i for i, g in enumerate(adata.uns['jacobian_pca']['regulators']) if g in regulator]
    y_gene_index = [i for i, g in enumerate(adata.uns['jacobian_pca']['effectors']) if g in effector]


    #Get Jacobian data for y axis
    if regulator != effector:
        y_jacobian = adata.uns['jacobian_pca']['jacobian_gene'][y_gene_index, x_gene_index, :].reshape(-1,1)
    else:
        y_jacobian = adata.uns['jacobian_pca']['jacobian_gene'].reshape(-1,1)

    #Drop zero expression cells
    if drop_zero_cells:
        # NOTE: For different regulators, valid_ids maybe different. Here, we check 
        # whether there are universal valid ids, then decide whether we need to calculte 
        # it seperately
        if 'universal_valid_id' in adata.uns:
            valid_ids = adata.uns['universal_valid_id']
        else:
            finite = np.isfinite(x_axis + y_axis)
            nonzero = np.abs(x_axis) > 0
            valid_ids = np.logical_and(finite, nonzero)
        x_axis = x_axis[valid_ids].squeeze()
        y_axis = y_axis[valid_ids].squeeze()
        y_jacobian = y_jacobian[valid_ids].squeeze()
        # cell_colors = cell_colors[valid_ids].squeeze()
        # cell_types = cell_types[valid_ids].squeeze()

    # Setting appropriate bandwidth for kde
    bandwidth = [bandwidth_nrd(x_axis), bandwidth_nrd(y_jacobian)]

    # Setting limits of grid points for kde
    lims=[min(x_axis), max(x_axis), min(y_jacobian), max(y_jacobian)]
    # Applying kde2d function 
   
    gx, gy, dens=kde2d(x_axis, y_jacobian, h = bandwidth,lims = lims)


    return gx, gy, dens, x_axis, y_jacobian

def jacobian_kde_3d(adata, regulator, coregulator, effector, axis_layer = 'M_t', drop_zero_cells =True, remove_outlier = True):
    """
    Calculate the jacobian value of specified regulator and effector,
    then perform the kernel density estimation. Return the grid points' coordinates
    along each axis and the density at each grid points.

    Arguments
    ---------
        regultor:    `String`
                      Name of the regulator gene.

        effector:    'String'
                      Name of the effector gene.

        axis_layer:  'String'
                     Specify the layer.

        drop_zero_cells: 'bool'
                        If True, drop out all the zero expression data points.

    Returns
    -------
        gx: 'numpy.array'
            The grid points' x axis coordinates | Dimension (n,).
            
        gy: 'numpy.array'
            The grid points' y axis coordinates | Dimension (n,).
            
        dens:  'numpy.array'
            The densities on each grid point | Dimension (n,n).
            
        x_axis: 'numpy.array'
            The regulator's expression array
            
        y_jacobian: 'numpy.array'
            The corresponding jacobian value of (/partial effector) / (/partial regulator)

    """
    #Get the gene expression profile for regulator(x), coregulator(y) and effector(z)
    try:
        from dynamo import dyn
    except ImportError:
        raise ImportError(
            "If you want to show jacobian analysis in plotting function, you need to install `dynamo` "
            "package via `pip install dynamo-release` see more details at https://dynamo-release.readthedocs.io/en/latest/,")
    
    x_axis = adata[:, regulator].layers[axis_layer].A.flatten().reshape(-1,1)
    y_axis = adata[:, coregulator].layers[axis_layer].A.flatten().reshape(-1,1)
    z_axis = adata[:, effector].layers[axis_layer].A.flatten().reshape(-1,1)

    #Calculate the Jacobian
    dyn.vf.jacobian(adata, regulators = [regulator, effector], effectors=[regulator, effector])

    #Specify the gene index within the jacobiann result
    x_gene_index = [i for i, g in enumerate(adata.uns['jacobian_pca']['regulators']) if g in regulator]
    z_gene_index = [i for i, g in enumerate(adata.uns['jacobian_pca']['effectors']) if g in effector]

    #Get Jacobian data for y axis
    if regulator != effector:
        z_jacobian = adata.uns['jacobian_pca']['jacobian_gene'][z_gene_index, x_gene_index, :].reshape(-1,1)
    else:
        z_jacobian = adata.uns['jacobian_pca']['jacobian_gene'].reshape(-1,1)

    #Drop zero expression cells
    if drop_zero_cells:
        finite = np.isfinite(x_axis + y_axis + z_axis)
        nonzero = np.abs(x_axis) + np.abs(y_axis) + np.abs(z_axis) > 0
        valid_ids = np.logical_and(finite, nonzero)
        x_axis = x_axis[valid_ids]
        y_axis = y_axis[valid_ids]
        z_axis = z_axis[valid_ids]
        z_jacobian = z_jacobian[valid_ids]
    #Remove Outlier
    if remove_outlier:
        points = np.column_stack((x_axis, y_axis, z_jacobian))
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps= 0.09, min_samples=5)
        labels = dbscan.fit_predict(points)
        x_axis = points[labels == 0, 0]
        y_axis = points[labels == 0, 1]
        z_jacobian = points[labels == 0, 2]
        
        # x_axis_outlier = points[labels == -1, 0]
        # y_axis_outlier = points[labels == -1, 1]
        # z_jacobian_outlier = points[labels == -1, 2]

    # Setting appropriate bandwidth for kde 
    bandwidth = [bandwidth_nrd(x_axis), bandwidth_nrd(y_axis), bandwidth_nrd(z_jacobian)]

    # Setting limits of grid points for kde
    lims=[min(x_axis), max(x_axis), min(y_axis), max(y_axis), min(z_jacobian), max(z_jacobian)]

    # Use kde3d function to calculate the density 
    gx, gy, gz, dens = kde3d(x_axis, y_axis, z_jacobian, h=bandwidth,lims = lims)

    return gx, gy, gz, dens, x_axis, y_axis, z_jacobian