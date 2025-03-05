import numpy as np
from .isodepth_scaling import adjust_isodepth
from .cluster_plotting import plot_isodepth
import matplotlib.colors as pltcolors



def restrict_spots(counts_mat, coords_mat, S, gaston_isodepth, gaston_labels, isodepth_min=0, isodepth_max=1, 
                   adjust_physical=True, scale_factor=1, 
                   plotisodepth=False, show_streamlines=False, gaston_model=None, rotate=None, figsize=(6,3), cmap='coolwarm',
                  arrowsize=2, neg_gradient=False,n_neighbors=1000):

    counts_mat2, coords_mat2, S2, gaston_isodepth2, gaston_labels2=filter_rescale_boundary(counts_mat, coords_mat, S,
                                                                                 gaston_isodepth, gaston_labels, 
                                                                                 isodepth_min=isodepth_min, isodepth_max=isodepth_max)
    # make sure labels go from 0, 1, ... and isodepth starts at 0
    if np.min(gaston_labels2) != 0:
        gaston_labels2=gaston_labels2-np.min(gaston_labels2)
    # gaston_isodepth2=gaston_isodepth2 - np.min(gaston_isodepth2)

    if adjust_physical:
        gaston_isodepth2=adjust_isodepth(gaston_isodepth2, gaston_labels2, coords_mat2, scale_factor=scale_factor)

    if plotisodepth:
        plot_isodepth(gaston_isodepth2, S2, gaston_model, figsize=figsize, 
                      streamlines=show_streamlines, rotate=rotate,cmap=cmap, 
                      norm=pltcolors.CenteredNorm(630),arrowsize=arrowsize,
                      neg_gradient=neg_gradient,n_neighbors=n_neighbors)

    return counts_mat2, coords_mat2, gaston_isodepth2, gaston_labels2, S2

def filter_rescale_boundary(counts_mat, coords_mat, S, gaston_isodepth, gaston_labels, isodepth_min=0, isodepth_max=1):
    locs=np.array( [i for i in range(len(gaston_isodepth)) if isodepth_min < gaston_isodepth[i] < isodepth_max] )
    
    counts_mat_subset=counts_mat[locs,:]
    print(f'restricting to {len(locs)} spots')
    coords_mat_subset=coords_mat[locs,:]
    S_subset=S[locs,:]
    gaston_labels_subset=gaston_labels[locs]
    gaston_labels_subset -= np.min( np.unique( gaston_labels[locs] ) )
    gaston_isodepth_subset=gaston_isodepth[locs]
    
    ####
    
    gaston_isodepth_subset0=gaston_isodepth_subset[gaston_labels_subset==0]
    gaston_isodepth_subset1=gaston_isodepth_subset[gaston_labels_subset==1]

    isodepth_ratio=np.std( gaston_isodepth_subset1 ) / np.std( gaston_isodepth_subset0 )

    gaston_isodepth_subset0_new=((gaston_isodepth_subset0 - np.max(gaston_isodepth_subset0)) * isodepth_ratio) + np.max(gaston_isodepth_subset0)

    gaston_isodepth_subset[gaston_labels_subset==0] = gaston_isodepth_subset0_new
    
    return counts_mat_subset, coords_mat_subset, S_subset, gaston_isodepth_subset, gaston_labels_subset
    
