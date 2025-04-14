import numpy as np
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

# Linearly scale isodepth range inside each domain s.t. isodepth approximately measures microns 

# isodepth: N x 1 numpy array of isodepth values per spot
# num_domains: number of domains
# q_vals: num_domains x 1 array
# For each domain i, the spots with the smallest and largest q_vals[i]-percentile isodepth values 
# are used as the "lower" and "upper" domain boundaries, respectively.
# The isodepth values in domain i are scaled such that the range of isodepth values
# is the median physical distance between the two domain boundaries, multiplied by scale_factor

# e.g. for 10Xvisium, use scale_factor=100 since distance between adjacent spots is 100 microns
# for slide-seq, use scale_factor=64/100

# if visualize=True, then also display the lower/upper domain boundaries - this helps with picking q_vals
def adjust_isodepth(gaston_isodepth, gaston_labels, coords_mat, num_domains=None, q_vals=None, scale_factor=1, 
                    visualize=False, figsize=(5,8), num_rows=1,s=5, return_scaling_factors=False):

    L=len(np.unique(gaston_labels))
    gaston_isodepth2=np.copy(gaston_isodepth - np.min(gaston_isodepth))
    if num_domains is None:
        num_domains=len(np.unique(gaston_labels))
    
    domain_ranges=[] # list of isodepth range values for each domain

    if q_vals is None:
        q_vals=[0.15 for i in range(num_domains)]

    if visualize:
        if num_rows is None:
            num_rows=1
        num_cols=int(np.ceil(num_domains/num_rows))
        fig,axs=plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for label in range(num_domains):
        q1=q_vals[label]
        q2=1-q1
        pts1=[t for t in range(len(gaston_isodepth)) if gaston_labels[t]==label and gaston_isodepth[t] < np.quantile(gaston_isodepth[gaston_labels==label],q1)]
        pts2=[t for t in range(len(gaston_isodepth)) if gaston_labels[t]==label and gaston_isodepth[t] > np.quantile(gaston_isodepth[gaston_labels==label],q2)]
    
        zzz=pairwise_distances(coords_mat[pts1,:],coords_mat[pts2,:])
        domain_ranges.append(np.median(np.min(zzz,1)) * scale_factor)

        if visualize:
            r=int(label/num_cols)
            c=label % num_cols
            axs[r,c].scatter(coords_mat[gaston_labels==label,0],coords_mat[gaston_labels==label,1],s=s,c='black')
            axs[r,c].scatter(coords_mat[pts1,0],coords_mat[pts1,1],c='green',s=s)
            axs[r,c].scatter(coords_mat[pts2,0],coords_mat[pts2,1],c='red',s=s)

    for l in range(num_domains):
        pts_l=np.where(gaston_labels==l)[0]
        
        if l > 0:
            pts_l1=np.where(gaston_labels==l-1)[0]
            gaston_isodepth2[pts_l] = gaston_isodepth2[pts_l] + ( np.max(gaston_isodepth2[pts_l1]) - np.min(gaston_isodepth2[pts_l]) )
        
        l_depth_min, l_depth_max = np.min(gaston_isodepth2[pts_l]), np.max(gaston_isodepth2[pts_l])
        length_l = l_depth_max - l_depth_min
        new_length_l=domain_ranges[l]
        
        gaston_isodepth2[pts_l]= 1/(length_l) * (gaston_isodepth2[pts_l] * new_length_l + l_depth_min*(l_depth_max - l_depth_min - new_length_l))

    if return_scaling_factors:
        scale_list=[]
        for l in range(L):
            pts_l=np.where(gaston_labels==l)[0]
            s=(np.max(gaston_isodepth2[pts_l]) - np.min(gaston_isodepth2[pts_l])) / (np.max(gaston_isodepth[pts_l]) - np.min(gaston_isodepth[pts_l]))
            scale_list.append(s)
        return gaston_isodepth2, scale_list
    
    return gaston_isodepth2