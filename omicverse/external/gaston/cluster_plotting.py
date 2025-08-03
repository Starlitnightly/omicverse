import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import normalize

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from .dp_related import rotate_by_theta

def plot_clusters(gaston_labels, S, fig=None, ax=None, figsize=(5,8), colors=None, color_palette=plt.cm.Dark2, s=20,labels=None,
                 lgd=False, rotate=None, show_boundary=False, gaston_isodepth=None, boundary_lw=10,bbox_to_anchor=None,
                 linear_transform=None):

    if rotate is not None:
        S=rotate_by_theta(S,rotate)
    elif linear_transform is not None:
        S=(linear_transform @ S.T).T
    
    
    if fig is None or ax is None:
        fig,ax=plt.subplots(figsize=figsize)

    if colors is None:
        colors=np.array([color_palette(i) for i in range(len(np.unique(gaston_labels)))])

    for i,t in enumerate(np.unique(gaston_labels)):
        pts_t=np.where(gaston_labels==t)[0]
        if labels is not None:
            l=labels[i]
        else:
            l=i
        ax.scatter(S[pts_t,0], S[pts_t,1],s=s, color=colors[int(t)], label=l)

    if show_boundary:
        plt.tricontour(S[:,0], S[:,1], gaston_isodepth, 
               levels=[np.min(gaston_isodepth[gaston_labels==i]) for i in range(len(np.unique(gaston_labels)))[1:]], 
               linewidths=boundary_lw, colors='k', linestyles='-')
    
    plt.axis('off')
    if lgd:
        plt.legend(bbox_to_anchor=bbox_to_anchor)
    
#######

def plot_clusters_restrict(gaston_labels, S, gaston_isodepth, isodepth_min=0, isodepth_max=1, restrict_color='lime', fig=None, ax=None, figsize=(5,8), colors=None, color_palette=plt.cm.Dark2, s=20,labels=None,lgd=False, rotate=None):

    if rotate is not None:
        S=rotate_by_theta(S,rotate)
    
    if fig is None or ax is None:
        fig,ax=plt.subplots(figsize=figsize)

    if colors is None:
        colors=np.array([color_palette(i) for i in range(len(np.unique(gaston_labels)))])

    for i,t in enumerate(np.unique(gaston_labels)):
        pts_t=np.where(gaston_labels==t)[0]
        if labels is not None:
            l=labels[i]
        else:
            l=i
        ax.scatter(S[pts_t,0], S[pts_t,1],s=s, color=colors[int(t)], label=l)
    plt.axis('off')
    if lgd:
        plt.legend()

    locs=np.array( [i for i in range(len(gaston_isodepth)) if isodepth_min < gaston_isodepth[i] < isodepth_max] )
    plt.scatter(S[locs,0],S[locs,1],c=restrict_color,alpha=0.5)

#######

    
def plot_isodepth(gaston_isodepth, S, mod, figsize=(5,8), contours=True, 
                  contour_levels=4, contour_lw=1, contour_fs=10, colorbar=True,
                  s=20,cbar_fs=10, axis_off=True, streamlines=False, streamlines_lw=1.5, 
                  rotate=None, cmap='coolwarm', norm=None,
                 arrowsize=2, neg_gradient=False, scaling_factors=None, 
                 gaston_labels_for_scaling=None, linear_transform=None,
                 n_neighbors=1000):
    
    if rotate is not None:
        S_rotated=rotate_by_theta(S,rotate)
    elif linear_transform is not None:
        S_rotated=(linear_transform @ S.T).T
    else:
        S_rotated=S
    
    fig,ax=plt.subplots(figsize=figsize)

    im1=ax.scatter(S_rotated[:,0], S_rotated[:,1], c=gaston_isodepth, cmap=cmap, s=s, norm=norm)
    if axis_off:
        plt.axis('off')

    if contours:
        CS=ax.tricontour(S_rotated[:,0], S_rotated[:,1], gaston_isodepth, levels=contour_levels, linewidths=contour_lw, colors='k', linestyles='solid')
        ax.clabel(CS, CS.levels, inline=True, fontsize=contour_fs)
    if colorbar:
        cbar=plt.colorbar(im1)
        cbar.ax.tick_params(labelsize=cbar_fs)
    if streamlines:
        x=torch.tensor(S,requires_grad=True).float()
        G=torch.autograd.grad(outputs=mod.spatial_embedding(x).flatten(),inputs=x, grad_outputs=torch.ones_like(x[:,0]))[0]
        G=G.detach().numpy()
        if neg_gradient:
            G=-1*G

        if scaling_factors is not None:
            L=len(np.unique(gaston_labels_for_scaling))
            for l in range(L):
                G[gaston_labels_for_scaling==l,:] = scaling_factors[l] * G[gaston_labels_for_scaling==l,:]
        
        if rotate is not None:
            G_rotated=rotate_by_theta(G,rotate)
        elif linear_transform is not None:
            G_rotated=(linear_transform @ G.T).T
        else:
            G_rotated=G
        
        # CODE FROM scVelo
        smooth=None
        min_mass=None
        
        cutoff_perc=0
        
        X_grid, V_grid = compute_velocity_on_grid(
                    X_emb=S_rotated,
                    V_emb=G_rotated,
                    density=1,
                    smooth=smooth,
                    min_mass=min_mass,
                    n_neighbors=n_neighbors,
                    adjust_for_stream=True,
                    cutoff_perc=cutoff_perc,
                )
        lengths = np.sqrt((V_grid**2).sum(0))
        linewidth=streamlines_lw
        linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
        
        
        density=1
        stream_kwargs = {
                "linewidth": linewidth,
                "density": density,
                "zorder": 3,
                "color": "k",
                "arrowsize": arrowsize,
                "arrowstyle": "-|>",
                "maxlength": 1000,
                "integration_direction": "both",
            }
        
        ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)
        plt.axis('off')

#######################################################

# streamlines code from

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal

def compute_velocity_on_grid(
    X_emb,
    V_emb,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    adjust_for_stream=False,
    cutoff_perc=None,
):
    """TODO."""
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid**2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

    return X_grid, V_grid