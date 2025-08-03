"""
Create linear color gradients
"""

from matplotlib.colors import ColorConverter, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

import numpy as np


def linear_gradient(cstart, cend, n=10):
    '''
    Return a gradient list of `n` colors going from `cstart` to `cend`.
    '''
    s = np.array(ColorConverter.to_rgb(cstart))
    f = np.array(ColorConverter.to_rgb(cend))

    rgb_list = [s + (t / (n - 1))*(f - s) for t in range(n)]

    return rgb_list


def gradient(start, end, min_angle, color1, color2, meshgrid, mask, ax,
             alpha):
    '''
    Create a linear gradient from `start` to `end`, which is translationally
    invarient in the orthogonal direction.
    The gradient is then cliped by the mask.
    '''
    xs, ys = start
    xe, ye = end

    X, Y = meshgrid

    # get the distance to each point
    d2start = (X - xs)*(X - xs) + (Y - ys)*(Y - ys)
    d2end   = (X - xe)*(X - xe) + (Y - ye)*(Y - ye)

    dmax = (xs - xe)*(xs - xe) + (ys - ye)*(ys - ye)

    # blur
    smin = 0.015*len(X)
    smax = max(smin, 0.1*len(X)*min(min_angle/120, 1))

    sigma = np.clip(dmax*len(X), smin, smax)

    Z = gaussian_filter((d2end < d2start).astype(float), sigma=sigma)

    # generate the colormap
    n_bin = 100

    color_list = linear_gradient(color1, color2, n_bin)

    cmap = LinearSegmentedColormap.from_list("gradient", color_list, N=n_bin)

    im = ax.imshow(Z, interpolation='bilinear', cmap=cmap,
                   origin='lower', extent=[-1, 1, -1, 1], alpha=alpha)

    im.set_clip_path(mask)