r"""
Visualization modules of SLAT 
"""
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import colors

from .multi_dataset import *


def hist(data,
         bins: Optional[int]=100,
         cut: Optional[float]=0.9,
         cut_height: Optional[int]=200
    )->None:
    r"""
    Histogram of distribution
    
    Parameters:
    -----------
    data
        array or list
    bins
        number of bins
    cut
        cutoff line location
    cut_height
        cutoff line height
    """
    plt.hist(data, bins=bins, facecolor="blue", edgecolor="blue", alpha=0.7)
    plt.vlines(cut, 0, cut_height, color="red", linestyles='dotted', label=str(cut))
    plt.xlabel(r'Similarity Score')
    plt.ylabel(r'Cells')
    plt.show()
    

def make_ground(x_shape,y_shape,color='white'):
    r"""
    Make the background of ImageContainer class
    """
    rgb = colors.to_rgb(color)
    ground = np.array(np.ones([x_shape,y_shape,3]) * rgb)
    return ground
