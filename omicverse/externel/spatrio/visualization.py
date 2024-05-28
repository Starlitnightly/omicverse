import pandas as pd
import scanpy as sc
import numpy as np

def spatial_plot(adata, info, color_by="type", size=None, mode=1, palette=None, save_path=None, figsize=None):
    """
    Plots original/mapped spatial coordinates.

    Args:
        adata: Spatial dataset to be plotted.
        info: A DataFrame containing spot/cell id and meta information.
        color_by: Which observation in adata is used to determine the color of each point in the plot.
        size: The size of each point in the plot.
        mode: Select the type of plot data. If mode=1, plot spot dataset. If mode=2, plot single-cell dataset.
        palette: Color palette for coloring the points.
        ax: Axes object to plot on.
        save_path: Path to save the plot as a PNG file.
        figsize: Tuple specifying the size of the figure in inches (width, height).

    """
    if figsize:
        plt.rcParams["figure.figsize"] = figsize

    if mode == 1:
        coor = info.copy()
        coor['id'] = coor.index.tolist()
        coor.columns = ['x', 'y', 'id']
    elif mode == 2:
        coor = info.copy()
        coor = coor[['Cell_xcoord', 'Cell_ycoord', 'cell']]
        coor.columns = ['x', 'y', 'id']

    adata = adata[adata.obs.index.isin(coor.id)]
    left_data = pd.DataFrame(adata.obs.index)
    left_data.columns = ['id']
    right_data = coor
    coordinate = pd.merge(left_data, right_data, on='id', how="left")
    coordinate.drop(['id'], axis=1, inplace=True)
    adata.obsm['spatial'] = np.array(coordinate)

    if save_path:
        sc.pl.embedding(adata, basis="spatial", color=color_by, size=size, palette=palette,show=False)
        plt.savefig(save_path)
    else:
        sc.pl.embedding(adata, basis="spatial", color=color_by, size=size, palette=palette)