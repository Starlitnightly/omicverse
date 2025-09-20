import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_absolute_abundances_1D(
    adata_sp,
    roi_subset=False,
    saving=False,
    celltype_subset=False,
    scaling=0.15,
    power=1,
    pws=[0, 0, 100, 500, 1000, 3000, 6000],
    dimName="VCDepth",
    xlab="Cortical Depth",
    colourCode=None,
    figureSize=(12, 8),
):
    r"""
    Plot absolute abundance of celltypes in a dotplot across 1 dimension

    :param adata_sp: anndata object for spatial data with celltype abundance included in .obs (this is returned by running cell2location first)
    :param celltype_subset: list of a subset of cell type names to be plotted
    :param slide&radial_position: if wanting to plot only data from one slide + one radial position, include in these parameters
    :param cell_types: parameter for only plotting specific cell types where column names in adata_sp.obs are meanSpot[celltype] format
    :param roi_subset: optionally a boolean for only using part of the data in adata_sp (corresponding to a specific ROI)
    :param saving: optionally a string value, which will result in the plot to be saved under this name
    :param scaling: how dot size should scale linearly with abundance values, default 0.15
    :param power: how dot size should scale non-linearly with abundance values, default 1 (no non-linear scaling)
    :param pws: which abundance values to show in the legend
    :param dimName: the name of the dimensions in adata_sp.obs to use for plotting
    :param xlab: the x-axis label for the plot
    :param colourCode: optionally a dictionary mapping cell type names to colours
    :param figureSize: size of the figure

    """

    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    def subset_obs_column(adata, celltype):
        obs_columns = adata.obs.loc[:, [celltype in x.split("mean_spot_factors") for x in adata.obs.columns]]
        columns_names = obs_columns.columns
        return columns_names

    def subset_anndata(adata, celltype, dimName):
        adata_subset = adata.copy()
        names = subset_obs_column(adata, celltype)
        adata_subset.obs = adata_subset.obs.loc[:, names]
        adata_subset.obs[dimName] = adata.obs[dimName]
        adata_subset.obs["Radial_position"] = adata.obs["Radial_position"]
        adata_subset.obs["slide"] = adata.obs["slide"]
        return adata_subset

    if celltype_subset:
        adata_sp = subset_anndata(adata_sp, celltype_subset, dimName)

    celltypes = [x.split("meanscell_abundance_w_sf_")[-1] for x in adata_sp.obsm["means_cell_abundance_w_sf"].columns]
    abundances = adata_sp.obsm["means_cell_abundance_w_sf"]

    if roi_subset:
        celltypesForPlot = np.repeat(celltypes, sum(roi_subset))
        vcForPlot = np.array([adata_sp.obs[dimName].loc[roi_subset] for j in range(len(celltypes))]).flatten()
        countsForPlot = np.array([abundances.iloc[:, j].loc[roi_subset] for j in range(len(celltypes))])
    else:
        celltypesForPlot = np.repeat(celltypes, np.shape(adata_sp)[0])
        vcForPlot = np.array([adata_sp.obs[dimName] for j in range(len(celltypes))]).flatten()
        countsForPlot = np.array([abundances.iloc[:, j] for j in range(len(celltypes))])

    if type(colourCode) is dict:
        colourCode = pd.DataFrame(data=colourCode.values(), index=colourCode.keys(), columns=["Colours"])
    else:
        colourCode = pd.DataFrame(data="black", index=celltypes, columns=["Colours"])

    coloursForPlot = np.array(colourCode.loc[np.array(celltypesForPlot), "Colours"])

    plt.figure(figsize=(figureSize))
    plt.scatter(
        vcForPlot,
        celltypesForPlot,
        s=(1 - np.amin(countsForPlot * scaling) + countsForPlot * scaling) ** power,
        c=coloursForPlot,
    )

    plt.xlabel(xlab)

    # make a legend:
    for pw in pws:
        plt.scatter(
            [], [], s=((1 - np.amin(countsForPlot * scaling) + pw * scaling)) ** power, c="black", label=str(pw)
        )

    h, leng = plt.gca().get_legend_handles_labels()
    plt.legend(
        h[1:],
        leng[1:],
        labelspacing=1.2,
        title="Total Number",
        borderpad=1,
        frameon=True,
        framealpha=0.6,
        edgecolor="k",
        facecolor="w",
        bbox_to_anchor=(1.55, 0.5),
    )
    plt.tight_layout()

    if saving:
        plt.savefig(saving)


def plot_density_1D(
    adata_sp,
    subset=None,
    saving=False,
    scaling=0.15,
    power=1,
    pws=[0, 0, 100, 500, 1000, 3000, 6000, 10000],
    dimName="VCDepth",
    areaName="AOISurfaceArea",
    xlab="Cortical Depth",
    colourCode=None,
    figureSize=(12, 8),
):
    r"""Plot density of celltypes in a dotplot across 1 dimension

    :param adata_sp: anndata object for spatial data with celltype abundance included in .obs (this is returned by running cell2location first)
    :param subset: optionally a boolean for only using part of the data in adata_sp
    :param saving: optionally a string value, which will result in the plot to be saved under this name
    :param scaling: how dot size should scale linearly with abundance values, default 0.15
    :param power: how dot size should scale non-linearly with abundance values, default 1 (no non-linear scaling)
    :param pws: which abundance values to show in the legend
    :param dimName: the name of the column in adata_sp.obs that contains the dimension used for plotting
    :param areaName: the name of the column in adata_sp.obs that contain the area of each ROI (assumed to be square micrometer)
    :param xlab: the x-axis label for the plot
    :param colourCode: optionally a dictionary mapping cell type names to colours
    :param figureSize: size of the figure
    """

    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    roi_area = np.array(adata_sp.obs[areaName])
    celltypes = [
        x.split("mean_spot_factors")[-1] for x in adata_sp.obs.columns if len(x.split("mean_spot_factors")) == 2
    ]
    abundances = adata_sp.obs.loc[:, [len(x.split("mean_spot_factors")) == 2 for x in adata_sp.obs.columns]]

    if subset:
        celltypesForPlot = np.repeat(celltypes, sum(subset))
        vcForPlot = np.array([adata_sp.obs[dimName].loc[subset] for j in range(len(celltypes))]).flatten()
        countsForPlot = np.array(
            [abundances.iloc[:, j].loc[subset] / roi_area[subset] * 10**6 for j in range(len(celltypes))]
        )
    else:
        celltypesForPlot = np.repeat(celltypes, np.shape(adata_sp)[0])
        vcForPlot = np.array([adata_sp.obs[dimName] for j in range(len(celltypes))]).flatten()
        countsForPlot = np.array([abundances.iloc[:, j] / roi_area * 10**6 for j in range(len(celltypes))])

    if type(colourCode) is dict:
        colourCode = pd.DataFrame(data=colourCode.values(), index=colourCode.keys(), columns=["Colours"])
    else:
        colourCode = pd.DataFrame(data="black", index=celltypes, columns=["Colours"])

    coloursForPlot = np.array(colourCode.loc[np.array((celltypesForPlot)), "Colours"])

    plt.figure(figsize=(figureSize))
    plt.scatter(
        vcForPlot,
        celltypesForPlot,
        s=((1 - np.amin(countsForPlot * scaling) + countsForPlot * scaling)) ** power,
        c=coloursForPlot,
    )

    plt.xlabel(xlab)

    # make a legend:
    for pw in pws:
        plt.scatter(
            [], [], s=((1 - np.amin(countsForPlot * scaling) + pw * scaling)) ** power, c="black", label=str(pw)
        )

    h, leng = plt.gca().get_legend_handles_labels()
    plt.legend(
        h[1:],
        leng[1:],
        labelspacing=1.2,
        title="Density ($cells/mm^2$)",
        borderpad=1,
        frameon=True,
        framealpha=0.6,
        edgecolor="k",
        facecolor="w",
        bbox_to_anchor=(1, 0.9),
    )
    plt.tight_layout()

    if saving:
        plt.savefig(saving)
