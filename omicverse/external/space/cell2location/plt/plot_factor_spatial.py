#!pip install plotnine
import numpy as np
import pandas as pd


def plot_factor_spatial(
    adata,
    fact,
    cluster_names,
    fact_ind=[0],
    trans="log",
    sample_name=None,
    samples_col="sample",
    obs_x="imagecol",
    obs_y="imagerow",
    n_columns=6,
    max_col=5000,
    col_breaks=[0.1, 100, 1000, 3000],
    figure_size=(24, 5.7),
    point_size=0.8,
    text_size=9,
):
    r"""Plot expression of factors / cell types in space.
    Convenient but not as powerful as scanpy plotting.

    :param adata: anndata object with spatial data
    :param fact: pd.DataFrame with spatial expression of factors (W), e.g. mod.spot_factors_df
    :param cluster_names: names of those factors to show on a plot
    :param fact_ind: index of factors to plot
    :param trans: transform colorscale? passed to plotnine.scale_color_cmap
    :param sample_name: if anndata object contains multiple samples specify which sample to plot (no warning given if not)
    :param samples_col: if anndata object contains multiple which .obs columns specifies sample?
    :param obs_x: which .obs columns specifies x coordinate?
    :param obs_y: which .obs columns specifies y coordinate?
    :param n_columns: how many factors / clusters to plot in each row (plotnine.facet_grid)
    :param max_col: colorscale maximum expression in fact
    :param col_breaks: colorscale breaks
    :param figure_size: figures size works weirdly (only x axis has an effect, use 24 for 6-column plot, 12 for 3, 8 for 2 ...).
    :param point_size: point size of spots
    :param text_size: text size
    """
    from plotnine import (
        aes,
        coord_fixed,
        element_line,
        element_rect,
        element_text,
        facet_wrap,
        geom_point,
        ggplot,
        ggtitle,
        scale_color_cmap,
        theme,
        theme_bw,
    )

    if sample_name is not None:
        sample_ind = np.isin(adata.obs[samples_col], sample_name)
    else:
        sample_ind = np.repeat(True, adata.shape[0])

    # adata.obsm['X_spatial'][:,0] vs adata.obs['imagecol'] & adata.obs['imagerow']

    for_plot = np.concatenate(
        (
            adata.obs[obs_x].values.reshape((adata.obs.shape[0], 1)),
            -adata.obs[obs_y].values.reshape((adata.obs.shape[0], 1)),
            fact.iloc[:, fact_ind[0]].values.reshape((adata.obs.shape[0], 1)),
            np.array([cluster_names[fact_ind[0]] for j in range(adata.obs.shape[0])]).reshape((adata.obs.shape[0], 1)),
        ),
        1,
    )
    for_plot = pd.DataFrame(for_plot, index=adata.obs.index, columns=["imagecol", "imagerow", "weights", "cluster"])
    # select only correct sample
    for_plot = for_plot.loc[sample_ind, :]

    for i in fact_ind[1:]:
        for_plot1 = np.concatenate(
            (
                adata.obs[obs_x].values.reshape((adata.obs.shape[0], 1)),
                -adata.obs[obs_y].values.reshape((adata.obs.shape[0], 1)),
                fact.iloc[:, i].values.reshape((adata.obs.shape[0], 1)),
                np.array([cluster_names[i] for j in range(adata.obs.shape[0])]).reshape((adata.obs.shape[0], 1)),
            ),
            1,
        )
        for_plot1 = pd.DataFrame(
            for_plot1, index=adata.obs.index, columns=["imagecol", "imagerow", "weights", "cluster"]
        )
        # select only correct sample
        for_plot1 = for_plot1.loc[sample_ind, :]
        for_plot = pd.concat((for_plot, for_plot1))

    for_plot["imagecol"] = pd.to_numeric(for_plot["imagecol"])
    for_plot["imagerow"] = pd.to_numeric(for_plot["imagerow"])
    for_plot["weights"] = pd.to_numeric(for_plot["weights"])
    for_plot["cluster"] = pd.Categorical(for_plot["cluster"], categories=cluster_names[fact_ind], ordered=True)

    # print(np.log(np.max(for_plot['weights'])))
    ax = (
        ggplot(for_plot, aes("imagecol", "imagerow", color="weights"))
        + geom_point(size=point_size)
        + scale_color_cmap("magma", trans=trans, limits=[0.1, max_col], breaks=col_breaks + [max_col])
        + coord_fixed()
        + theme_bw()
        + theme(
            panel_background=element_rect(fill="black", colour="black", size=0, linetype="solid"),
            panel_grid_major=element_line(size=0, linetype="solid", colour="black"),
            panel_grid_minor=element_line(size=0, linetype="solid", colour="black"),
            strip_text=element_text(size=text_size),
        )
        + facet_wrap("~cluster", ncol=n_columns)
        + ggtitle("nUMI from each cell type")
        + theme(figure_size=figure_size)
    )

    return ax


def plot_categ_spatial(mod, adata, sample_col, color, n_columns=2, figure_size=(24, 5.7), point_size=0.8, text_size=9):
    from plotnine import (
        aes,
        coord_fixed,
        element_line,
        element_rect,
        element_text,
        facet_wrap,
        geom_point,
        ggplot,
        theme,
        theme_bw,
    )

    for_plot = adata.obs[["imagecol", "imagerow", sample_col]]
    for_plot["color"] = color

    # fix types
    for_plot["color"] = pd.Categorical(for_plot["color"], ordered=True)
    # for_plot['color'] = pd.to_numeric(for_plot['color'])
    for_plot["sample"] = pd.Categorical(for_plot[sample_col], ordered=False)
    for_plot["imagecol"] = pd.to_numeric(for_plot["imagecol"])
    for_plot["imagerow"] = -pd.to_numeric(for_plot["imagerow"])

    ax = (
        ggplot(for_plot, aes(x="imagecol", y="imagerow", color="color"))
        + geom_point(size=point_size)  # + scale_color_cmap()
        + coord_fixed()
        + theme_bw()
        + theme(
            panel_background=element_rect(fill="black", colour="black", size=0, linetype="solid"),
            panel_grid_major=element_line(size=0, linetype="solid", colour="black"),
            panel_grid_minor=element_line(size=0, linetype="solid", colour="black"),
            strip_text=element_text(size=text_size),
        )
        + facet_wrap("~sample", ncol=n_columns)
        + theme(figure_size=figure_size)
    )

    return ax
