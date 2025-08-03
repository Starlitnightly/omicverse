import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_assignment_entropy(
    ad,
    title="Entropy of Metacell Assignment",
    save_as=None,
    show=True,
    bins=None,
    figsize=(5, 5),
):
    """Plot the distribution of assignment entropy over all cells.

    Each cell is assigned with a partial weight
    to a Metacell, and these weights can be used to compute the entropy of assignment as a proxy for confidence
    of each Metacell assignment - lower entropy assignments are more confidence than high entropy assignments.

    :param ad: annData containing 'Metacells_Entropy' column in .obs
    :param title: (str) title for figure
    :param save_as: (str or None) file name to which figure is saved
    :param bins: (int) number of bins for histogram
    :param figsize: (int,int) tuple of integers representing figure size
    :return:
    """
    plt.figure(figsize=figsize)
    sns.distplot(ad.obs["Metacell_Entropy"], bins=bins)
    plt.title(title)
    sns.despine()

    if save_as is not None:
        plt.savefig(save_as, dpi=150, transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_2D(
    ad,
    key="X_umap",
    colour_metacells=True,
    title="Metacell Assignments",
    save_as=None,
    show=True,
    cmap="Set2",
    figsize=(5, 5),
    SEACell_size=20,
    cell_size=10,
):
    """Plot 2D visualization of metacells using the embedding provided in 'key'.

    :param ad: annData containing 'Metacells' label in .obs
    :param key: (str) 2D embedding of data. Default: 'X_umap'
    :param colour_metacells: (bool) whether to colour cells by metacell assignment. Default: True
    :param title: (str) title for figure
    :param save_as: (str or None) file name to which figure is saved
    :param cmap: (str) matplotlib colormap for metacells. Default: 'Set2'
    :param figsize: (int,int) tuple of integers representing figure size
    """
    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs["SEACell"])
    umap["SEACell"] = umap["SEACell"].astype("category")
    mcs = umap.groupby("SEACell").mean().reset_index()

    plt.figure(figsize=figsize)
    if colour_metacells:
        sns.scatterplot(
            x=0, y=1, hue="SEACell", data=umap, s=cell_size, cmap=cmap, legend=None
        )
        sns.scatterplot(
            x=0,
            y=1,
            s=SEACell_size,
            hue="SEACell",
            data=mcs,
            cmap=cmap,
            edgecolor="black",
            linewidth=1.25,
            legend=None,
        )
    else:
        sns.scatterplot(
            x=0, y=1, color="grey", data=umap, s=cell_size, cmap=cmap, legend=None
        )
        sns.scatterplot(
            x=0,
            y=1,
            s=SEACell_size,
            color="red",
            data=mcs,
            cmap=cmap,
            edgecolor="black",
            linewidth=1.25,
            legend=None,
        )

    plt.xlabel(f"{key}-0")
    plt.ylabel(f"{key}-1")
    plt.title(title)
    ax = plt.gca()
    ax.set_axis_off()

    if save_as is not None:
        plt.savefig(save_as, dpi=150, transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_SEACell_sizes(
    ad,
    save_as=None,
    show=True,
    title="Distribution of Metacell Sizes",
    bins=None,
    figsize=(5, 5),
):
    """Plot distribution of number of cells contained per metacell.

    :param ad: annData containing 'Metacells' label in .obs
    :param save_as: (str) path to which figure is saved. If None, figure is not saved.
    :param title: (str) title of figure.
    :param bins: (int) number of bins for histogram
    :param figsize: (int,int) tuple of integers representing figure size
    :return: None.
    """
    assert "SEACell" in ad.obs, 'AnnData must contain "SEACell" in obs DataFrame.'
    label_df = ad.obs[["SEACell"]].reset_index()
    plt.figure(figsize=figsize)
    sns.distplot(label_df.groupby("SEACell").count().iloc[:, 0], bins=bins)
    sns.despine()
    plt.xlabel("Number of Cells per SEACell")
    plt.title(title)

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()
    return pd.DataFrame(label_df.groupby("SEACell").count().iloc[:, 0]).rename(
        columns={"index": "size"}
    )


def plot_initialization(
    ad,
    model,
    plot_basis="X_umap",
    save_as=None,
    show=True,
):
    """Plot archetype initizlation.

    :param ad: annData containing 'Metacells' label in .obs
    :param model: Initilized SEACells model
    :return: None.
    """
    plt.figure()

    plt.scatter(
        ad.obsm[plot_basis][:, 0], ad.obsm[plot_basis][:, 1], s=1, color="lightgrey"
    )
    init_points = ad.obs_names[model.archetypes]
    plt.scatter(
        ad[init_points].obsm[plot_basis][:, 0],
        ad[init_points].obsm[plot_basis][:, 1],
        s=20,
    )
    ax = plt.gca()
    ax.set_axis_off()

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()
