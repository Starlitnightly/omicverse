# +
import numpy as np
import pandas as pd

from .plot_spatial import plot_spatial_general as plot_spatial


def interpolate_coord(start=10, end=5, steps=100, accel_power=3, accelerate=True, jitter=None):
    r"""
    Interpolate coordinates between start_array and end_array positions in N steps
    with non-linearity in movement according to acc_power,
    and accelerate change in coordinates (True) or slow it down (False).

    :param jitter: shift positions by a random number by sampling:
      new_coord = np.random.normal(mean=coord, sd=jitter), reasonable values 0.01-0.1
    """

    seq = np.linspace(np.zeros_like(start), np.ones_like(end), steps)
    seq = seq**accel_power

    if jitter is not None:
        seq = np.random.normal(loc=seq, scale=jitter * np.abs(seq))
        seq[0] = np.zeros_like(start)
        seq[steps - 1] = np.ones_like(end)

    if accelerate:
        seq = 1 - seq
        start

    seq = seq * (start - end) + end
    if not accelerate:
        seq = np.flip(seq, axis=0)

    return seq


def expand_1by1(df):
    col6 = [df.copy() for i in range(df.shape[1])]
    index = df.index.astype(str)
    columns = df.columns
    for i in range(len(col6)):
        col6_1 = col6[i]

        col6_1_new = np.zeros_like(col6_1)
        col6_1_new[:, i] = col6_1[col6_1.columns[i]].values

        col6_1_new = pd.DataFrame(col6_1_new, index=index + str(i), columns=columns)
        col6[i] = col6_1_new

    return pd.concat(col6, axis=0)


def plot_video_mapping(
    adata_vis,
    adata,
    sample_ids,
    spot_factors_df,
    sel_clust,
    sel_clust_col,
    sample_id,
    sc_img=None,
    sp_img=None,
    sp_img_scaling_fac=1,
    adata_cluster_col="annotation_1",
    cell_fact_df=None,
    step_n=[20, 100, 15, 45, 80, 30],
    step_quantile=[1, 1, 1, 1, 0.95, 0.95],
    sc_point_size=1,
    aver_point_size=20,
    sp_point_size=5,
    reorder_cmap=range(7),
    label_clusters=False,
    style="dark_background",
    adjust_text=False,
    sc_alpha=0.6,
    sp_alpha=0.8,
    img_alpha=0.8,
    sc_power=20,
    sp_power=20,
    sc_accel_power=3,
    sp_accel_power=3,
    sc_accel_decel=True,
    sp_accel_decel=False,
    sc_jitter=None,
    sp_jitter=None,
    save_path="./results/mouse_viseum_snrna/std_model/mapping_video/",
    crop_x=None,
    crop_y=None,
    save_extension="png",
    colorbar_shape={"vertical_gaps": 2, "horizontal_gaps": 0.13},
):
    r"""
    Create frames for a video illustrating the approach from UMAP of single cells to their spatial locations.
    We use linear interpolation of UMAP and spot coordinates to create movement.

    :param adata_vis: anndata with Visium data (including spatial slot in `.obsm`)
    :param adata: anndata with single cell data (including X_umap slot in `.obsm`)
    :param sample_ids: pd.Series - sample ID for each spot
    :param spot_factors_df: output of the model showing spatial expression of cell types / factors.
    :param sel_clust: selected cluster names in `adata_cluster_col` column of adata.obs
    :param sel_clust_col: selected cluster column name in spot_factors_df
    :param sample_id: sample id to use for visualisation
    :param adata_cluster_col: column in adata.obs containing cluster annotations
    :param cell_fact_df: alternative to adata_cluster_col, pd.DataFrame specifying class for each cell (can be continuous).
    :param step_n: how many frames to record in each step: UMAP, UMAP collapsing into averages, averages, averages expanding into locations, locations.
    :param step_quantile: how to choose maximum colorscale limit in each step? (quantile) Use 1 for discrete values.
    :param sc_point_size: point size for cells
    :param aver_point_size: point size for averages
    :param sp_point_size: point size for spots
    :param fontsize: size of text label of averages
    :param adjust_text: adjust text label position to avoid overlaps
    :param sc_alpha, sp_alpha: color alpha scaling for single cells and spatial.
    :param sc_power, sp_power: change dot size nonlinearly with this exponent
    :param sc_accel_power, sp_accel_power: change movement speed size nonlinearly with this exponent
    :param sc_accel_decel, sp_accel_decel: accelerate (True) or decelereate (False)
    :param save_path: path where to save frames (named according to order of steps)
    """

    from tqdm.auto import tqdm

    # extract spot expression and coordinates
    coords = adata_vis.obsm["spatial"].copy() * sp_img_scaling_fac

    s_ind = sample_ids.isin([sample_id])
    sel_clust_df = spot_factors_df.loc[s_ind, sel_clust_col]
    sel_coords = coords[s_ind, :]
    sample_id = sample_ids[s_ind]

    if sc_img is None:
        # create a black background image
        xy = sel_coords.max(0) + sel_coords.max(0) * 0.05
        sc_img = np.zeros((int(xy[1]), int(xy[0]), 3))

    if sp_img is None:
        # create a black background image
        xy = sel_coords.max(0) + sel_coords.max(0) * 0.05
        sp_img = np.zeros((int(xy[1]), int(xy[0]), 3))
        img_alpha = 1
        img_alpha_seq = 1
    else:
        img_alpha_seq = interpolate_coord(
            start=0, end=img_alpha, steps=step_n[3] + 1, accel_power=sc_power, accelerate=True, jitter=None
        )

    # extract umap coordinates
    umap_coord = adata.obsm["X_umap"].copy()

    # make positive and rescale to fill the image
    umap_coord[:, 0] = umap_coord[:, 0] + abs(umap_coord[:, 0].min()) + abs(umap_coord[:, 0].max()) * 0.01
    umap_coord[:, 1] = -umap_coord[:, 1]  # flip y axis
    umap_coord[:, 1] = umap_coord[:, 1] + abs(umap_coord[:, 1].min()) + abs(umap_coord[:, 1].max()) * 0.01

    if crop_x is None:
        img_width = sc_img.shape[0] * 0.99
        x_offset = 0
        umap_coord[:, 0] = umap_coord[:, 0] / umap_coord[:, 0].max() * img_width
    else:
        img_width = abs(crop_x[0] - crop_x[1]) * 0.99
        x_offset = np.array(crop_x).min()
        umap_coord[:, 0] = umap_coord[:, 0] / umap_coord[:, 0].max() * img_width
        umap_coord[:, 0] = umap_coord[:, 0] + x_offset

    if crop_y is None:
        img_height = sc_img.shape[1] * 0.99
        y_offset = 0
        # y_offset2 = 0
        umap_coord[:, 1] = umap_coord[:, 1] / umap_coord[:, 1].max() * img_height
    else:
        img_height = abs(crop_y[0] - crop_y[1]) * 0.99
        y_offset = np.array(crop_y).min()
        # y_offset2 = sp_img.shape[1] - np.array(crop_y).max()
        umap_coord[:, 1] = umap_coord[:, 1] / umap_coord[:, 1].max() * img_height
        umap_coord[:, 1] = umap_coord[:, 1] + y_offset

    if cell_fact_df is None:
        cell_fact_df = pd.get_dummies(adata.obs[adata_cluster_col], columns=[adata_cluster_col])

    cell_fact_df = cell_fact_df[sel_clust]
    cell_fact_df.columns = cell_fact_df.columns.tolist()
    cell_fact_df["other"] = (cell_fact_df.sum(1) == 0).astype(np.int64)

    # compute average position weighted by cell density
    aver_coord = pd.DataFrame()
    for c in cell_fact_df.columns:
        dens = cell_fact_df[c].values
        dens = dens / dens.sum(0)
        aver = np.array((umap_coord * dens.reshape((cell_fact_df.shape[0], 1))).sum(0))
        aver_coord_1 = pd.DataFrame(aver.reshape((1, 2)), index=[c], columns=["x", "y"])
        aver_coord_1["column"] = c
        aver_coord = pd.concat([aver_coord, aver_coord_1])

    aver_coord = aver_coord.loc[aver_coord.index != "other"]

    # compute movement of cells toward averages (increasing size)
    moving_averages1 = [
        interpolate_coord(
            start=umap_coord,
            end=np.ones_like(umap_coord) * aver_coord.loc[i, ["x", "y"]].values,
            steps=step_n[1] + 1,
            accel_power=sc_accel_power,
            accelerate=sc_accel_decel,
            jitter=sc_jitter,
        )
        for i in aver_coord.index
    ]
    moving_averages1 = np.array(moving_averages1)

    # (increasing dot size) for cells -> averages
    circ_diam1 = interpolate_coord(
        start=sc_point_size,
        end=aver_point_size,
        steps=step_n[1] + 1,
        accel_power=sc_power,
        accelerate=sc_accel_decel,
        jitter=None,
    )

    # compute movement of spots from averages to locations
    moving_averages2 = [
        interpolate_coord(
            start=np.ones_like(sel_coords) * aver_coord.loc[i, ["x", "y"]].values,
            end=sel_coords,
            steps=step_n[4] + 1,
            accel_power=sp_accel_power,
            accelerate=sp_accel_decel,
            jitter=sp_jitter,
        )
        for i in aver_coord.index
    ]
    moving_averages2 = np.array(moving_averages2)

    # (decreasing dot size) for averages -> locations
    circ_diam2 = interpolate_coord(
        start=aver_point_size,
        end=sp_point_size,
        steps=step_n[4] + 1,
        accel_power=sp_power,
        accelerate=sp_accel_decel,
        jitter=None,
    )

    #### start saving plots ####
    # plot UMAP with no changes
    for i0 in tqdm(range(step_n[0])):
        fig = plot_spatial(
            cell_fact_df,
            coords=umap_coord,
            labels=cell_fact_df.columns,
            circle_diameter=sc_point_size,
            alpha_scaling=sc_alpha,
            img=sc_img,
            img_alpha=1,
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[0],  # set to 1 to pick max - essential for discrete scaling
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + 1}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot evolving UMAP from cells to averages
    for i1 in tqdm(range(step_n[1])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != "other"]]
        ann_no_other = expand_1by1(ann_no_other)
        coord = np.concatenate(moving_averages1[:, i1, :, :], axis=0)

        fig = plot_spatial(
            ann_no_other,
            coords=coord,
            labels=ann_no_other.columns,
            circle_diameter=circ_diam1[i1],
            alpha_scaling=sc_alpha,
            img=sc_img,
            img_alpha=1,
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[1],  # set to 1 to pick max - essential for discrete scaling
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + 2}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot averages
    if label_clusters:
        label_clusters = aver_coord[["x", "y", "column"]]
    else:
        label_clusters = None
    for i2 in tqdm(range(step_n[2])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != "other"]]
        ann_no_other = expand_1by1(ann_no_other)
        coord = np.concatenate(moving_averages1[:, i1 + 1, :, :], axis=0)

        fig = plot_spatial(
            ann_no_other,
            coords=coord,
            labels=ann_no_other.columns,
            text=label_clusters,
            circle_diameter=circ_diam1[i1 + 1],
            alpha_scaling=sc_alpha,
            img=sc_img,
            img_alpha=1,
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[2],  # set to 1 to pick max - essential for discrete scaling
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + 3}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot averages & fade-in histology image
    for i22 in tqdm(range(step_n[3])):
        ann_no_other = cell_fact_df[cell_fact_df.columns[cell_fact_df.columns != "other"]]
        ann_no_other = expand_1by1(ann_no_other)
        coord = np.concatenate(moving_averages1[:, i1 + 1, :, :], axis=0)

        fig = plot_spatial(
            ann_no_other,
            coords=coord,
            labels=ann_no_other.columns,
            text=label_clusters,
            circle_diameter=circ_diam1[i1 + 1],
            alpha_scaling=sc_alpha,
            img=sp_img,
            img_alpha=img_alpha_seq[i22],
            style=style,
            # determine max color level using data quantiles
            max_color_quantile=step_quantile[3],  # set to 1 to pick max - essential for discrete scaling
            adjust_text=adjust_text,
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + i22 + 4}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot evolving from averages to spatial locations
    for i3 in tqdm(range(step_n[4])):
        # sel_clust_df_1 = expand_1by1(sel_clust_df)

        dfs = []
        clusters = []
        for i in range(sel_clust_df.shape[1]):
            idx = sel_clust_df.values.argmax(axis=1) == i
            dfs.append(moving_averages2[i, i3, idx, :])
            clusters.append(sel_clust_df[idx])
        # coord = moving_averages2[0, i3, :, :]
        coord = np.concatenate(dfs, axis=0)

        fig = plot_spatial(
            pd.concat(clusters, axis=0),
            coords=coord,
            labels=sel_clust_df.columns,
            circle_diameter=circ_diam2[i3],
            alpha_scaling=sp_alpha,
            img=sp_img,
            img_alpha=img_alpha,
            style=style,
            max_color_quantile=step_quantile[4],
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + i2 + i3 + 5}.{save_extension}", bbox_inches="tight")
        fig.clear()

    # plot a few final images
    for i4 in tqdm(range(step_n[5])):
        dfs = []
        clusters = []
        for i in range(sel_clust_df.shape[1]):
            idx = sel_clust_df.values.argmax(axis=1) == i
            dfs.append(moving_averages2[i, i3 + 1, idx, :])
            clusters.append(sel_clust_df[idx])
        # coord = moving_averages2[0, i3, :, :]
        for d in dfs:
            print(d.shape)
        coord = np.concatenate(dfs, axis=0)

        fig = plot_spatial(
            pd.concat(clusters, axis=0),
            coords=coord,
            labels=sel_clust_df.columns,
            circle_diameter=circ_diam2[i3 + 1],
            alpha_scaling=sp_alpha,
            img=sp_img,
            img_alpha=img_alpha,
            style=style,
            max_color_quantile=step_quantile[5],
            crop_x=crop_x,
            crop_y=crop_y,
            colorbar_position="right",
            colorbar_shape=colorbar_shape,
            reorder_cmap=reorder_cmap,
        )
        fig.savefig(f"{save_path}cell_maps_{i0 + i1 + i2 + i2 + i3 + i4 + 6}.{save_extension}", bbox_inches="tight")
        fig.clear()
