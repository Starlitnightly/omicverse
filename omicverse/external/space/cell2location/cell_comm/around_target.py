import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def compute_weighted_average_around_target(
    adata,
    target_cell_type_quantile: float = 0.995,
    source_cell_type_quantile: float = 0.95,
    normalisation_quantile: float = 0.95,
    distance_bin: list = None,
    sample_key: str = "sample",
    genes_to_use_as_source: list = None,
    gene_symbols: str = None,
    obsm_spatial_key: str = "X_spatial",
    normalisation_key: str = None,
    layer: str = None,
    cell_abundance_key: str = "cell_abundance_w_sf",
    cell_abundance_quantile_key: str = "q05",
):
    """
    Compute average abundance of source cell types or genes around each target cell type.

    Parameters
    ----------
    adata
        AnnData object of spatial dataset with cell2location results
    target_cell_type_quantile
        Quantile of target cell type abundance to use for defining
        a set locations with highest abundance of target cell types.
        Cell abundance below this thereshold is set to 0.
    source_cell_type_quantile
        Quantile of source cell type abundance to use for defining
        a set locations with highest abundance of source cell types.
        Cell abundance or RNA abundance for genes below this thereshold is set to 0.
    normalisation_quantile
        Quantile of source cell type or source RNA abundance for genes to use as normalising constant.
        This step can be seen as scaling that puts all source cell types or genes into the same scale.
    distance_bin
        If using concentric bins list with two elements specifying inner and outer edge of the bin.
        Distances specified in coordinates of `obsm_spatial_key`.
    sample_key
        `adata.obs` column key specifying distinct sections across
        which distance bin computation is invalid.
    genes_to_use_as_source
        To request RNA abundance of genes around target cells provide a list of
        var_names or gene SYMBOLs.
    gene_symbols
        `adata.var` column key containing gene symbols
    obsm_spatial_key
        `adata.obsm` key containing spatial coordinates (can be 2D or 3D or N-D).
    normalisation_key
        RNA abundance must be normalised using y_s technical effect term
        estimated by cell2location. Provide `adata.obsm` key containing this normalisation term.
    layer
        adata.layers to use for getting RNA abundance. Default: `adata.X`
    cell_abundance_key
        which cell2location variable to use as cell abundance
    cell_abundance_quantile_key
        which quantile of cell abundance to use

    Returns
    -------
    pd.DataFrame of average abundance of source cell types or RNA abundance of requested genes
    around target cell types.

    """
    # save initial names
    if genes_to_use_as_source is None:
        source_names = adata.uns["mod"]["factor_names"]
    else:
        source_names = genes_to_use_as_source
        # if using gene symbols get var names:
        if gene_symbols is not None:
            source_names = adata.var[gene_symbols][adata.var[gene_symbols].isin(genes_to_use_as_source).values]
            genes_to_use_as_source = adata.var_names[adata.var[gene_symbols].isin(genes_to_use_as_source).values]

    cell_abundance_key_ = cell_abundance_quantile_key + cell_abundance_key
    cell_abundance_key = cell_abundance_quantile_key + "_" + cell_abundance_key

    # create result data frame to be completed
    weighted_avg = pd.DataFrame(
        index=[f"target {ct}" for ct in adata.uns["mod"]["factor_names"]],
        columns=source_names,
    )
    if genes_to_use_as_source is None:
        # pick locations where source cell type abundance is above source_cell_type_quantile
        source_cell_type_filter = adata.obsm[cell_abundance_key] > adata.obsm[cell_abundance_key].quantile(
            source_cell_type_quantile
        )
        # zero-out source cell abundance below selected quantile
        source_cell_type_data = adata.obsm[cell_abundance_key] * source_cell_type_filter
        # get normalising quantile values
        source_normalisation_quantile = adata.obsm[cell_abundance_key].quantile(normalisation_quantile, axis=0)
        # compute average abundance above this quantile
        source_normalisation_quantile = np.average(
            adata.obsm[cell_abundance_key],
            weights=adata.obsm[cell_abundance_key] > source_normalisation_quantile,
            axis=0,
        )
    else:
        # get RNA abundance data
        if layer is None:
            source_cell_type_data = adata[:, genes_to_use_as_source].X.toarray()
        else:
            source_cell_type_data = adata[:, genes_to_use_as_source].layers[layer].toarray()
        # apply technical across-location normalisation
        if normalisation_key:
            source_cell_type_data = source_cell_type_data / adata.obsm[normalisation_key]
        # pick locations where source cell type abundance is above source_cell_type_quantile
        source_cell_type_filter = source_cell_type_data > np.quantile(
            source_cell_type_data, q=source_cell_type_quantile, axis=0
        )
        # zero-out source cell abundance below selected quantile
        source_cell_type_data = source_cell_type_data * source_cell_type_filter
        # create a dataframe with initial source RNA abundance
        source_cell_type_data = pd.DataFrame(
            source_cell_type_data,
            index=adata.obs_names,
            columns=source_names,
        )
        # get normalising quantile values
        source_normalisation_quantile = source_cell_type_data.quantile(normalisation_quantile, axis=0)
        # compute average abundance above this quantile
        source_normalisation_quantile = np.average(
            source_cell_type_data,
            weights=source_cell_type_data > source_normalisation_quantile,
            axis=0,
        )

    # [optional] compute average source_cell_type_data across closes locations (concentric circles)
    if distance_bin is not None:
        # iterate over samples of connected location from the same sections
        # or independent chunks registered 3D data
        for s in adata.obs[sample_key].unique():
            # get sample observations
            sample_ind = adata.obs[sample_key].isin([s])

            # compute distances bewteen locations
            from scipy.spatial.distance import cdist

            distances = cdist(adata[sample_ind, :].obsm[obsm_spatial_key], adata[sample_ind, :].obsm[obsm_spatial_key])
            # select locations in distance bin
            binary_distance = csr_matrix((distances > distance_bin[0]) & (distances <= distance_bin[1]))
            # compute average abundance across locations within a bin
            data_ = (
                (binary_distance @ csr_matrix(source_cell_type_data.loc[sample_ind, :].values))
                .multiply(1 / binary_distance.sum(1))
                .toarray()
            )
            # to account for locations with no neighbours within a bin (sum == 0)
            data_[np.isnan(data_)] = 0
            # complete the average for a given sample
            source_cell_type_data.loc[sample_ind, :] = data_
    # normalise data by normalising quantile (global value across distance bins)
    source_cell_type_data = source_cell_type_data / source_normalisation_quantile
    # account for cases of undetected signal
    source_cell_type_data[source_cell_type_data.isna()] = 0

    # compute average for each target cell type
    for ct in adata.uns["mod"]["factor_names"]:
        # find locations containing high abundance of target cell type
        target_cell_type_filter = adata.obsm[cell_abundance_key][f"{cell_abundance_key_}_{ct}"] > adata.obsm[
            cell_abundance_key
        ][f"{cell_abundance_key_}_{ct}"].quantile(target_cell_type_quantile)
        # use thresholded abundance of target cell type as a weight
        weights = adata.obsm[cell_abundance_key][f"{cell_abundance_key_}_{ct}"] * target_cell_type_filter
        # normalise for target cell type abundance
        target_quantile = adata.obsm[cell_abundance_key][f"{cell_abundance_key_}_{ct}"].quantile(normalisation_quantile)
        target_quantile = np.average(
            adata.obsm[cell_abundance_key][f"{cell_abundance_key_}_{ct}"].values,
            weights=adata.obsm[cell_abundance_key][f"{cell_abundance_key_}_{ct}"].values > target_quantile,
        ).flatten()
        assert target_quantile.shape == (1,), target_quantile.shape
        weights = weights / target_quantile
        # compute the final weighted average
        weighted_avg_ = np.average(
            source_cell_type_data,
            weights=weights,
            axis=0,
        )
        # weighted_avg_[weighted_avg_.isna()] = 0

        weighted_avg_ = pd.Series(weighted_avg_, name=ct, index=source_names)

        # hack to make self interactions less apparent
        weighted_avg_[ct] = weighted_avg_[~weighted_avg_.index.isin([ct])].max() + 0.02
        # complete the results dataframe
        weighted_avg.loc[f"target {ct}", :] = weighted_avg_

    return weighted_avg.astype("float32")


def melt_data_frame_per_signal(weighted_avg_dict, source_var, distance_bins):
    source_var_1 = pd.DataFrame(
        np.array([weighted_avg_dict[str(distance_bin)][source_var].values for distance_bin in distance_bins]),
        columns=weighted_avg_dict[str(distance_bins[0])].index,
        index=[np.mean(distance_bin) for distance_bin in distance_bins],
    ).T

    source_var_1 = source_var_1.melt(
        value_name="Abundance",
        var_name="Distance bin",
        ignore_index=False,
    )
    source_var_1["Target"] = source_var_1.index
    source_var_1["Signal"] = source_var
    return source_var_1


def melt_signal_target_data_frame(weighted_avg_dict, distance_bins):
    source_vars = weighted_avg_dict[str(distance_bins[0])].columns

    return pd.concat(
        [melt_data_frame_per_signal(weighted_avg_dict, source_var, distance_bins) for source_var in source_vars]
    )
