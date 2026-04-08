from __future__ import annotations

from anndata import AnnData

from .._registry import register_function


def _get_cellcharter_backend():
    from ..external.cellcharter import Cluster, aggregate_neighbors, remove_long_links

    return Cluster, aggregate_neighbors, remove_long_links


@register_function(
    aliases=["CellCharter", "cellcharter", "空间域聚类", "空间聚类cellcharter"],
    category="space",
    description="CellCharter-style spatial clustering using spatial graphs, neighborhood aggregation, and GMM clustering",
    prerequisites={"optional_functions": ["space.spatial_neighbors"]},
    requires={"obsm": ["spatial"], "obs": []},
    produces={"obsm": ["X_cellcharter"], "obs": ["cellcharter"]},
    auto_fix="none",
    examples=[
        "model = ov.space.cellcharter(adata, n_clusters=8, use_rep='X_pca')",
        "adata.obs['cellcharter']",
    ],
    related=["space.spatial_neighbors", "utils.cluster"],
)
def cellcharter(
    adata: AnnData,
    n_clusters: int,
    *,
    use_rep: str = "X_pca",
    spatial_key: str = "spatial",
    n_layers: int = 3,
    aggregations: str | list[str] = "mean",
    out_key: str = "X_cellcharter",
    cluster_key: str = "cellcharter",
    connectivity_key: str = "spatial_connectivities",
    distances_key: str = "spatial_distances",
    sample_key: str | None = None,
    build_spatial_graph: bool = True,
    delaunay: bool = True,
    n_neighs: int = 6,
    radius=None,
    trim_long_links: bool = True,
    distance_percentile: float = 99.0,
    random_state: int = 1024,
    covariance_type: str = "full",
    backend: str = "auto",
    batch_size: int | None = None,
    trainer_params: dict | None = None,
):
    """Run a minimal CellCharter workflow on a spatial AnnData object."""
    if use_rep is not None and use_rep not in adata.obsm:
        raise KeyError(
            f"`{use_rep}` was not found in `adata.obsm`. "
            "Provide a dense representation such as `X_pca`."
        )
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"`{spatial_key}` was not found in `adata.obsm`. "
            "CellCharter clustering requires spatial coordinates."
        )

    if connectivity_key not in adata.obsp or distances_key not in adata.obsp:
        if not build_spatial_graph:
            raise KeyError(
                f"`{connectivity_key}` and `{distances_key}` must exist in `adata.obsp` "
                "when `build_spatial_graph=False`."
            )
        if sample_key is not None and sample_key in adata.obs and adata.obs[sample_key].nunique() > 1:
            raise ValueError(
                "Automatic spatial graph construction for CellCharter currently assumes a single sample. "
                "For multi-sample data, build per-sample spatial graphs first and then rerun clustering."
            )
        from ._svg import spatial_neighbors

        spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            n_neighs=n_neighs,
            radius=radius,
            delaunay=delaunay,
            key_added="spatial",
        )

    Cluster, aggregate_neighbors, remove_long_links = _get_cellcharter_backend()

    if trim_long_links:
        remove_long_links(
            adata,
            distance_percentile=distance_percentile,
            connectivity_key=connectivity_key,
            distances_key=distances_key,
        )

    aggregate_neighbors(
        adata,
        n_layers=n_layers,
        aggregations=aggregations,
        connectivity_key=connectivity_key,
        use_rep=use_rep,
        sample_key=sample_key,
        out_key=out_key,
    )

    model = Cluster(
        n_clusters=n_clusters,
        covariance_type=covariance_type,
        batch_size=batch_size,
        trainer_params=trainer_params,
        random_state=random_state,
        backend=backend,
    )
    model.fit(adata, use_rep=out_key)
    adata.obs[cluster_key] = model.predict(adata, use_rep=out_key)
    return model
