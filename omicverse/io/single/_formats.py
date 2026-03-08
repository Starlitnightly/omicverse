from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import anndata.utils
import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from ..._registry import register_function

try:
    from anndata.io import read_h5ad as _anndata_read_h5ad
    from anndata.io import read_mtx as _anndata_read_mtx
except ImportError:
    from anndata import read_h5ad as _anndata_read_h5ad
    from anndata import read_mtx as _anndata_read_mtx


@register_function(
    aliases=["read_h5ad", "h5ad reader", "读取h5ad", "single-cell h5ad", "anndata reader"],
    category="io",
    description="Read an .h5ad file into an AnnData object using the anndata backend.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.single.read_h5ad('pbmc3k.h5ad')",
        "adata = ov.io.single.read_h5ad('sample.h5ad', backed='r')",
    ],
    related=["io.single.read", "io.single.read_10x_h5", "io.single.read_10x_mtx"],
)
def read_h5ad(filename, **kwargs):
    r"""Read an ``.h5ad`` file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the input ``.h5ad`` file.
    **kwargs
        Additional keyword arguments forwarded to :func:`anndata.read_h5ad`.

    Returns
    -------
    anndata.AnnData
        Loaded AnnData object. 若文件由 :func:`~omicverse.io.spatial.read_visium_hd_seg`
        生成，将自动从 ``obs['geometry']`` 重建 GeoDataFrame 并写入
        ``uns['spatial'][sample]['geometries']``（需安装 geopandas）。
    """
    return _anndata_read_h5ad(filename, **kwargs)


@register_function(
    aliases=["read_10x_h5", "10x h5 reader", "读取10x h5", "cellranger h5", "10x matrix h5"],
    category="io",
    description="Read 10x Genomics HDF5 matrices (Cell Ranger format) and return an AnnData object.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.single.read_10x_h5('filtered_feature_bc_matrix.h5')",
        "adata = ov.io.single.read_10x_h5('filtered_feature_bc_matrix.h5', gex_only=True)",
    ],
    related=["io.single.read_10x_mtx", "io.single.read_h5ad", "io.single.read"],
)
def read_10x_h5(
    filename,
    *,
    genome: str | None = None,
    gex_only: bool = True,
    backup_url: str | None = None,
) -> AnnData:
    r"""Read a 10x Genomics HDF5 matrix file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the 10x ``.h5`` matrix file.
    genome : str or None, default=None
        Genome identifier to keep for legacy multi-genome files.
        Ignored for single-genome inputs.
    gex_only : bool, default=True
        If ``True``, keep only features with ``feature_types == 'Gene Expression'``.
    backup_url : str or None, default=None
        Reserved parameter for API compatibility. Remote fallback is not implemented.

    Returns
    -------
    anndata.AnnData
        AnnData object with barcodes in ``obs_names`` and features in ``var_names``.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(str(path), "r") as f:
        v3 = "/matrix" in f

    if v3:
        with warnings.catch_warnings():
            if genome or gex_only:
                warnings.filterwarnings("ignore", r".*names are not unique", UserWarning)
            adata = _read_10x_h5(path, _read_v3_10x_h5)
        if genome:
            if genome not in adata.var["genome"].values:
                raise ValueError(
                    f"Could not find data corresponding to genome {genome!r} in {path}. "
                    f"Available genomes are: {list(adata.var['genome'].unique())}."
                )
            adata = adata[:, adata.var["genome"] == genome]
        if gex_only:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"]
        if adata.is_view:
            adata = adata.copy()
    else:
        adata = _read_10x_h5(path, partial(_read_legacy_10x_h5, genome=genome))

    return adata


def _read_10x_h5(path: Path, cb) -> AnnData:
    with h5py.File(str(path), "r") as f:
        try:
            return cb(f)
        except KeyError as e:
            raise Exception("File is missing one or more required datasets.") from e


def _collect_datasets(dsets: dict, group: h5py.Group) -> None:
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            dsets[k] = v[()]
        else:
            _collect_datasets(dsets, v)


def _read_v3_10x_h5(f: h5py.File) -> AnnData:
    from scipy.sparse import csr_matrix

    dsets = {}
    _collect_datasets(dsets, f["matrix"])

    n_cols, n_rows = dsets["shape"]  # transposed
    data = dsets["data"]
    if dsets["data"].dtype == np.dtype("int32"):
        data = dsets["data"].view("float32")
        data[:] = dsets["data"]
    matrix = csr_matrix(
        (data, dsets["indices"], dsets["indptr"]),
        shape=(n_rows, n_cols),
    )
    obs_dict = {"obs_names": dsets["barcodes"].astype(str)}
    var_dict = {"var_names": dsets["name"].astype(str)}

    if "gene_id" not in dsets:
        var_dict["gene_ids"] = dsets["id"].astype(str)
    else:
        var_dict.update({
            "gene_ids": dsets["gene_id"].astype(str),
            "probe_ids": dsets["id"].astype(str),
        })
    var_dict["feature_types"] = dsets["feature_type"].astype(str)
    if "filtered_barcodes" in f["matrix"]:
        obs_dict["filtered_barcodes"] = dsets["filtered_barcodes"].astype(bool)

    if "features" in f["matrix"]:
        var_dict.update(
            (
                feature_metadata_name,
                dsets[feature_metadata_name].astype(
                    bool if feature_metadata_item.dtype.kind == "b" else str
                ),
            )
            for feature_metadata_name, feature_metadata_item in f["matrix"]["features"].items()
            if isinstance(feature_metadata_item, h5py.Dataset)
            and feature_metadata_name not in ["name", "feature_type", "id", "gene_id", "_all_tag_keys"]
        )
    else:
        raise ValueError("10x h5 has no features group")

    return AnnData(matrix, obs=obs_dict, var=var_dict)


def _read_legacy_10x_h5(f: h5py.File, genome: str | None) -> AnnData:
    from scipy.sparse import csr_matrix

    children = list(f.keys())
    if not genome:
        if len(children) > 1:
            raise ValueError(
                f"{f.filename} contains more than one genome. "
                "For legacy 10x h5 files you must specify the genome "
                "if more than one is present. "
                f"Available genomes are: {children}"
            )
        genome = children[0]
    elif genome not in children:
        raise ValueError(
            f"Could not find genome {genome!r} in {f.filename}. "
            f"Available genomes are: {children}"
        )

    dsets = {}
    _collect_datasets(dsets, f[genome])

    n_cols, n_rows = dsets["shape"]
    data = dsets["data"]
    if dsets["data"].dtype == np.dtype("int32"):
        data = dsets["data"].view("float32")
        data[:] = dsets["data"]
    matrix = csr_matrix(
        (data, dsets["indices"], dsets["indptr"]),
        shape=(n_rows, n_cols),
    )
    return AnnData(
        matrix,
        obs=dict(obs_names=dsets["barcodes"].astype(str)),
        var=dict(
            var_names=dsets["gene_names"].astype(str),
            gene_ids=dsets["genes"].astype(str),
        ),
    )


@register_function(
    aliases=["read_10x_mtx", "10x mtx reader", "读取10x mtx", "cellranger mtx", "10x matrix market"],
    category="io",
    description="Read 10x Genomics Matrix Market output directory into an AnnData object.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.single.read_10x_mtx('filtered_feature_bc_matrix')",
        "adata = ov.io.single.read_10x_mtx('filtered_feature_bc_matrix', var_names='gene_ids')",
    ],
    related=["io.single.read_10x_h5", "io.single.read_h5ad", "io.single.read"],
)
def read_10x_mtx(
    path,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    gex_only: bool = True,
    prefix: str | None = None,
    compressed: bool = True,
) -> AnnData:
    r"""Read a 10x Genomics Matrix Market directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing ``matrix.mtx`` and matching feature/barcode files.
    var_names : {'gene_symbols', 'gene_ids'}, default='gene_symbols'
        Which feature column should be used as ``adata.var_names``.
    make_unique : bool, default=True
        Whether to enforce unique ``var_names`` when using gene symbols.
    gex_only : bool, default=True
        If ``True``, keep only features labeled as ``Gene Expression`` (v3+ format).
    prefix : str or None, default=None
        Optional filename prefix before ``matrix.mtx``, ``features.tsv``, and ``barcodes.tsv``.
    compressed : bool, default=True
        Whether v3+ files are expected to be gzip-compressed (``.gz``).
        Set ``False`` for plain-text exports such as some STARsolo outputs.

    Returns
    -------
    anndata.AnnData
        Loaded expression matrix with barcodes and feature annotations.
    """
    path = Path(path)
    prefix = "" if prefix is None else prefix
    is_legacy = (path / f"{prefix}genes.tsv").is_file()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r".*names are not unique", UserWarning)
        adata = _read_10x_mtx(
            path,
            var_names=var_names,
            make_unique=make_unique,
            prefix=prefix,
            is_legacy=is_legacy,
            compressed=compressed,
        )
    if is_legacy or not gex_only:
        return adata
    gex_rows = adata.var["feature_types"] == "Gene Expression"
    return adata[:, gex_rows].copy()


def _read_10x_mtx(
    path: Path,
    *,
    var_names: Literal["gene_symbols", "gene_ids"] = "gene_symbols",
    make_unique: bool = True,
    prefix: str = "",
    is_legacy: bool,
    compressed: bool = True,
) -> AnnData:
    suffix = "" if is_legacy else (".gz" if compressed else "")
    adata = _anndata_read_mtx(path / f"{prefix}matrix.mtx{suffix}").T
    genes = pd.read_csv(
        path / f"{prefix}{'genes' if is_legacy else 'features'}.tsv{suffix}",
        header=None,
        sep="\t",
    )
    if var_names == "gene_symbols":
        var_names_idx = pd.Index(genes[1].array)
        if make_unique:
            var_names_idx = anndata.utils.make_index_unique(var_names_idx)
        adata.var_names = var_names_idx.astype("str")
        adata.var["gene_ids"] = genes[0].array
    elif var_names == "gene_ids":
        adata.var_names = genes[0].array.astype("str")
        adata.var["gene_symbols"] = genes[1].array
    else:
        raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
    if not is_legacy:
        adata.var["feature_types"] = genes[2].array
    barcodes = pd.read_csv(path / f"{prefix}barcodes.tsv{suffix}", header=None)
    adata.obs_names = barcodes[0].array
    return adata
