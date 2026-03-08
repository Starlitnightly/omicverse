"""
Data reading functions for OmicVerse.

This module provides functions for reading spatial transcriptomics data,
particularly from SpaceRanger output (both bin-level and cell segmentation data).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Literal, Optional, Union
import ast
import warnings
from anndata import AnnData
from PIL import Image
from ..._registry import register_function
from ..single import read_10x_h5, read_10x_mtx

try:
    from ..._settings import Colors
except Exception:
    class Colors:
        """Fallback ANSI color codes when omicverse._settings import is unavailable."""
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

def _require_geopandas():
    try:
        import geopandas as gpd
        from shapely import wkt
    except ImportError as exc:
        raise ImportError(
            "`read_visium_hd_seg` requires `geopandas` and `shapely`. "
            "Install with: pip install geopandas shapely"
        ) from exc
    return gpd, wkt


def _infer_sample_name(path: Path) -> str:
    if path.name == "segmented_outputs" and path.parent.name == "outs":
        return path.parent.parent.name
    if path.name == "outs":
        return path.parent.parent.name
    return path.name


def _read_table_with_auto_sep(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, sep=",")
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")


def _read_spatial_images(
    root: Path,
    hires_image_path: str,
    lowres_image_path: str,
):
    try:
        with Image.open(root / hires_image_path) as hires_img:
            hires = np.asarray(hires_img)
        with Image.open(root / lowres_image_path) as lowres_img:
            lowres = np.asarray(lowres_img)
        return hires, lowres
    except FileNotFoundError as exc:
        warnings.warn(f"Could not load tissue images: {exc}")
        return None, None


def _read_scalefactors(root: Path, scalefactors_path: str) -> dict:
    try:
        with open(root / scalefactors_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as exc:
        warnings.warn(f"Could not load scalefactors: {exc}")
        return {}


def _normalize_classification_entry(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []

    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = ast.literal_eval(parsed)
        except Exception:
            return []

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        normalized = []
        for item in parsed:
            if isinstance(item, str):
                try:
                    item = ast.literal_eval(item)
                except Exception:
                    continue
            if isinstance(item, dict):
                normalized.append(item)
        return normalized
    return []


def _init_spatial_slot(
    adata: AnnData,
    sample: str,
    hires_img,
    lowres_img,
    scalefactors: dict,
) -> None:
    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"][sample] = {}
    if hires_img is not None and lowres_img is not None:
        adata.uns["spatial"][sample]["images"] = {
            "hires": hires_img,
            "lowres": lowres_img,
        }
    adata.uns["spatial"][sample]["scalefactors"] = scalefactors


def _progress(message: str, level: str = "info") -> None:
    color = Colors.CYAN
    if level == "success":
        color = Colors.GREEN
    elif level == "warn":
        color = Colors.WARNING
    print(f"{color}[VisiumHD] {message}{Colors.ENDC}")


@register_function(
    aliases=["read_visium_hd_bin", "visium hd bin", "读取visium hd bin", "10x spatial bin", "space ranger bin"],
    category="io",
    description="Read Visium HD bin-level expression outputs and attach spatial coordinates, images, and scale factors.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_visium_hd_bin('outs/binned_outputs/square_016um')",
        "adata = ov.io.spatial.read_visium_hd_bin('outs', binsize=16, sample='sample1')",
    ],
    related=["io.spatial.read_visium_hd", "io.spatial.read_visium_hd_seg", "io.single.read_10x_h5"],
)
def read_visium_hd_bin(
    path: Union[str, Path],
    sample: Optional[str] = None,
    binsize: int = 16,
    count_h5_path: str = "filtered_feature_bc_matrix.h5",
    count_mtx_dir: str = "filtered_feature_bc_matrix",
    tissue_positions_path: str = "spatial/tissue_positions.parquet",
    hires_image_path: str = "spatial/tissue_hires_image.png",
    lowres_image_path: str = "spatial/tissue_lowres_image.png",
    scalefactors_path: str = "spatial/scalefactors_json.json",
) -> AnnData:
    """
    Read Visium HD bin-level output and attach spatial metadata.

    Parameters
    ----------
    path : str or Path
        Path to the SpaceRanger output directory.
    sample : str, optional
        Sample key stored in ``adata.uns['spatial']``. If ``None``, inferred from path.
    binsize : int, default 16
        Bin size metadata (for example 2/8/16).
    count_h5_path : str
        Relative path to bin-level 10x H5 matrix.
    count_mtx_dir : str
        Relative path to bin-level MTX directory (fallback when H5 unavailable).
    tissue_positions_path : str
        Relative path to tissue positions table (parquet/csv).
    hires_image_path : str
        Relative path to hires tissue image.
    lowres_image_path : str
        Relative path to lowres tissue image.
    scalefactors_path : str
        Relative path to scalefactors JSON.

    Returns
    -------
    anndata.AnnData
        Bin-level AnnData with ``obsm['spatial']`` and ``uns['spatial'][sample]`` metadata.
    """
    root = Path(path).resolve()
    if sample is None:
        sample = _infer_sample_name(root)
    _progress(f"Reading bin-level data from: {root}")
    _progress(f"Sample key: {sample}")

    h5_path = root / count_h5_path
    mtx_path = root / count_mtx_dir
    _progress(f"Loading count matrix (h5='{count_h5_path}', mtx='{count_mtx_dir}')")
    if h5_path.exists():
        try:
            adata = read_10x_h5(h5_path)
        except Exception as exc:
            warnings.warn(f"Failed to read H5 matrix ({h5_path}): {exc}. Falling back to MTX directory.")
            _progress("H5 read failed, falling back to MTX directory", level="warn")
            if not mtx_path.exists():
                raise FileNotFoundError(f"Neither count_h5_path nor count_mtx_dir exists under {root}")
            adata = read_10x_mtx(mtx_path)
    elif mtx_path.exists():
        adata = read_10x_mtx(mtx_path)
    else:
        raise FileNotFoundError(f"Neither {h5_path} nor {mtx_path} found")

    tissue_path = root / tissue_positions_path
    if not tissue_path.exists() and tissue_positions_path.endswith(".parquet"):
        csv_fallback = root / tissue_positions_path.replace(".parquet", ".csv")
        if csv_fallback.exists():
            tissue_path = csv_fallback
    if not tissue_path.exists():
        raise FileNotFoundError(f"Tissue positions file not found: {tissue_path}")
    _progress(f"Loading tissue positions: {tissue_path}")

    try:
        tissue_df = _read_table_with_auto_sep(tissue_path)
    except Exception as exc:
        raise ValueError(f"Could not read tissue positions file {tissue_path}: {exc}")

    if 'barcode' in tissue_df.columns:
        tissue_df = tissue_df.set_index('barcode')
    elif tissue_df.index.name != 'barcode' and len(tissue_df.columns) > 0:
        tissue_df = tissue_df.set_index(tissue_df.columns[0])

    adata.obs = pd.merge(adata.obs, tissue_df, left_index=True, right_index=True, how='left')

    coord_cols = None
    for col_pair in (
        ["pxl_col_in_fullres", "pxl_row_in_fullres"],
        ["pxl_col", "pxl_row"],
        ["x", "y"],
        ["array_col", "array_row"],
    ):
        if all(col in adata.obs.columns for col in col_pair):
            coord_cols = col_pair
            break
    if coord_cols is None:
        raise ValueError(
            "Could not find spatial coordinate columns. "
            "Expected one of: ['pxl_col_in_fullres', 'pxl_row_in_fullres'], "
            "['pxl_col', 'pxl_row'], ['x', 'y'], or ['array_col', 'array_row']"
        )
    adata.obsm["spatial"] = adata.obs[coord_cols].values

    _progress("Loading images and scale factors")
    hires_img, lowres_img = _read_spatial_images(root, hires_image_path, lowres_image_path)
    scalefactors = _read_scalefactors(root, scalefactors_path)
    _init_spatial_slot(adata, sample, hires_img, lowres_img, scalefactors)
    adata.uns["spatial"][sample]["binsize"] = binsize
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata


@register_function(
    aliases=["read_visium_hd_seg", "visium hd segmentation", "读取visium hd 分割", "10x spatial cellseg", "space ranger segmentation"],
    category="io",
    description="Read Visium HD cell-segmentation outputs and attach polygon geometry with centroid coordinates.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_visium_hd_seg('outs/segmented_outputs')",
        "adata = ov.io.spatial.read_visium_hd_seg('outs/segmented_outputs', sample='tumor_A')",
    ],
    related=["io.spatial.read_visium_hd", "io.spatial.read_visium_hd_bin", "io.single.read_10x_h5"],
)
def read_visium_hd_seg(
    path: Union[str, Path],
    sample: Optional[str] = None,
    cell_segmentations_path: str = "graphclust_annotated_cell_segmentations.geojson",
    count_h5_path: str = "filtered_feature_cell_matrix.h5",
    hires_image_path: str = "spatial/tissue_hires_image.png",
    lowres_image_path: str = "spatial/tissue_lowres_image.png",
    scalefactors_path: str = "spatial/scalefactors_json.json",
) -> AnnData:
    """
    Read Visium HD cell-segmentation output and attach geometries + spatial metadata.

    Parameters
    ----------
    path : str or Path
        Path to the cell-segmentation output directory.
    sample : str, optional
        Sample key stored in ``adata.uns['spatial']``. If ``None``, inferred from path.
    cell_segmentations_path : str
        Relative path to segmentation GeoJSON.
    count_h5_path : str
        Relative path to filtered feature-cell matrix (H5).
    hires_image_path : str
        Relative path to hires tissue image.
    lowres_image_path : str
        Relative path to lowres tissue image.
    scalefactors_path : str
        Relative path to scalefactors JSON.

    Returns
    -------
    anndata.AnnData
        Cell-level AnnData with ``obsm['spatial']``, segmentation geometry, and spatial image metadata.
    """
    root = Path(path).resolve()
    if sample is None:
        sample = _infer_sample_name(root)
    _progress(f"Reading cell-segmentation data from: {root}")
    _progress(f"Sample key: {sample}")
    gpd, wkt = _require_geopandas()

    seg_path = root / cell_segmentations_path
    if not seg_path.exists():
        alternative_names = (
            "cell_segmentations.geojson",
            "cell_segmentations_annotated.geojson",
            "annotated_cell_segmentations.geojson",
        )
        fallback = None
        for alt_name in alternative_names:
            candidate = root / alt_name
            if candidate.exists():
                fallback = candidate
                warnings.warn(
                    f"Specified segmentation file '{cell_segmentations_path}' not found. "
                    f"Using '{alt_name}' instead."
                )
                break
        if fallback is None:
            raise FileNotFoundError(
                f"Cell segmentations file not found: {seg_path}\n"
                f"Also tried: {list(alternative_names)}"
            )
        seg_path = fallback

    _progress(f"Loading segmentation geometry: {seg_path}")
    gdf_seg = gpd.read_file(seg_path)
    df = pd.DataFrame(gdf_seg)
    if "cell_id" in df.columns:
        df["cellid"] = df["cell_id"].apply(lambda x: f"cellid_{str(x).zfill(9)}-1")
    elif "cellid" not in df.columns:
        raise ValueError("Segmentation file must contain either 'cell_id' or 'cellid' column.")

    matrix_path = root / count_h5_path
    if not matrix_path.exists():
        raise FileNotFoundError(f"Cell matrix file not found: {matrix_path}")
    _progress(f"Loading count matrix: {matrix_path}")
    adata = read_10x_h5(matrix_path)

    adata = adata[adata.obs_names.isin(df["cellid"]), :]
    if adata.n_obs == 0:
        raise ValueError(
            "No overlapping cell IDs between matrix and segmentation file. "
            "Please confirm `cellid` naming and matrix source."
        )
    df = df.set_index("cellid").loc[adata.obs_names]

    if isinstance(df["geometry"].iloc[0], str):
        df["geometry"] = df["geometry"].apply(wkt.loads)

    df["x"] = df["geometry"].apply(lambda poly: poly.centroid.x)
    df["y"] = df["geometry"].apply(lambda poly: poly.centroid.y)
    adata.obsm["spatial"] = np.array(df[["x", "y"]])


    _progress("Loading images and scale factors")
    hires_img, lowres_img = _read_spatial_images(root, hires_image_path, lowres_image_path)
    scalefactors = _read_scalefactors(root, scalefactors_path)

    if "spot_diameter_fullres" not in scalefactors:
        if "fiducial_diameter_fullres" in scalefactors:
            scalefactors["spot_diameter_fullres"] = scalefactors["fiducial_diameter_fullres"] / 40.0
        else:
            scalefactors["spot_diameter_fullres"] = 20.0

    _init_spatial_slot(adata, sample, hires_img, lowres_img, scalefactors)
    # WKT 字符串存入 obs（h5ad 可序列化），是几何信息的唯一持久化存储
    adata.obs["geometry"] = df["geometry"].apply(lambda g: wkt.dumps(g) if g is not None else "")
    # 写入标记，供 ov.read_h5ad 读取时自动重建 GeoDataFrame
    adata.uns["omicverse_io"] = {"type": "visium_hd_seg", "sample": sample}
    # GeoDataFrame 不存入 uns（无法序列化到 h5ad）
    # 需要时调用 ov.io.spatial.load_geometries(adata, sample) 或直接用 obs["geometry"]
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata


@register_function(
    aliases=["read_visium_hd", "visium hd reader", "读取visium hd", "10x visium hd", "space ranger reader"],
    category="io",
    description="Unified Visium HD entry point that dispatches to bin-level or cell-segmentation readers.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata_bin = ov.io.spatial.read_visium_hd('outs', data_type='bin', binsize=16)",
        "adata_cell = ov.io.spatial.read_visium_hd('outs/segmented_outputs', data_type='cellseg')",
    ],
    related=["io.spatial.read_visium_hd_bin", "io.spatial.read_visium_hd_seg"],
)
def read_visium_hd(
    path: Union[str, Path],
    data_type: Literal["bin", "cellseg"] = "bin",
    sample: Optional[str] = None,
    binsize: int = 16,
    count_h5_path: str = "filtered_feature_bc_matrix.h5",
    count_mtx_dir: str = "filtered_feature_bc_matrix",
    tissue_positions_path: str = "spatial/tissue_positions.parquet",
    cell_segmentations_path: str = "graphclust_annotated_cell_segmentations.geojson",
    cell_matrix_h5_path: str = "filtered_feature_cell_matrix.h5",
    hires_image_path: str = "spatial/tissue_hires_image.png",
    lowres_image_path: str = "spatial/tissue_lowres_image.png",
    scalefactors_path: str = "spatial/scalefactors_json.json",
) -> AnnData:
    """
    Read 10x Visium HD outputs with a single entry point.

    This function dispatches to:
    - ``read_visium_hd_bin`` when ``data_type='bin'``
    - ``read_visium_hd_seg`` when ``data_type='cellseg'``

    Parameters
    ----------
    path : str or Path
        Root directory of a Visium HD run (typically the SpaceRanger output folder).
    data_type : {'bin', 'cellseg'}, default 'bin'
        Data mode to load.
        - ``'bin'``: bin-level matrix + tissue positions.
        - ``'cellseg'``: cell-segmentation polygons + cell matrix.
    sample : str, optional
        Sample key used under ``adata.uns['spatial'][sample]``.
        If not provided, it is inferred by the underlying reader.
    binsize : int, default 16
        Bin size metadata for ``data_type='bin'``.
    count_h5_path : str, default "filtered_feature_bc_matrix.h5"
        Relative H5 matrix path for ``data_type='bin'``.
    count_mtx_dir : str, default "filtered_feature_bc_matrix"
        Relative matrix directory path for ``data_type='bin'``.
    tissue_positions_path : str, default "spatial/tissue_positions.parquet"
        Relative tissue positions path for ``data_type='bin'``.
    cell_segmentations_path : str, default "graphclust_annotated_cell_segmentations.geojson"
        Relative segmentation GeoJSON path for ``data_type='cellseg'``.
    cell_matrix_h5_path : str, default "filtered_feature_cell_matrix.h5"
        Relative cell matrix path for ``data_type='cellseg'``.
    hires_image_path : str, default "spatial/tissue_hires_image.png"
        Relative hires image path (used by both modes).
    lowres_image_path : str, default "spatial/tissue_lowres_image.png"
        Relative lowres image path (used by both modes).
    scalefactors_path : str, default "spatial/scalefactors_json.json"
        Relative scalefactors JSON path (used by both modes).

    Returns
    -------
    AnnData
        AnnData object with expression matrix, spatial coordinates, and image/scalefactor metadata.

    Raises
    ------
    ValueError
        If ``data_type`` is not ``'bin'`` or ``'cellseg'``.

    Examples
    --------
    >>> adata_bin = read_visium_hd("outs", data_type="bin", binsize=16)
    >>> adata_cell = read_visium_hd("outs/segmented_outputs", data_type="cellseg")
    """
    _progress(f"read_visium_hd entry (data_type='{data_type}')")
    if data_type == "bin":
        return read_visium_hd_bin(
            path=path,
            sample=sample,
            binsize=binsize,
            count_h5_path=count_h5_path,
            count_mtx_dir=count_mtx_dir,
            tissue_positions_path=tissue_positions_path,
            hires_image_path=hires_image_path,
            lowres_image_path=lowres_image_path,
            scalefactors_path=scalefactors_path,
        )

    if data_type == "cellseg":
        return read_visium_hd_seg(
            path=path,
            sample=sample,
            cell_segmentations_path=cell_segmentations_path,
            count_h5_path=cell_matrix_h5_path,
            hires_image_path=hires_image_path,
            lowres_image_path=lowres_image_path,
            scalefactors_path=scalefactors_path,
        )

    raise ValueError("`data_type` must be one of {'bin', 'cellseg'}.")


