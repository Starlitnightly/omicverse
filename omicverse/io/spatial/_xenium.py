"""10x Xenium In Situ reader for OmicVerse spatial I/O."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ..._registry import register_function
from ..single import read_10x_h5

try:
    from ..._settings import Colors
except Exception:  # pragma: no cover
    class Colors:
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"


def _progress(message: str, level: str = "info") -> None:
    color = Colors.CYAN
    if level == "success":
        color = Colors.GREEN
    elif level == "warn":
        color = Colors.WARNING
    print(f"{color}[Xenium] {message}{Colors.ENDC}")


def _resolve(root: Path, *candidates: str) -> Optional[Path]:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def _read_cells_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _boundaries_to_wkt(
    root: Path,
    cell_index: pd.Index,
) -> Optional[pd.Series]:
    """Turn ``cell_boundaries.parquet`` / ``.csv.gz`` into per-cell WKT polygons.

    Xenium ships cell boundaries as a long table — one row per polygon vertex,
    with columns ``cell_id``, ``vertex_x``, ``vertex_y`` (all in microns). We
    group by ``cell_id``, close the ring if needed, and emit a WKT ``POLYGON``
    string. The resulting :class:`pandas.Series` indexed by cell id is what
    :func:`omicverse.pl.spatialseg` consumes via ``adata.obs['geometry']``.
    """
    path = _resolve(root, "cell_boundaries.parquet", "cell_boundaries.csv.gz", "cell_boundaries.csv")
    if path is None:
        return None

    try:
        if path.suffix == ".parquet":
            bnd = pd.read_parquet(path)
        else:
            bnd = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed to read cell boundaries ({path.name}): {exc}")
        return None

    id_col = next((c for c in ("cell_id", "cellID", "CellID", "cell_ID") if c in bnd.columns), None)
    if id_col is None or "vertex_x" not in bnd.columns or "vertex_y" not in bnd.columns:
        warnings.warn(
            f"Unexpected columns in {path.name}: {list(bnd.columns)}; expected "
            "`cell_id`, `vertex_x`, `vertex_y`."
        )
        return None

    bnd[id_col] = bnd[id_col].astype(str)
    # Build WKT POLYGON strings per cell. Use the builtin split-by-group over
    # sorted-by-cell data to avoid a Python loop over every vertex.
    grouped = bnd.groupby(id_col, sort=False)[["vertex_x", "vertex_y"]]

    def _to_wkt(block: pd.DataFrame) -> str:
        xs = block["vertex_x"].to_numpy()
        ys = block["vertex_y"].to_numpy()
        if len(xs) < 3:
            return ""
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
        parts = ", ".join(f"{x:.4f} {y:.4f}" for x, y in zip(xs, ys))
        return f"POLYGON (({parts}))"

    wkts = grouped.apply(_to_wkt)
    wkts.index = wkts.index.astype(str)
    series = wkts.reindex(cell_index.astype(str)).fillna("")
    return series


def _load_experiment_metadata(root: Path) -> dict:
    path = _resolve(root, "experiment.xenium")
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not parse experiment.xenium: {exc}")
        return {}


def _load_morphology_image(root: Path, name: str) -> Optional[np.ndarray]:
    """Load the hires morphology image (OME-TIFF) if present.

    Xenium output ships either a flat ``morphology_focus.ome.tif`` / ``morphology_mip.ome.tif``
    (V1) or a ``morphology_focus/morphology_focus_0000.ome.tif`` directory layout (V2+).
    Only the first page / first resolution level is read — enough for scatter-overlay
    plotting without keeping the full multi-channel pyramid in memory.
    """
    candidates = []
    focus_dir = root / "morphology_focus"
    if focus_dir.is_dir():
        candidates.extend(sorted(focus_dir.glob("morphology_focus_*.ome.tif")))
    candidates.append(root / f"{name}.ome.tif")
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            import tifffile

            with tifffile.TiffFile(cand) as tif:
                series = tif.series[0]
                # Prefer a lower-resolution level to keep memory in check.
                levels = getattr(series, "levels", None) or [series]
                target = levels[-1] if len(levels) > 1 else levels[0]
                arr = target.asarray()
            # Collapse any Z/channel axes — keep a 2-D grayscale for display.
            while arr.ndim > 2:
                arr = arr[0]
            return arr
        except ImportError:
            warnings.warn(
                "tifffile not installed — skipping morphology image. "
                "`pip install tifffile` to enable H&E / DAPI overlay."
            )
            return None
        except Exception as exc:
            warnings.warn(f"Failed to read {cand.name}: {exc}")
    return None


@register_function(
    aliases=["read_xenium", "xenium", "读取xenium", "10x xenium", "xenium in situ"],
    category="io",
    description="Read 10x Genomics Xenium In Situ output bundle (cell_feature_matrix + cells metadata + optional morphology image).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_xenium('/path/to/Xenium_outs/')",
        "adata = ov.io.spatial.read_xenium(",
        "    '/path/to/Xenium_outs/',",
        "    library_id='Breast_Rep1',",
        "    load_image=False,",
        ")",
    ],
    related=["io.spatial.read_nanostring", "io.spatial.read_visium_hd"],
)
def read_xenium(
    path: Union[str, Path],
    *,
    library_id: Optional[str] = None,
    load_image: bool = True,
    image_key: str = "morphology_focus",
    load_boundaries: bool = True,
) -> AnnData:
    """Read a 10x Xenium ``outs`` directory into an AnnData object.

    Handles the standard flat layout shipped by Xenium Onboard Analysis:

    .. code-block:: text

        outs/
          cell_feature_matrix.h5            # gene × cell sparse matrix
          cells.csv.gz    (or cells.parquet)# per-cell metadata incl. centroids
          experiment.xenium                 # run / panel metadata (JSON)
          morphology_focus.ome.tif          # V1 focused morphology image (optional)
          morphology_focus/                 # V2+ morphology folder (optional)
              morphology_focus_0000.ome.tif

    Parameters
    ----------
    path
        The Xenium ``outs`` directory (or any directory containing the files above).
    library_id
        Identifier used as the key under ``adata.uns['spatial']``. Defaults to
        ``experiment.xenium``'s ``region_name`` / ``run_name`` if present, else the
        directory name.
    load_image
        When ``True`` and ``tifffile`` is installed, loads the morphology image so
        :func:`omicverse.pl.spatial` can overlay it. The morphology TIFF is large
        (hundreds of MB); pass ``False`` for lightweight tutorials that only need
        the centroid scatter.
    image_key
        Which morphology image to prefer when both ``morphology_focus`` and
        ``morphology_mip`` are shipped. One of ``'morphology_focus'``,
        ``'morphology_mip'``, ``'morphology'``.
    load_boundaries
        When ``True`` and ``cell_boundaries.parquet`` / ``.csv.gz`` is present,
        converts per-cell polygon vertices to WKT strings stored in
        ``obs['geometry']`` — required by :func:`omicverse.pl.spatialseg`.

    Returns
    -------
    AnnData
        - ``X``: CSR sparse, ``int32`` counts (cells × genes)
        - ``obs``: cell metadata from ``cells.csv.gz``
        - ``obsm['spatial']``: ``(n_obs, 2)`` cell centroids in **microns**
        - ``var``: gene panel metadata
        - ``uns['spatial'][library_id]``:
            - ``'images'['hires']``: morphology image (if loaded)
            - ``'scalefactors'['tissue_hires_scalef']``: ``1 / pixel_size``
              — mapping micron coords to image pixels
            - ``'scalefactors'['spot_diameter_fullres']``: mean cell diameter in
              fullres (image) pixels, for default spot sizing
            - ``'metadata'``: contents of ``experiment.xenium``
    """
    root = Path(path).resolve()
    _progress(f"Reading Xenium data from: {root}")

    mat_path = _resolve(root, "cell_feature_matrix.h5")
    if mat_path is None:
        raise FileNotFoundError(
            f"`cell_feature_matrix.h5` not found in {root}. Download the Xenium outs bundle "
            "and pass its directory as `path`."
        )
    cells_path = _resolve(root, "cells.parquet", "cells.csv.gz", "cells.csv")
    if cells_path is None:
        raise FileNotFoundError(f"`cells.parquet` / `cells.csv.gz` not found in {root}.")

    adata = read_10x_h5(str(mat_path))
    if hasattr(adata.var, "columns") and "feature_types" in adata.var.columns:
        non_gene = adata.var["feature_types"].isin(
            ["Gene Expression", "Multiplexing Capture", "Antibody Capture"]
        )
        # Keep only Gene Expression targets — drop control probes / codewords so
        # downstream QC / HVG / PCA don't waste capacity on them.
        gene_mask = adata.var["feature_types"] == "Gene Expression"
        dropped = int((~gene_mask).sum())
        if dropped:
            _progress(
                f"Dropping {dropped} non-Gene-Expression features "
                f"(control probes / codewords) out of {adata.n_vars}"
            )
            adata = adata[:, gene_mask].copy()

    cells = _read_cells_table(cells_path)
    id_col = next(
        (c for c in ("cell_id", "cellID", "CellID", "cell_ID") if c in cells.columns),
        cells.columns[0],
    )
    cells[id_col] = cells[id_col].astype(str)
    cells = cells.set_index(id_col)

    # Match row order — the matrix and cells file come from the same pipeline so
    # the barcodes intersect exactly, but be defensive about ordering.
    matrix_ids = pd.Index(adata.obs_names.astype(str))
    common = matrix_ids.intersection(cells.index)
    if len(common) != len(matrix_ids):
        warnings.warn(
            f"{len(matrix_ids) - len(common)} cells in cell_feature_matrix.h5 are absent "
            f"from cells metadata and will be dropped."
        )
        adata = adata[adata.obs_names.astype(str).isin(common)].copy()
        matrix_ids = pd.Index(adata.obs_names.astype(str))
    cells = cells.reindex(matrix_ids)

    xy_pairs = [("x_centroid", "y_centroid"), ("CenterX_local_px", "CenterY_local_px")]
    xy = next((pair for pair in xy_pairs if all(c in cells.columns for c in pair)), None)
    if xy is None:
        raise ValueError(
            "Could not find centroid columns in cells metadata. "
            f"Expected one of {xy_pairs}, found {list(cells.columns)}."
        )
    adata.obsm["spatial"] = cells[list(xy)].to_numpy(dtype=np.float32)
    adata.obs = cells.drop(columns=list(xy))

    exp_meta = _load_experiment_metadata(root)
    if library_id is None:
        library_id = (
            exp_meta.get("region_name")
            or exp_meta.get("run_name")
            or root.name
            or "xenium"
        )
    library_id = str(library_id).strip() or "xenium"

    pixel_size_um = float(exp_meta.get("pixel_size", 0.2125))
    # Mean cell diameter (microns) → spot_diameter_fullres (pixels). Uses cell_area
    # when available; falls back to a typical Xenium cell (~15 μm diameter).
    mean_diam_um = 15.0
    if "cell_area" in adata.obs.columns:
        mean_area = float(np.nanmean(adata.obs["cell_area"].to_numpy()))
        if np.isfinite(mean_area) and mean_area > 0:
            mean_diam_um = 2.0 * np.sqrt(mean_area / np.pi)
    spot_diameter_fullres = float(mean_diam_um / pixel_size_um)

    uns_spatial: dict[str, Any] = {
        "images": {},
        "scalefactors": {
            # coord_in_microns * tissue_hires_scalef → coord_in_image_pixels
            "tissue_hires_scalef": 1.0 / pixel_size_um,
            "spot_diameter_fullres": spot_diameter_fullres,
        },
        "metadata": exp_meta,
    }

    if load_image:
        img = _load_morphology_image(root, image_key)
        if img is not None:
            _progress(f"Loaded morphology image {img.shape} from {root}")
            uns_spatial["images"]["hires"] = img
        else:
            _progress("No morphology image loaded (set load_image=False to silence).", level="warn")

    has_geometry = False
    if load_boundaries:
        wkts = _boundaries_to_wkt(root, cell_index=pd.Index(adata.obs_names.astype(str)))
        if wkts is not None:
            adata.obs["geometry"] = wkts.values
            has_geometry = bool((wkts != "").any())
            if has_geometry:
                _progress(
                    f"Loaded cell polygons (geometry WKT) for "
                    f"{int((wkts != '').sum())}/{len(wkts)} cells"
                )
            else:
                _progress("cell_boundaries present but no valid polygons extracted.", level="warn")

    adata.uns["spatial"] = {library_id: uns_spatial}
    adata.uns["omicverse_io"] = {
        "type": "xenium_seg" if has_geometry else "xenium",
        "library_id": library_id,
    }

    _progress(
        f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars}, library_id={library_id})",
        level="success",
    )
    return adata


__all__ = ["read_xenium"]
