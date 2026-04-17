import os
import time
from pathlib import Path

import pandas as pd

try:
    from anndata.io import read_h5ad as _anndata_read_h5ad
except ImportError:
    from anndata import read_h5ad as _anndata_read_h5ad

from ..._registry import register_function


def _log_rust_read(path: str, size_mb: float | None, elapsed: float) -> None:
    size_str = f"  ({size_mb:.1f} MB)" if size_mb is not None else ""
    print(
        "📂 Reading with anndata-rs (Rust · out-of-memory)\n"
        f"   {path}{size_str}\n"
        f"   ✓ Loaded in {elapsed:.2f}s\n\n"
        "💡 Data stays on disk. Use ov.pp.* for chunked processing.\n"
        "   adata.close() when done · adata.to_adata() to materialise"
    )


def _read_h5ad_rust(path, **kwargs):
    try:
        import anndataoom
    except ImportError:
        raise ImportError(
            "Rust backend requires the 'anndataoom' package. "
            "Install with:  pip install omicverse[rust]   (or  pip install anndataoom)"
        ) from None

    kwargs.setdefault("backed", "r")
    try:
        size_mb = os.path.getsize(path) / 1024**2
    except OSError:
        size_mb = None

    t0 = time.perf_counter()
    adata = anndataoom.read(str(path), **kwargs)
    _log_rust_read(str(path), size_mb, time.perf_counter() - t0)
    return adata


@register_function(
    aliases=["读取数据", "read", "load_data", "数据读取", "file_reader"],
    category="utils",
    description="Universal file reader for common bioinformatics data formats including h5ad, csv, tsv, txt, and gzipped files",
    examples=[
        "# Read AnnData file",
        "adata = ov.read('data.h5ad')",
        "# Read CSV file",
        "df = ov.read('data.csv')",
        "# Read TSV file",
        "df = ov.read('data.tsv')",
        "# Read compressed file",
        "df = ov.read('data.csv.gz')",
        "# Pass additional parameters",
        "df = ov.read('data.csv', index_col=0, header=0)"
    ],
    related=["utils.read_csv", "utils.read_h5ad", "pp.preprocess"]
)
def read(path, backend='python', **kwargs):
    r"""
    Read common omics file formats into AnnData or pandas DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.
    backend : {'python', 'rust'}, default='python'
        Backend used for ``.h5ad`` reading.
    **kwargs
        Additional keyword arguments forwarded to backend readers.

    Returns
    -------
    anndata.AnnData or pandas.DataFrame
        Loaded AnnData object (for ``.h5ad``) or DataFrame (for table files).

    Raises
    ------
    ImportError
        If ``backend='rust'`` is requested but ``snapatac2`` is unavailable.
    ValueError
        If ``backend`` is invalid for ``.h5ad`` reading or the file suffix is unsupported.
    """
    ext = Path(path).suffix.lower()

    if ext == '.h5ad':
        if backend == 'python':
            return _anndata_read_h5ad(path, **kwargs)

        if backend == 'rust':
            return _read_h5ad_rust(path, **kwargs)

        raise ValueError("backend must be 'python' or 'rust'")

    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')
