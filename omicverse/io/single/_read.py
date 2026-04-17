from pathlib import Path

import pandas as pd

try:
    from anndata.io import read_h5ad as _anndata_read_h5ad
except ImportError:
    from anndata import read_h5ad as _anndata_read_h5ad

from ..._registry import register_function


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
            # Try backends in order of preference:
            #   1. anndataoom (our prebuilt package, includes the Python wrapper)
            #   2. anndata_rs (standalone, built from source)
            #   3. snapatac2 (bundles the same Rust AnnData implementation)
            rs_module = None
            try:
                import anndataoom
                # If anndataoom is available, use its read() directly
                # (it already wraps the result in AnnDataOOM with full API)
                import time, os as _os
                from ..anndata_oom._repr import _format_read_message
                try:
                    size_mb = _os.path.getsize(str(path)) / 1024**2
                except Exception:
                    size_mb = None
                if 'backed' not in kwargs:
                    kwargs['backed'] = 'r'
                t0 = time.time()
                adata = anndataoom.read(str(path), **kwargs)
                elapsed = time.time() - t0
                print(_format_read_message(str(path), size_mb, elapsed))
                return adata
            except ImportError:
                pass

            try:
                import anndata_rs as rs_module
            except ImportError:
                try:
                    import snapatac2 as rs_module  # snapatac2 bundles anndata-rs
                except ImportError:
                    raise ImportError(
                        "No Rust AnnData backend available. Install one of:\n"
                        "  1. pip install anndataoom    (prebuilt, includes full Python wrapper)\n"
                        "  2. pip install snapatac2     (bundles anndata-rs, ~200 MB)\n"
                        "  3. Build anndata-rs from source"
                    )

            from ..anndata_oom import AnnDataOOM
            from ..anndata_oom._repr import _format_read_message
            import time
            import os

            # Default to read-only mode to avoid modifying the original file
            if 'backed' not in kwargs:
                kwargs['backed'] = 'r'
            try:
                size_mb = os.path.getsize(str(path)) / 1024**2
            except Exception:
                size_mb = None

            t0 = time.time()
            rs_adata = rs_module.read(str(path), **kwargs)
            elapsed = time.time() - t0
            adata = AnnDataOOM(rs_adata)
            print(_format_read_message(str(path), size_mb, elapsed))
            return adata

        raise ValueError("backend must be 'python' or 'rust'")

    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')
