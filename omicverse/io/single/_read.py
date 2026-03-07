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
            try:
                import snapatac2 as snap
            except ImportError:
                raise ImportError('snapatac2 is not installed. `pip install snapatac2`')

            print('Using anndata-rs to read h5ad file')
            print('You should run adata.close() after analysis')
            print('Not all function support Rust backend')
            return snap.read(path, **kwargs)

        raise ValueError("backend must be 'python' or 'rust'")

    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')
