import os
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from ..._registry import register_function


class _PlainColors:
    HEADER = ""
    BOLD = ""
    CYAN = ""
    BLUE = ""
    WARNING = ""
    GREEN = ""
    ENDC = ""


Colors = _PlainColors()


@register_function(
    aliases=["转换为pandas", "convert_to_pandas", "to_pandas", "DataFrame转换", "rust_to_pandas"],
    category="utils",
    description="Convert PyDataFrameElem or Rust DataFrame objects to pandas DataFrame",
    examples=[
        "# Convert Rust backend obs to pandas",
        "adata = ov.read('data.h5ad', backend='rust')",
        "obs_df = ov.utils.convert_to_pandas(adata.obs)",
        "print(obs_df)  # Displays as pandas DataFrame",
        "# Convert Rust backend var to pandas",
        "var_df = ov.utils.convert_to_pandas(adata.var)",
    ],
    related=["utils.read", "pp.preprocess", "utils.store_layers"]
)
def convert_to_pandas(df_obj):
    """Convert Rust-backed dataframe-like objects to ``pandas.DataFrame``.

    Parameters
    ----------
    df_obj : Any
        Input dataframe-like object. Supported objects include wrappers that expose
        ``to_pandas()``, slicing-based dataframe access, or column-based retrieval.

    Returns
    -------
    pandas.DataFrame
        Converted pandas DataFrame. Returns an empty DataFrame when conversion fails.
    """
    try:
        if hasattr(df_obj, 'to_pandas'):
            return df_obj.to_pandas()
    except Exception:
        pass

    try:
        import polars as pl
        df_slice = df_obj[:]
        if hasattr(df_slice, 'to_pandas'):
            return df_slice.to_pandas()
        if isinstance(df_slice, pl.DataFrame):
            return df_slice.to_pandas()
        return df_slice
    except Exception:
        pass

    try:
        if hasattr(df_obj, '__getitem__'):
            import polars as pl
            data = {}
            if hasattr(df_obj, 'columns'):
                columns = df_obj.columns
            else:
                return pd.DataFrame()

            for col in columns:
                try:
                    series = df_obj[col]
                    if hasattr(series, 'to_pandas'):
                        data[col] = series.to_pandas()
                    elif isinstance(series, pl.Series):
                        data[col] = series.to_pandas()
                    else:
                        data[col] = series
                except Exception:
                    pass

            if data:
                return pd.DataFrame(data)
    except Exception:
        pass

    return pd.DataFrame()


class PyDataFrameElemWrapper:
    """A wrapper that provides pandas-like interface for PyDataFrameElem."""

    def __init__(self, df_obj):
        self._df_obj = df_obj
        self._pandas_cache = None

    def _get_pandas(self):
        if self._pandas_cache is None:
            self._pandas_cache = convert_to_pandas(self._df_obj)
        return self._pandas_cache

    def head(self, n=5):
        return self._get_pandas().head(n)

    def tail(self, n=5):
        return self._get_pandas().tail(n)

    @property
    def shape(self):
        return self._get_pandas().shape

    @property
    def columns(self):
        return self._get_pandas().columns

    @property
    def index(self):
        return self._get_pandas().index

    @property
    def dtypes(self):
        return self._get_pandas().dtypes

    def info(self, *args, **kwargs):
        return self._get_pandas().info(*args, **kwargs)

    def describe(self, *args, **kwargs):
        return self._get_pandas().describe(*args, **kwargs)

    def to_pandas(self):
        return self._get_pandas()

    def __getitem__(self, key):
        return self._get_pandas()[key]

    def __repr__(self):
        return repr(self._get_pandas())

    def __str__(self):
        return str(self._get_pandas())

    def __len__(self):
        return len(self._get_pandas())

    def __getattr__(self, name):
        return getattr(self._df_obj, name)


@register_function(
    aliases=["包装PyDataFrame", "wrap_dataframe", "pandas_wrapper", "DataFrame包装器"],
    category="utils",
    description="Wrap PyDataFrameElem to provide pandas DataFrame-like interface",
    examples=[
        "adata = ov.read('data.h5ad', backend='rust')",
        "obs_wrapper = ov.utils.wrap_dataframe(adata.obs)",
    ],
    related=["utils.convert_to_pandas", "utils.read"]
)
def wrap_dataframe(df_obj):
    """Wrap a Rust-backed dataframe-like object with a pandas-style interface.

    Parameters
    ----------
    df_obj : Any
        Input dataframe-like object (for example, Rust backend ``obs``/``var``).

    Returns
    -------
    PyDataFrameElemWrapper
        Wrapper object that lazily converts content to pandas and exposes common
        DataFrame methods (``head``, ``tail``, ``shape``, ``columns``, etc.).
    """
    return PyDataFrameElemWrapper(df_obj)


@register_function(
    aliases=["AnnData兼容转换", "convert_adata_for_rust", "fix_adata_compatibility", "修复兼容性", "rust_compatibility"],
    category="utils",
    description="Convert old Python-backend h5ad AnnData to be compatible with Rust backend requirements using snapatac2.AnnData",
    examples=[
        "adata_rust = ov.utils.convert_adata_for_rust(adata, output_file='fixed_data.h5ad')",
        "adata = ov.read('fixed_data.h5ad', backend='rust')",
    ],
    related=["utils.read", "utils.convert_to_pandas", "pp.preprocess"]
)
def convert_adata_for_rust(adata, output_file=None, verbose=True, close_file=True):
    """Convert an AnnData object into a Rust-backend compatible ``.h5ad`` file.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object to convert.
    output_file : str or None, default=None
        Output ``.h5ad`` file path. If ``None``, a temporary file is created.
    verbose : bool, default=True
        Whether to print conversion progress and diagnostics.
    close_file : bool, default=True
        Whether to close the created ``snapatac2.AnnData`` handle before returning.

    Returns
    -------
    str
        Path to the converted Rust-compatible ``.h5ad`` file.

    Raises
    ------
    ImportError
        If ``snapatac2`` is not installed.
    Exception
        Re-raises backend conversion exceptions after cleanup.
    """
    try:
        import snapatac2 as snap
    except ImportError:
        raise ImportError("snapatac2 is required for Rust backend conversion. Install with: pip install snapatac2")

    if output_file is None:
        fd, output_file = tempfile.mkstemp(suffix='.h5ad')
        os.close(fd)

    if verbose:
        print(f"{Colors.HEADER}{Colors.BOLD}🔧 Converting AnnData for Rust Backend using anndata-rs{Colors.ENDC}")
        print(f"   {Colors.CYAN}Original shape: {adata.shape}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Output file: {output_file}{Colors.ENDC}")
        print(f"   {Colors.BLUE}📝 Ensuring unique names...{Colors.ENDC}")

    adata_copy = adata.copy()
    adata_copy.var_names_make_unique()
    adata_copy.obs_names_make_unique()

    def _clean_matrix(X):
        if X is None:
            return None
        if issparse(X):
            X = X.tocsr(copy=True)
            X.sum_duplicates()
            X.sort_indices()
            if np.isnan(X.data).any() or np.isinf(X.data).any():
                X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
                X.eliminate_zeros()
            return X

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X_clean.dtype.kind == 'V':
            if verbose:
                print(f"   {Colors.WARNING}⚠️  Converting void array to float32{Colors.ENDC}")
            X_clean = X_clean.astype(np.float32)
        elif X_clean.dtype == np.float64:
            X_clean = X_clean.astype(np.float32)
        elif X_clean.dtype.kind in ['U', 'S']:
            if verbose:
                print(f"   {Colors.WARNING}⚠️  Converting string array to float32{Colors.ENDC}")
            try:
                X_clean = pd.to_numeric(X_clean.flatten(), errors='coerce').values.reshape(X_clean.shape).astype(np.float32)
                X_clean = np.nan_to_num(X_clean, nan=0.0)
            except Exception:
                X_clean = np.zeros(X_clean.shape, dtype=np.float32)
        return X_clean

    def _clean_dataframe(df):
        if df is None or df.empty:
            return df
        df_clean = df.copy().reset_index(drop=True)
        cols_to_drop = []
        for col in df_clean.columns:
            if df_clean[col].isna().all():
                cols_to_drop.append(col)
                continue
            if pd.api.types.is_categorical_dtype(df_clean[col]):
                try:
                    cat_data = df_clean[col]
                    if len(cat_data.cat.categories) == 0:
                        df_clean[col] = df_clean[col].astype(str)
                    elif not cat_data.cat.ordered:
                        try:
                            from natsort import natsorted
                            new_categories = natsorted(cat_data.cat.categories.astype(str))
                        except ImportError:
                            new_categories = sorted(cat_data.cat.categories.astype(str))
                        df_clean[col] = cat_data.cat.reorder_categories(new_categories)
                except Exception:
                    df_clean[col] = df_clean[col].astype(str)
            elif df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = df_clean[col].fillna('').astype(str)
                except Exception:
                    cols_to_drop.append(col)
                    continue
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                try:
                    if df_clean[col].dtype.kind == 'f':
                        col_data = df_clean[col].values
                        if np.isnan(col_data).any() or np.isinf(col_data).any():
                            col_clean = np.nan_to_num(
                                col_data,
                                nan=0.0,
                                posinf=np.finfo(col_data.dtype).max,
                                neginf=np.finfo(col_data.dtype).min,
                            )
                            df_clean[col] = col_clean
                    elif df_clean[col].dtype.kind in ['i', 'u'] and df_clean[col].isna().any():
                        df_clean[col] = df_clean[col].fillna(0).astype(df_clean[col].dtype)
                except Exception:
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
                    except Exception:
                        cols_to_drop.append(col)
                        continue
            elif df_clean[col].dtype == 'bool':
                df_clean[col] = df_clean[col].fillna(False)

        if cols_to_drop:
            if verbose:
                print(f"   {Colors.WARNING}⚠️  Dropping problematic columns: {cols_to_drop}{Colors.ENDC}")
            df_clean = df_clean.drop(columns=cols_to_drop)

        if df_clean.empty:
            df_clean = pd.DataFrame({'placeholder': [''] * len(df)})
        return df_clean

    if verbose:
        print(f"   {Colors.BLUE}📊 Cleaning data matrices...{Colors.ENDC}")

    X_clean = _clean_matrix(adata_copy.X)
    obs_clean = _clean_dataframe(adata_copy.obs)
    var_clean = _clean_dataframe(adata_copy.var)

    if verbose:
        print(f"   {Colors.BLUE}🔧 Creating anndata-rs AnnData object...{Colors.ENDC}")

    try:
        if verbose:
            print(f"   {Colors.BLUE}📋 Data summary before anndata-rs creation:{Colors.ENDC}")
            print(f"      X: {type(X_clean)} {X_clean.shape if X_clean is not None else 'None'}")
            print(f"      obs: {type(obs_clean)} {obs_clean.shape if obs_clean is not None else 'None'}")
            print(f"      var: {type(var_clean)} {var_clean.shape if var_clean is not None else 'None'}")

        adata_snap = snap.AnnData(filename=output_file, X=X_clean)

        if obs_clean is not None and not obs_clean.empty:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding obs data...{Colors.ENDC}")
            for col in obs_clean.columns:
                if obs_clean[col].dtype == 'object' or pd.api.types.is_categorical_dtype(obs_clean[col]):
                    obs_clean[col] = obs_clean[col].astype(str)
            adata_snap.close()
            adata_snap = snap.AnnData(filename=output_file, X=X_clean, obs=obs_clean)

        if var_clean is not None and not var_clean.empty:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding var data...{Colors.ENDC}")
            for col in var_clean.columns:
                if var_clean[col].dtype == 'object' or pd.api.types.is_categorical_dtype(var_clean[col]):
                    var_clean[col] = var_clean[col].astype(str)
            adata_snap.close()
            adata_snap = snap.AnnData(filename=output_file, X=X_clean, obs=obs_clean, var=var_clean)

        if verbose:
            print(f"   {Colors.BLUE}📝 Setting obs_names and var_names...{Colors.ENDC}")
        adata_snap.obs_names = [str(name) for name in adata_copy.obs_names]
        adata_snap.var_names = [str(name) for name in adata_copy.var_names]

        if hasattr(adata_copy, 'obsp') and adata_copy.obsp:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding obsp matrices...{Colors.ENDC}")
            for key, value in adata_copy.obsp.items():
                if value is not None:
                    adata_snap.obsp[key] = _clean_matrix(value)

        if hasattr(adata_copy, 'varp') and adata_copy.varp:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding varp matrices...{Colors.ENDC}")
            for key, value in adata_copy.varp.items():
                if value is not None:
                    adata_snap.varp[key] = _clean_matrix(value)

        if hasattr(adata_copy, 'layers') and adata_copy.layers:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding layers...{Colors.ENDC}")
            for key, value in adata_copy.layers.items():
                if value is not None:
                    adata_snap.layers[key] = _clean_matrix(value)

        if close_file:
            adata_snap.close()

        if verbose:
            print(f"   {Colors.GREEN}🎉 Conversion completed successfully!{Colors.ENDC}")
            print(f"   {Colors.GREEN}✅ Rust-compatible file saved: {output_file}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
        return output_file

    except Exception as e:
        if verbose:
            print(f"   {Colors.WARNING}❌ Error during conversion: {e}{Colors.ENDC}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass
        raise


def _fix_dataframe_for_rust(df, df_type="dataframe"):
    """Fix DataFrame for Rust backend compatibility."""
    if df is None or df.empty:
        return df

    df_fixed = df.copy()
    if not isinstance(df_fixed.index, pd.RangeIndex):
        df_fixed.index = pd.Index(df_fixed.index, name=df_fixed.index.name)

    for col in df_fixed.columns:
        if pd.api.types.is_categorical_dtype(df_fixed[col]):
            cat_data = df_fixed[col]
            if not cat_data.cat.ordered:
                try:
                    from natsort import natsorted
                    new_categories = natsorted(cat_data.cat.categories.astype(str))
                except ImportError:
                    new_categories = sorted(cat_data.cat.categories.astype(str))
                df_fixed[col] = cat_data.cat.reorder_categories(new_categories)
        elif df_fixed[col].dtype == 'object':
            try:
                non_null_values = df_fixed[col].dropna()
                if len(non_null_values) > 0 and not all(isinstance(x, str) for x in non_null_values):
                    df_fixed[col] = df_fixed[col].astype(str)
            except Exception:
                pass
        elif pd.api.types.is_numeric_dtype(df_fixed[col]) and df_fixed[col].dtype.kind == 'f':
            col_data = df_fixed[col].values
            if np.isnan(col_data).any() or np.isinf(col_data).any():
                col_clean = np.nan_to_num(
                    col_data,
                    nan=0.0,
                    posinf=np.finfo(col_data.dtype).max,
                    neginf=np.finfo(col_data.dtype).min,
                )
                df_fixed[col] = col_clean
    return df_fixed


def _fix_uns_for_rust(uns_dict):
    """Fix uns dictionary for Rust backend compatibility."""
    if not isinstance(uns_dict, dict):
        return uns_dict

    uns_fixed = {}
    for key, value in uns_dict.items():
        if value is None:
            uns_fixed[key] = value
        elif isinstance(value, np.ndarray):
            if value.dtype.kind == 'f':
                uns_fixed[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                uns_fixed[key] = value
        elif issparse(value):
            sorted_csr = value.tocsr(copy=True)
            sorted_csr.sum_duplicates()
            sorted_csr.sort_indices()
            uns_fixed[key] = sorted_csr
        elif isinstance(value, pd.DataFrame):
            uns_fixed[key] = _fix_dataframe_for_rust(value, f"uns[{key}]")
        elif isinstance(value, dict):
            uns_fixed[key] = _fix_uns_for_rust(value)
        elif isinstance(value, list):
            try:
                if all(isinstance(x, (int, float, np.integer, np.floating)) for x in value):
                    arr = np.array(value)
                    if arr.dtype.kind == 'f':
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    uns_fixed[key] = arr
                else:
                    uns_fixed[key] = value
            except Exception:
                uns_fixed[key] = value
        else:
            uns_fixed[key] = value
    return uns_fixed
