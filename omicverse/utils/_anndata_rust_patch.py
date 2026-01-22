r"""
AnnData Rust backend compatibility patches.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import Iterable

import numpy as np

from .._settings import Colors


def _ov_debug_enabled() -> bool:
    try:
        val = os.environ.get("OV_DEBUG") or os.environ.get("OMICVERSE_DEBUG")
        if val is None:
            return False
        return str(val).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _dbg(msg: str) -> None:
    if _ov_debug_enabled():
        try:
            print(msg)
        except Exception:
            pass


def _is_rust_backend(adata) -> bool:
    return hasattr(adata.X, "__module__") and "snapatac2" in adata.X.__module__


def _convert_df_to_pandas(df_obj):
    try:
        if hasattr(df_obj, "to_pandas"):
            return df_obj.to_pandas()
    except Exception:
        pass
    try:
        import polars as pl
        df_slice = df_obj[:]
        if hasattr(df_slice, "to_pandas"):
            return df_slice.to_pandas()
        if isinstance(df_slice, pl.DataFrame):
            return df_slice.to_pandas()
        return df_slice
    except Exception:
        pass
    try:
        import pandas as pd
        if hasattr(df_obj, "columns"):
            data = {}
            for col in df_obj.columns:
                try:
                    series = df_obj[col]
                    if hasattr(series, "to_pandas"):
                        data[col] = series.to_pandas()
                    else:
                        data[col] = series
                except Exception:
                    pass
            if data:
                return pd.DataFrame(data)
    except Exception:
        pass
    return None


class RustDataFrameWrapper:
    """
    Wrapper for Rust PyDataFrameElem that adds attribute access to columns.
    """

    def __init__(self, df_obj):
        object.__setattr__(self, "_wrapped", df_obj)
        try:
            object.__setattr__(self, "_columns_cache", list(df_obj.columns))
        except Exception:
            object.__setattr__(self, "_columns_cache", None)

    def __getattr__(self, name):
        wrapped = object.__getattribute__(self, "_wrapped")
        columns_cache = object.__getattribute__(self, "_columns_cache")
        if columns_cache is not None and name in columns_cache:
            try:
                return wrapped[name]
            except Exception:
                pass
        try:
            return getattr(wrapped, name)
        except AttributeError:
            pass
        try:
            return wrapped[name]
        except Exception:
            pass
        raise AttributeError(f"'{type(wrapped).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        return object.__getattribute__(self, "_wrapped")[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, "_wrapped")[key] = value

    def __setattr__(self, name, value):
        if name in ("_wrapped", "_columns_cache"):
            object.__setattr__(self, name, value)
        else:
            wrapped = object.__getattribute__(self, "_wrapped")
            try:
                wrapped[name] = value
            except Exception:
                setattr(wrapped, name, value)

    def __dir__(self):
        wrapped = object.__getattribute__(self, "_wrapped")
        columns_cache = object.__getattribute__(self, "_columns_cache")
        attrs = set(dir(wrapped))
        if columns_cache is not None:
            attrs.update(columns_cache)
        return sorted(attrs)

    def __repr__(self):
        return repr(object.__getattribute__(self, "_wrapped"))

    def __str__(self):
        return str(object.__getattribute__(self, "_wrapped"))

    @property
    def columns(self):
        cache = object.__getattribute__(self, "_columns_cache")
        if cache is not None:
            return cache
        return object.__getattribute__(self, "_wrapped").columns

    @property
    def index(self):
        return object.__getattribute__(self, "_wrapped").index

    @property
    def shape(self):
        return object.__getattribute__(self, "_wrapped").shape

    def head(self, *args, **kwargs):
        return object.__getattribute__(self, "_wrapped").head(*args, **kwargs)

    def info(self, *args, **kwargs):
        return object.__getattribute__(self, "_wrapped").info(*args, **kwargs)

    def to_pandas(self, *args, **kwargs):
        return object.__getattribute__(self, "_wrapped").to_pandas(*args, **kwargs)

    def keys(self):
        return self.columns

    def __contains__(self, key):
        columns_cache = object.__getattribute__(self, "_columns_cache")
        if columns_cache is not None:
            return key in columns_cache
        try:
            return key in object.__getattribute__(self, "_wrapped").columns
        except Exception:
            return False

    def __len__(self):
        return object.__getattribute__(self, "_wrapped").shape[0]

    def __iter__(self):
        return iter(self.columns)

    def get_columns(self):
        wrapped = object.__getattribute__(self, "_wrapped")
        try:
            if hasattr(wrapped, "get_columns"):
                return wrapped.get_columns()
        except Exception:
            pass
        try:
            if hasattr(wrapped, "columns"):
                return list(wrapped.columns)
        except Exception:
            pass
        try:
            df_slice = wrapped[:]
            if hasattr(df_slice, "columns"):
                return list(df_slice.columns)
        except Exception:
            pass
        return []


def _sync_obs_index_to_names(adata_obj):
    """Best-effort: make obs/var index show obs_names/var_names for display."""
    # Skip patching if it's a rust backend to avoid accessing empty slots
    if _is_rust_backend(adata_obj):
        return adata_obj

    def _safe_get_names(obj, attr):
        # Avoid accessing rust backend attributes that might be empty slots
        if _is_rust_backend(obj):
            return None
        try:
            names = getattr(obj, attr, None)
        except Exception:
            return None
        if names is None:
            return None
        try:
            if hasattr(names, "to_list"):
                names_list = names.to_list()
            else:
                names_list = list(names)
        except Exception:
            return None
        if len(names_list) == 0:
            return None
        return names_list

    try:
        names_list = _safe_get_names(adata_obj, "obs_names")
        if names_list is None:
            return adata_obj

        import pandas as pd
        try:
            obs_obj = getattr(adata_obj, "obs", None)
        except Exception:
            obs_obj = None
        if obs_obj is not None and hasattr(obs_obj, "index") and obs_obj.__class__.__module__.startswith("pandas"):
            try:
                obs_obj.index = pd.Index(names_list, name="obs_names")
                return adata_obj
            except Exception:
                pass

        if obs_obj is None:
            return adata_obj

        def _smart_to_pandas(self):
            import pandas as pd
            try:
                import polars as pl
                try:
                    df_slice = self[:]
                    if hasattr(df_slice, "to_pandas"):
                        pdf = df_slice.to_pandas()
                    elif isinstance(df_slice, pl.DataFrame):
                        pdf = df_slice.to_pandas()
                    else:
                        pdf = df_slice
                except Exception:
                    data = {}
                    if hasattr(self, "columns"):
                        for col in self.columns:
                            try:
                                series = self[col]
                                if hasattr(series, "to_pandas"):
                                    data[col] = series.to_pandas()
                                elif isinstance(series, pl.Series):
                                    data[col] = series.to_pandas()
                                else:
                                    data[col] = series
                            except Exception:
                                pass
                    pdf = pd.DataFrame(data)

                try:
                    adata_shape = getattr(adata_obj, "shape", (0, 0))
                    df_rows = len(pdf)
                    if df_rows == adata_shape[0]:
                        current_names = getattr(adata_obj, "obs_names", None)
                        name_type = "obs_names"
                    elif df_rows == adata_shape[1]:
                        current_names = getattr(adata_obj, "var_names", None)
                        name_type = "var_names"
                    else:
                        current_names = None
                        name_type = "unknown"

                    if current_names is not None:
                        names_list_local = current_names.to_list() if hasattr(current_names, "to_list") else list(current_names)
                        if len(names_list_local) == len(pdf):
                            pdf.index = pd.Index(names_list_local, name=name_type)
                except Exception:
                    pass

                return pdf
            except Exception:
                return pd.DataFrame()

        def _smart_head(self, n=5):
            return _smart_to_pandas(self).head(n)

        try:
            from types import MethodType
            try:
                setattr(obs_obj, "to_pandas", MethodType(_smart_to_pandas, obs_obj))
                setattr(obs_obj, "head", MethodType(_smart_head, obs_obj))
                _dbg(f"   {Colors.GREEN}OK Set smart obs methods on instance{Colors.ENDC}")
            except Exception as e:
                _dbg(f"   {Colors.WARNING}WARN  Instance-level setting failed: {e}{Colors.ENDC}")
                obs_cls = obs_obj.__class__
                setattr(obs_cls, "to_pandas", _smart_to_pandas)
                setattr(obs_cls, "head", _smart_head)
                _dbg(f"   {Colors.GREEN}OK Set smart obs methods on class{Colors.ENDC}")

            try:
                import pandas as pd

                def _obs_index_property(self):
                    try:
                        df_slice = self[:]
                        if hasattr(df_slice, "index"):
                            return df_slice.index
                        if hasattr(df_slice, "to_pandas"):
                            return df_slice.to_pandas().index
                    except Exception:
                        pass
                    return pd.Index([], name="obs_names")

                def _obs_columns_property(self):
                    try:
                        df_slice = self[:]
                        if hasattr(df_slice, "columns"):
                            return df_slice.columns
                        if hasattr(df_slice, "to_pandas"):
                            return df_slice.to_pandas().columns
                    except Exception:
                        pass
                    return pd.Index([], name="columns")

                def _obs_shape_property(self):
                    try:
                        df_slice = self[:]
                        if hasattr(df_slice, "shape"):
                            return df_slice.shape
                        if hasattr(df_slice, "to_pandas"):
                            pdf = df_slice.to_pandas()
                            return pdf.shape
                    except Exception:
                        pass
                    return (0, 0)

                def _obs_info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None):
                    try:
                        pdf = _smart_to_pandas(self)
                        return pdf.info(
                            verbose=verbose,
                            buf=buf,
                            max_cols=max_cols,
                            memory_usage=memory_usage,
                            show_counts=show_counts,
                        )
                    except Exception:
                        print(f"<class 'PyDataFrameElem'>\nShape: {_obs_shape_property(self)}")

                obs_cls = obs_obj.__class__
                if not hasattr(obs_cls, "_ov_obs_enhanced"):
                    obs_cls.index = property(_obs_index_property)
                    obs_cls.columns = property(_obs_columns_property)
                    obs_cls.shape = property(_obs_shape_property)
                    obs_cls.info = _obs_info
                    obs_cls._ov_obs_enhanced = True
                    _dbg(f"   {Colors.GREEN}OK Set obs properties on class level{Colors.ENDC}")
            except Exception as e:
                _dbg(f"   {Colors.WARNING}WARN  Failed to set obs properties: {e}{Colors.ENDC}")
        except Exception:
            pass
    except Exception:
        pass

    try:
        vnames_list = _safe_get_names(adata_obj, "var_names")
        if vnames_list is None:
            return adata_obj

        import pandas as pd
        try:
            var_obj = getattr(adata_obj, "var", None)
        except Exception:
            var_obj = None
        if var_obj is not None and hasattr(var_obj, "index") and var_obj.__class__.__module__.startswith("pandas"):
            try:
                var_obj.index = pd.Index(vnames_list, name="var_names")
                return adata_obj
            except Exception:
                pass

        if var_obj is None:
            return adata_obj

        try:
            from types import MethodType
            try:
                setattr(var_obj, "to_pandas", MethodType(_smart_to_pandas, var_obj))
                setattr(var_obj, "head", MethodType(_smart_head, var_obj))
                _dbg(f"   {Colors.GREEN}OK Set smart var methods on instance{Colors.ENDC}")
            except Exception as e:
                _dbg(f"   {Colors.WARNING}WARN  Var instance-level setting failed: {e}{Colors.ENDC}")
                _dbg(f"   {Colors.BLUE}INFO  Using class-level smart methods for var{Colors.ENDC}")

            try:
                import pandas as pd

                def _var_index_property(self):
                    try:
                        df_slice = self[:]
                        if hasattr(df_slice, "index"):
                            return df_slice.index
                        if hasattr(df_slice, "to_pandas"):
                            return df_slice.to_pandas().index
                    except Exception:
                        pass
                    return pd.Index([], name="var_names")

                def _var_columns_property(self):
                    try:
                        df_slice = self[:]
                        if hasattr(df_slice, "columns"):
                            return df_slice.columns
                        if hasattr(df_slice, "to_pandas"):
                            return df_slice.to_pandas().columns
                    except Exception:
                        pass
                    return pd.Index([], name="columns")

                def _var_shape_property(self):
                    try:
                        df_slice = self[:]
                        if hasattr(df_slice, "shape"):
                            return df_slice.shape
                        if hasattr(df_slice, "to_pandas"):
                            pdf = df_slice.to_pandas()
                            return pdf.shape
                    except Exception:
                        pass
                    return (0, 0)

                def _var_info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None):
                    try:
                        pdf = _smart_to_pandas(self)
                        return pdf.info(
                            verbose=verbose,
                            buf=buf,
                            max_cols=max_cols,
                            memory_usage=memory_usage,
                            show_counts=show_counts,
                        )
                    except Exception:
                        print(f"<class 'PyDataFrameElem'>\nShape: {_var_shape_property(self)}")

                var_cls = var_obj.__class__
                if not hasattr(var_cls, "_ov_var_enhanced"):
                    if not hasattr(var_cls, "_ov_obs_enhanced"):
                        var_cls.index = property(_var_index_property)
                        var_cls.columns = property(_var_columns_property)
                        var_cls.shape = property(_var_shape_property)
                        var_cls.info = _var_info
                    var_cls._ov_var_enhanced = True
                    _dbg(f"   {Colors.GREEN}OK Set var properties on class level{Colors.ENDC}")
            except Exception as e:
                _dbg(f"   {Colors.WARNING}WARN  Failed to set var properties: {e}{Colors.ENDC}")
        except Exception:
            pass
    except Exception:
        pass

    return adata_obj


# ------------------ Patch helpers ------------------

def _patch_dataframe_access(adata, verbose: bool = False) -> None:
    for name in ("var", "obs"):
        df_obj = getattr(adata, name, None)
        if df_obj is None:
            continue
        df_cls = type(df_obj)
        if "pandas" in df_cls.__module__.lower():
            continue

        is_pydataframe = df_cls.__module__ == "builtins" or "PyDataFrameElem" in df_cls.__name__
        if not is_pydataframe:
            continue

        try:
            wrapped = RustDataFrameWrapper(df_obj)
            setattr(adata, name, wrapped)
            if verbose:
                print(f"   {Colors.GREEN}OK Wrapped {name} with RustDataFrameWrapper{Colors.ENDC}")
        except Exception as e:
            if verbose:
                print(f"   {Colors.WARNING}WARN  Failed to wrap {name}: {e}{Colors.ENDC}")

            if getattr(df_cls, "_ov_attr_patched", False):
                continue

            def _pydataframe_getattr(self, attr):
                try:
                    return self[attr]
                except Exception:
                    pass
                try:
                    get_column = object.__getattribute__(self, "get_column")
                except Exception:
                    get_column = None
                if get_column is not None:
                    try:
                        return get_column(attr)
                    except Exception:
                        pass
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

            def _pydataframe_get_columns(self):
                try:
                    if hasattr(self, "columns"):
                        return list(self.columns)
                except Exception:
                    pass
                try:
                    df_slice = self[:]
                    if hasattr(df_slice, "columns"):
                        return list(df_slice.columns)
                except Exception:
                    pass
                return []

            try:
                if not hasattr(df_cls, "__getattr__"):
                    df_cls.__getattr__ = _pydataframe_getattr
                if not hasattr(df_cls, "get_columns"):
                    df_cls.get_columns = _pydataframe_get_columns
                df_cls._ov_attr_patched = True
                if verbose:
                    print(f"   {Colors.GREEN}OK Patched {df_cls.__name__} for column attribute access{Colors.ENDC}")
            except Exception as inner_e:
                if verbose:
                    print(f"   {Colors.WARNING}WARN  Failed to patch {df_cls.__name__}: {inner_e}{Colors.ENDC}")


def _patch_boolean_indexing(adata, verbose: bool = False) -> None:
    cls = type(adata)
    if getattr(cls, "_ov_getitem_patched", False):
        return

    original_getitem = cls.__getitem__ if hasattr(cls, "__getitem__") else None
    if original_getitem is None:
        try:
            original_getitem = getattr(adata, "__getitem__", None)
        except Exception:
            original_getitem = None

    def _rust_compatible_getitem(self, index):
        def _call_original(idx):
            if original_getitem is not None:
                return original_getitem(self, idx)
            try:
                inst_getitem = object.__getattribute__(self, "__getitem__")
                func = getattr(inst_getitem, "__func__", None)
                if func is not _rust_compatible_getitem:
                    return inst_getitem(idx)
            except Exception:
                pass
            raise NotImplementedError("__getitem__ not implemented")

        if isinstance(index, tuple) and len(index) == 2:
            obs_idx, var_idx = index
            if _is_rust_backend(self):
                obs_idx = _normalize_bool_index(obs_idx)
                var_idx = _normalize_bool_index(var_idx)
                index = (obs_idx, var_idx)
            try:
                return _call_original(index)
            except NotImplementedError:
                if _is_rust_backend(self):
                    return _slice_adata_rust(self, obs_idx, var_idx)
                raise

        if _is_rust_backend(self):
            index = _normalize_bool_index(index)
        try:
            return _call_original(index)
        except NotImplementedError:
            if _is_rust_backend(self):
                return _slice_adata_rust(self, index, slice(None))
            raise

    def _rust_compatible_setitem(self, index, value):
        def _call_original_set(idx, val):
            try:
                inst_setitem = object.__getattribute__(self, "__setitem__")
                func = getattr(inst_setitem, "__func__", None)
                if func is not _rust_compatible_setitem:
                    inst_setitem(idx, val)
                    return True
            except Exception:
                pass
            return False

        if isinstance(index, tuple) and len(index) == 2:
            obs_idx, var_idx = index
            if _is_rust_backend(self):
                obs_idx = _normalize_bool_index(obs_idx)
                var_idx = _normalize_bool_index(var_idx)
                index = (obs_idx, var_idx)
        elif _is_rust_backend(self):
            index = _normalize_bool_index(index)

        if _call_original_set(index, value):
            return

        if not _is_rust_backend(self):
            raise NotImplementedError("__setitem__ not implemented")

        try:
            x = object.__getattribute__(self, "X")
            x[index] = value
            return
        except Exception:
            pass

        try:
            x = object.__getattribute__(self, "X")[:]
            x[index] = value
            setattr(self, "X", x)
            return
        except Exception:
            pass

        raise NotImplementedError("Rust backend __setitem__ not implemented")

    def _coerce_bool_array(idx):
        if hasattr(idx, "to_numpy"):
            try:
                return np.asarray(idx.to_numpy())
            except Exception:
                pass
        if hasattr(idx, "to_ndarray"):
            try:
                return np.asarray(idx.to_ndarray())
            except Exception:
                pass
        if hasattr(idx, "to_list"):
            try:
                return np.asarray(idx.to_list())
            except Exception:
                pass
        return np.asarray(idx)

    def _normalize_bool_index(idx):
        if isinstance(idx, str) or isinstance(idx, slice):
            return idx
        if hasattr(idx, "dtype") and getattr(idx, "dtype", None) == bool:
            return np.where(idx)[0]
        try:
            arr = _coerce_bool_array(idx)
            if arr.dtype == bool:
                return np.where(arr)[0]
        except Exception:
            pass
        return idx

    try:
        cls.__getitem__ = _rust_compatible_getitem
        cls._ov_getitem_patched = True
        if verbose:
            print(f"   {Colors.GREEN}OK Patched __getitem__ for Rust boolean indexing{Colors.ENDC}")
    except Exception as e:
        if verbose:
            print(f"   {Colors.WARNING}WARN  Failed to patch __getitem__: {e}{Colors.ENDC}")

    try:
        if not hasattr(cls, "__setitem__"):
            cls.__setitem__ = _rust_compatible_setitem
            if verbose:
                print(f"   {Colors.GREEN}OK Patched __setitem__ for Rust assignment{Colors.ENDC}")
    except Exception as e:
        if verbose:
            print(f"   {Colors.WARNING}WARN  Failed to patch __setitem__: {e}{Colors.ENDC}")


def _slice_adata_rust(adata, obs_idx, var_idx):
    def _index_to_int_list(idx, n):
        if isinstance(idx, slice):
            if idx == slice(None):
                return None
            start, stop, step = idx.indices(n)
            return np.arange(start, stop, step)
        if idx is None:
            return None
        if isinstance(idx, (list, tuple, np.ndarray)):
            return np.asarray(idx)
        return idx

    n_obs = getattr(adata, "n_obs", None)
    n_vars = getattr(adata, "n_vars", None)
    obs_idx_norm = _index_to_int_list(obs_idx, n_obs) if n_obs is not None else obs_idx
    var_idx_norm = _index_to_int_list(var_idx, n_vars) if n_vars is not None else var_idx

    if hasattr(adata, "subset"):
        try:
            target = adata.copy() if hasattr(adata, "copy") else adata
            res = target.subset(obs_idx_norm, var_idx_norm)
            return target if res is None else res
        except Exception:
            pass

    if hasattr(adata, "subset_obs") or hasattr(adata, "subset_var"):
        try:
            target = adata.copy() if hasattr(adata, "copy") else adata
            if hasattr(target, "subset_obs") and obs_idx_norm is not None:
                target.subset_obs(obs_idx_norm)
            if hasattr(target, "subset_var") and var_idx_norm is not None:
                target.subset_var(var_idx_norm)
            return target
        except Exception:
            pass

    raise NotImplementedError("Rust backend subset API not available")


# ------------------ Public patch APIs ------------------

def _patch_ann_compat(adata):
    """
    Patch anndata-rs AnnData class/instance with missing AnnData APIs.
    """
    try:
        import pandas as pd
        from pandas.api.types import infer_dtype
    except Exception:
        pd, infer_dtype = None, None
    try:
        import polars as pl
    except Exception:
        pl = None
    try:
        from natsort import natsorted
    except Exception:
        natsorted = sorted

    cls = type(adata)

    def _try_set_class_attr(name, value):
        try:
            if not hasattr(cls, name):
                setattr(cls, name, value)
            return True
        except Exception:
            return False

    _try_set_class_attr("is_view", property(lambda self: False))

    def _obsm_keys(self):
        o = getattr(self, "obsm", None)
        if o is None:
            return []
        return list(o.keys()) if hasattr(o, "keys") else list(o)

    def _varm_keys(self):
        o = getattr(self, "varm", None)
        if o is None:
            return []
        return list(o.keys()) if hasattr(o, "keys") else list(o)

    if not _try_set_class_attr("obsm_keys", _obsm_keys):
        if not hasattr(adata, "obsm_keys"):
            adata.obsm_keys = MethodType(_obsm_keys, adata)

    if not _try_set_class_attr("varm_keys", _varm_keys):
        if not hasattr(adata, "varm_keys"):
            adata.varm_keys = MethodType(_varm_keys, adata)

    if not hasattr(adata, "raw"):
        _try_set_class_attr("raw", property(lambda self: None))

    def _strings_to_categoricals(self, df=None):
        dont_modify = False
        if df is None:
            dfs = [self.obs, self.var]
            if getattr(self, "is_view", False) and getattr(self, "isbacked", False):
                dont_modify = True
        else:
            dfs = [df]

        for _df in dfs:
            mod = type(_df).__module__
            if pd is not None and mod.startswith("pandas"):
                string_cols = [
                    k for k in _df.columns
                    if (infer_dtype(_df[k]) in ("string", "unicode")) or (_df[k].dtype == "object")
                ]
                for key in string_cols:
                    c = pd.Categorical(_df[key])
                    if len(c.categories) >= len(c):
                        continue
                    cats_sorted = list(natsorted(list(c.categories)))
                    if dont_modify:
                        raise RuntimeError(
                            "Please call `.strings_to_categoricals()` on full AnnData, not on a backed view."
                        )
                    if list(c.categories) != cats_sorted:
                        c = c.reorder_categories(cats_sorted)
                    _df[key] = c
            elif pl is not None and mod.startswith("polars"):
                string_cols = [k for k, dt in _df.schema.items() if dt == pl.Utf8]
                for key in string_cols:
                    s = _df[key]
                    if dont_modify:
                        raise RuntimeError(
                            "Call `.strings_to_categoricals()` on full AnnData, not on a backed view."
                        )
                    if s.n_unique(approx=False) < s.len():
                        _df[key] = s.cast(pl.Categorical)
        return None

    ok1 = _try_set_class_attr("strings_to_categoricals", _strings_to_categoricals)
    ok2 = _try_set_class_attr("_sanitize", _strings_to_categoricals)

    if not ok1 and not hasattr(adata, "strings_to_categoricals"):
        adata.strings_to_categoricals = MethodType(_strings_to_categoricals, adata)
    if not ok2 and not hasattr(adata, "_sanitize"):
        adata._sanitize = adata.strings_to_categoricals

    def _safe_get_index(df):
        if hasattr(df, "index"):
            return df.index
        try:
            if hasattr(df, "index_names") and df.index_names:
                return pd.Index(df.index_names) if pd is not None else df.index_names
            if hasattr(df, "to_pandas"):
                return df.to_pandas().index
        except Exception:
            pass
        return None

    def _safe_set_names(obj, attr_name, new_names: Iterable):
        if hasattr(obj, attr_name):
            setattr(obj, attr_name, new_names)
            return
        if attr_name == "var_names" and hasattr(obj, "var"):
            try:
                if hasattr(obj.var, "index"):
                    obj.var.index = new_names
            except Exception:
                pass
        if attr_name == "obs_names" and hasattr(obj, "obs"):
            try:
                if hasattr(obj.obs, "index"):
                    obj.obs.index = new_names
            except Exception:
                pass

    def make_index_unique(index_or_names, join: str = "-"):
        if not hasattr(index_or_names, "is_unique"):
            index_or_names = pd.Index(index_or_names) if pd is not None else index_or_names
        if hasattr(index_or_names, "is_unique") and index_or_names.is_unique:
            return index_or_names

        from collections import Counter
        import warnings

        values = np.array(list(index_or_names))
        if pd is not None:
            values = pd.Index(values).values.copy()
        indices_dup = pd.Index(values).duplicated(keep="first") if pd is not None else []
        values_dup = values[indices_dup] if len(indices_dup) else []
        values_set = set(values)
        counter = Counter()
        issue_interpretation_warning = False
        example_colliding_values = []
        for i, v in enumerate(values_dup):
            while True:
                counter[v] += 1
                tentative_new_name = str(v) + join + str(counter[v])
                if tentative_new_name not in values_set:
                    values_set.add(tentative_new_name)
                    values_dup[i] = tentative_new_name
                    break
                issue_interpretation_warning = True
                if len(example_colliding_values) < 5:
                    example_colliding_values.append(tentative_new_name)

        if issue_interpretation_warning:
            msg = (
                f"Suffix used ({join}[0-9]+) to deduplicate index values may make index values difficult to interpret. "
                "Consider using a different delimiter by passing `join={delimiter}`. "
                f"Example key collisions generated by the make_index_unique algorithm: {example_colliding_values}"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        if len(indices_dup):
            values[indices_dup] = values_dup
        if pd is not None:
            return pd.Index(values, name=getattr(index_or_names, "name", None))
        return values

    def var_names_make_unique(self, join: str = "-"):
        current_names = getattr(self, "var_names", None)
        if current_names is None:
            var_index = _safe_get_index(self.var)
            if var_index is not None:
                current_names = var_index
            else:
                try:
                    current_names = list(range(self.n_vars))
                except Exception:
                    return
        new_names = make_index_unique(current_names, join)
        _safe_set_names(self, "var_names", new_names)

    def obs_names_make_unique(self, join: str = "-"):
        current_names = getattr(self, "obs_names", None)
        if current_names is None:
            obs_index = _safe_get_index(self.obs)
            if obs_index is not None:
                current_names = obs_index
            else:
                try:
                    current_names = list(range(self.n_obs))
                except Exception:
                    return
        new_names = make_index_unique(current_names, join)
        _safe_set_names(self, "obs_names", new_names)

    ok3 = _try_set_class_attr("var_names_make_unique", var_names_make_unique)
    ok4 = _try_set_class_attr("obs_names_make_unique", obs_names_make_unique)

    if not ok3 and not hasattr(adata, "var_names_make_unique"):
        adata.var_names_make_unique = MethodType(var_names_make_unique, adata)
    if not ok4 and not hasattr(adata, "obs_names_make_unique"):
        adata.obs_names_make_unique = MethodType(obs_names_make_unique, adata)

    _dbg(f"   {Colors.CYAN}NOTE Simplified display patching without global mapping{Colors.ENDC}")


def _series_to_arraylike(s):
    try:
        import pandas as pd
        if s.__class__.__module__.startswith("pandas"):
            return s.values
    except Exception:
        pass
    try:
        import polars as pl
        if isinstance(s, pl.Series):
            if s.dtype == pl.Categorical:
                import pandas as pd
                return pd.Categorical(s.to_list())
            return np.asarray(s.to_list())
    except Exception:
        pass
    for attr in ("to_numpy", "to_ndarray", "to_list"):
        if hasattr(s, attr):
            return np.asarray(getattr(s, attr)())
    return np.asarray(s)


def _patch_vector_api(adata):
    import warnings

    try:
        import scipy.sparse as sp
        _issparse = sp.issparse
    except Exception:
        _issparse = lambda x: False

    cls = type(adata)

    def _ensure_list_names(names):
        try:
            return names.tolist()
        except Exception:
            return list(names)

    def _find_pos(names, key):
        try:
            return int(names.get_loc(key))
        except Exception:
            seq = _ensure_list_names(names)
            try:
                return seq.index(key)
            except ValueError:
                for i, v in enumerate(seq):
                    if str(v) == str(key):
                        return i
                raise

    def _try_get_column(df, key):
        try:
            s = df[key]
            return True, s
        except Exception:
            pass
        if hasattr(df, "get_column"):
            try:
                return True, df.get_column(key)
            except Exception:
                pass
        if hasattr(df, "select"):
            try:
                tmp = df.select(key)
                if hasattr(tmp, "to_series"):
                    return True, tmp.to_series()
                if hasattr(tmp, "columns") and tmp.columns:
                    return True, tmp[tmp.columns[0]]
            except Exception:
                pass
        return False, None

    def _get_X(self, layer=None):
        if layer is None:
            return self.X
        if layer == "X":
            try:
                if "X" in self.layers:
                    return self.layers["X"]
            except Exception:
                pass
            warnings.warn(
                "In a future AnnData, `layer='X'` will be removed; use layer=None.",
                FutureWarning,
                stacklevel=2,
            )
            return self.X
        return self.layers[layer]

    def _make_slice(self, key, selected_dim):
        if selected_dim == 1:
            j = _find_pos(self.var_names, key)
            return (slice(None), [j])
        i = _find_pos(self.obs_names, key)
        return ([i], slice(None))

    def _normalize_indices(self, idx):
        return idx

    def _get_vector(self, k, coldim, idxdim, layer=None):
        dims = ("obs", "var")
        obj = getattr(self, coldim)
        idx_names = getattr(self, f"{idxdim}_names")

        in_col, col_series = _try_get_column(obj, k)
        in_idx = (k in idx_names)

        if (in_col + in_idx) == 2:
            raise ValueError(f"Key {k} could be found in both .{idxdim}_names and .{coldim}.columns")
        if (in_col + in_idx) == 0:
            raise KeyError(f"Could not find key {k} in .{idxdim}_names or .{coldim}.columns.")
        if in_col:
            return _series_to_arraylike(col_series)
        selected_dim = dims.index(idxdim)
        idx = self._normalize_indices(_make_slice(self, k, selected_dim))
        a = self._get_X(layer=layer)[idx]
        if _issparse(a):
            a = a.toarray()
        return np.ravel(a)

    def _obs_vector(self, k, *, layer=None):
        return _get_vector(self, k, "obs", "var", layer=layer)

    def _var_vector(self, k, *, layer=None):
        return _get_vector(self, k, "var", "obs", layer=layer)

    def _set_cls(name, obj):
        try:
            if not hasattr(cls, name):
                setattr(cls, name, obj)
            return True
        except Exception:
            return False

    if not _set_cls("_get_X", _get_X):
        if not hasattr(adata, "_get_X"):
            adata._get_X = MethodType(_get_X, adata)
    if not _set_cls("_normalize_indices", _normalize_indices):
        if not hasattr(adata, "_normalize_indices"):
            adata._normalize_indices = MethodType(_normalize_indices, adata)
    if not _set_cls("get_vector", _get_vector):
        if not hasattr(adata, "get_vector"):
            adata.get_vector = MethodType(_get_vector, adata)
    if not _set_cls("obs_vector", _obs_vector):
        if not hasattr(adata, "obs_vector"):
            adata.obs_vector = MethodType(_obs_vector, adata)
    if not _set_cls("var_vector", _var_vector):
        if not hasattr(adata, "var_vector"):
            adata.var_vector = MethodType(_var_vector, adata)


def patch_rust_adata(
    adata,
    *,
    apply_vector_api: bool = True,
    sync_index: bool = True,
    patch_dataframe_access: bool = True,
    patch_boolean_getitem: bool = True,
    verbose: bool = False,
):
    """
    Apply a collection of compatibility patches to an anndata-rs AnnData instance.
    """
    _patch_ann_compat(adata)
    if patch_dataframe_access:
        _patch_dataframe_access(adata, verbose=verbose)
    if patch_boolean_getitem:
        _patch_boolean_indexing(adata, verbose=verbose)
    if apply_vector_api:
        _patch_vector_api(adata)
    if sync_index:
        try:
            _sync_obs_index_to_names(adata)
        except Exception:
            pass
    return adata
