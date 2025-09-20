r"""
Pyomic data (Pyomic.utils._data)
"""


import time
import requests
import os
import pandas as pd
import scanpy as sc
from pathlib import Path
from ._genomics import read_gtf,Gtf
import anndata
import numpy as np
from typing import Callable, List, Mapping, Optional,Dict
from ._enum import ModeEnum
from scipy.sparse import diags, issparse, spmatrix, csr_matrix, isspmatrix_csr
import warnings
from scipy.stats import norm
from .._settings import Colors  # Import Colors from settings
from .registry import register_function


# Internal debug logger (opt-in via env OV_DEBUG/OMICVERSE_DEBUG)
def _ov_debug_enabled():
    try:
        val = os.environ.get("OV_DEBUG") or os.environ.get("OMICVERSE_DEBUG")
        if val is None:
            return False
        return str(val).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _dbg(msg):
    try:
        if _ov_debug_enabled():
            print(msg)
    except Exception:
        pass




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
    Arguments:
        path: The path of the file to read
        backend: 'python' | 'rust'
    Returns:
        AnnData-like object
    """
    ext = Path(path).suffix.lower()

    if ext == '.h5ad':
        def _sync_obs_index_to_names(adata_obj):
            """Best‑effort: make obs index show obs_names for display.

            - For pandas obs: set `obs.index = obs_names` directly.
            - For Rust/Polars obs: override display helpers so head(), to_pandas(),
              and .index reflect `obs_names`. Only affects display, not storage.
            """
            try:
                names = getattr(adata_obj, 'obs_names', None)
                if names is None:
                    return adata_obj
                try:
                    # Normalize to list
                    try:
                        names_list = list(names) if not hasattr(names, 'to_list') else names.to_list()
                    except Exception:
                        names_list = list(names)

                    # Case 1: pandas DataFrame
                    import pandas as pd  # noqa: F401
                    if hasattr(adata_obj, 'obs') and hasattr(adata_obj.obs, 'index') and adata_obj.obs.__class__.__module__.startswith('pandas'):
                        try:
                            adata_obj.obs.index = pd.Index(names_list, name='obs_names')
                            return adata_obj
                        except Exception:
                            pass

                    # Case 2: Non-pandas (e.g., anndata-rs PyDataFrameElem)
                    obs_obj = getattr(adata_obj, 'obs', None)
                    if obs_obj is None:
                        return adata_obj

                    # 创建通用的智能显示方法，能自动识别 obs 还是 var
                    def _smart_to_pandas(self):
                        import pandas as pd
                        try:
                            # 获取基础 DataFrame
                            import polars as pl
                            try:
                                df_slice = self[:]
                                if hasattr(df_slice, 'to_pandas'):
                                    pdf = df_slice.to_pandas()
                                elif isinstance(df_slice, pl.DataFrame):
                                    pdf = df_slice.to_pandas()
                                else:
                                    pdf = df_slice
                            except Exception:
                                # 备用方法：从列构建
                                data = {}
                                if hasattr(self, 'columns'):
                                    for col in self.columns:
                                        try:
                                            series = self[col]
                                            if hasattr(series, 'to_pandas'):
                                                data[col] = series.to_pandas()
                                            elif isinstance(series, pl.Series):
                                                data[col] = series.to_pandas()
                                            else:
                                                data[col] = series
                                        except Exception:
                                            pass
                                pdf = pd.DataFrame(data)
                            
                            # 智能识别：通过形状判断是 obs 还是 var
                            try:
                                adata_shape = getattr(adata_obj, 'shape', (0, 0))
                                df_rows = len(pdf)
                                
                                if df_rows == adata_shape[0]:  # 行数匹配 n_obs，是 obs
                                    current_names = getattr(adata_obj, 'obs_names', None)
                                    name_type = 'obs_names'
                                elif df_rows == adata_shape[1]:  # 行数匹配 n_vars，是 var
                                    current_names = getattr(adata_obj, 'var_names', None)
                                    name_type = 'var_names'
                                else:
                                    current_names = None
                                    name_type = 'unknown'
                                
                                if current_names is not None:
                                    if hasattr(current_names, 'to_list'):
                                        names_list = current_names.to_list()
                                    else:
                                        names_list = list(current_names)
                                    
                                    # 设置正确的索引
                                    if len(names_list) == len(pdf):
                                        pdf.index = pd.Index(names_list, name=name_type)
                            except Exception:
                                pass
                            
                            return pdf
                            
                        except Exception:
                            return pd.DataFrame()

                    def _smart_head(self, n=5):
                        return _smart_to_pandas(self).head(n)

                    # Try to set methods on instance level, with fallback to class level
                    try:
                        from types import MethodType
                        # Use smart methods for both obs and var (same methods work for both)
                        try:
                            setattr(obs_obj, 'to_pandas', MethodType(_smart_to_pandas, obs_obj))
                            setattr(obs_obj, 'head', MethodType(_smart_head, obs_obj))
                            _dbg(f"   {Colors.GREEN}✅ Set smart obs methods on instance{Colors.ENDC}")
                        except Exception as e:
                            _dbg(f"   {Colors.WARNING}⚠️  Instance-level setting failed: {e}{Colors.ENDC}")
                            # Fallback to class-level but with smart methods
                            obs_cls = obs_obj.__class__
                            setattr(obs_cls, 'to_pandas', _smart_to_pandas)
                            setattr(obs_cls, 'head', _smart_head)
                            _dbg(f"   {Colors.GREEN}✅ Set smart obs methods on class{Colors.ENDC}")

                        # Add pandas-like properties specifically for obs
                        try:
                            import pandas as pd
                            
                            # Index property for obs - always return obs_names
                            def _obs_index_property(self):
                                current_obs_names = getattr(adata_obj, 'obs_names', None)
                                if current_obs_names is not None:
                                    if hasattr(current_obs_names, 'to_list'):
                                        obs_names_list = current_obs_names.to_list()
                                    else:
                                        obs_names_list = list(current_obs_names)
                                    return pd.Index(obs_names_list, name='obs_names')
                                return pd.Index([], name='obs_names')
                            
                            # Columns property for obs
                            def _obs_columns_property(self):
                                try:
                                    df_slice = self[:]
                                    if hasattr(df_slice, 'columns'):
                                        return df_slice.columns
                                    elif hasattr(df_slice, 'to_pandas'):
                                        return df_slice.to_pandas().columns
                                except Exception:
                                    pass
                                return pd.Index([], name='columns')
                            
                            # Shape property for obs
                            def _obs_shape_property(self):
                                try:
                                    obs_len = len(getattr(adata_obj, 'obs_names', []))
                                    df_slice = self[:]
                                    if hasattr(df_slice, 'shape'):
                                        return (obs_len, df_slice.shape[1])
                                    elif hasattr(df_slice, 'to_pandas'):
                                        pdf = df_slice.to_pandas()
                                        return (obs_len, len(pdf.columns))
                                except Exception:
                                    pass
                                return (0, 0)
                            
                            # Info method for obs
                            def _obs_info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None):
                                try:
                                    pdf = _smart_to_pandas(self)
                                    return pdf.info(verbose=verbose, buf=buf, max_cols=max_cols, 
                                                  memory_usage=memory_usage, show_counts=show_counts)
                                except Exception:
                                    print(f"<class 'PyDataFrameElem'>\nShape: {_obs_shape_property(self)}")
                            
                            # Use class-level properties (since instance-level setting fails)
                            try:
                                obs_cls = obs_obj.__class__
                                if not hasattr(obs_cls, '_ov_obs_enhanced'):
                                    # Set all properties on the class level
                                    obs_cls.index = property(_obs_index_property)
                                    obs_cls.columns = property(_obs_columns_property)
                                    obs_cls.shape = property(_obs_shape_property)
                                    obs_cls.info = _obs_info
                                    obs_cls._ov_obs_enhanced = True
                                    _dbg(f"   {Colors.GREEN}✅ Set obs properties on class level{Colors.ENDC}")
                            except Exception as e:
                                _dbg(f"   {Colors.WARNING}⚠️  Class-level setting failed: {e}{Colors.ENDC}")
                                # Final fallback: set as instance attributes
                                try:
                                    setattr(obs_obj, 'index', _obs_index_property(obs_obj))
                                    setattr(obs_obj, 'columns', _obs_columns_property(obs_obj))
                                    setattr(obs_obj, 'shape', _obs_shape_property(obs_obj))
                                    setattr(obs_obj, 'info', MethodType(_obs_info, obs_obj))
                                except Exception:
                                    pass
                                
                        except Exception as e:
                            _dbg(f"   {Colors.WARNING}⚠️  Failed to set obs properties: {e}{Colors.ENDC}")
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
            
            # Also sync var.index with var_names for display
            try:
                vnames = getattr(adata_obj, 'var_names', None)
                if vnames is None:
                    return adata_obj
                try:
                    try:
                        vnames_list = list(vnames) if not hasattr(vnames, 'to_list') else vnames.to_list()
                    except Exception:
                        vnames_list = list(vnames)

                    import pandas as pd  # noqa: F401
                    if hasattr(adata_obj, 'var') and hasattr(adata_obj.var, 'index') and adata_obj.var.__class__.__module__.startswith('pandas'):
                        try:
                            adata_obj.var.index = pd.Index(vnames_list, name='var_names')
                            return adata_obj
                        except Exception:
                            pass

                    var_obj = getattr(adata_obj, 'var', None)
                    if var_obj is None:
                        return adata_obj

                    # Use the same smart methods for var as well
                    try:
                        from types import MethodType
                        # Use smart methods that auto-detect obs vs var
                        try:
                            setattr(var_obj, 'to_pandas', MethodType(_smart_to_pandas, var_obj))
                            setattr(var_obj, 'head', MethodType(_smart_head, var_obj))
                            _dbg(f"   {Colors.GREEN}✅ Set smart var methods on instance{Colors.ENDC}")
                        except Exception as e:
                            _dbg(f"   {Colors.WARNING}⚠️  Var instance-level setting failed: {e}{Colors.ENDC}")
                            # Fallback to class-level (should already be set from obs processing)
                            _dbg(f"   {Colors.BLUE}ℹ️  Using class-level smart methods for var{Colors.ENDC}")

                        # Add pandas-like properties specifically for var
                        try:
                            import pandas as pd
                            
                            # Index property for var - always return var_names
                            def _var_index_property(self):
                                current_var_names = getattr(adata_obj, 'var_names', None)
                                if current_var_names is not None:
                                    if hasattr(current_var_names, 'to_list'):
                                        var_names_list = current_var_names.to_list()
                                    else:
                                        var_names_list = list(current_var_names)
                                    return pd.Index(var_names_list, name='var_names')
                                return pd.Index([], name='var_names')
                            
                            # Columns property for var
                            def _var_columns_property(self):
                                try:
                                    df_slice = self[:]
                                    if hasattr(df_slice, 'columns'):
                                        return df_slice.columns
                                    elif hasattr(df_slice, 'to_pandas'):
                                        return df_slice.to_pandas().columns
                                except Exception:
                                    pass
                                return pd.Index([], name='columns')
                            
                            # Shape property for var
                            def _var_shape_property(self):
                                try:
                                    var_len = len(getattr(adata_obj, 'var_names', []))
                                    df_slice = self[:]
                                    if hasattr(df_slice, 'shape'):
                                        return (var_len, df_slice.shape[1])
                                    elif hasattr(df_slice, 'to_pandas'):
                                        pdf = df_slice.to_pandas()
                                        return (var_len, len(pdf.columns))
                                except Exception:
                                    pass
                                return (0, 0)
                            
                            # Info method for var
                            def _var_info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None):
                                try:
                                    pdf = _smart_to_pandas(self)
                                    return pdf.info(verbose=verbose, buf=buf, max_cols=max_cols, 
                                                  memory_usage=memory_usage, show_counts=show_counts)
                                except Exception:
                                    print(f"<class 'PyDataFrameElem'>\nShape: {_var_shape_property(self)}")
                            
                            # Use class-level properties (since instance-level setting fails)
                            # But we need to be careful since obs and var share the same class
                            try:
                                var_cls = var_obj.__class__
                                # Only set var-specific properties if obs hasn't already set them
                                if not hasattr(var_cls, '_ov_var_enhanced'):
                                    # Check if obs properties are already there
                                    if hasattr(var_cls, '_ov_obs_enhanced'):
                                        # obs already set class properties, but they're obs-specific
                                        # We need var-specific versions. Use a different approach.
                                        # Let's override with var-specific methods only if needed
                                        pass
                                    else:
                                        # Set var properties on class level
                                        var_cls.index = property(_var_index_property)
                                        var_cls.columns = property(_var_columns_property)
                                        var_cls.shape = property(_var_shape_property)
                                        var_cls.info = _var_info
                                    var_cls._ov_var_enhanced = True
                                    _dbg(f"   {Colors.GREEN}✅ Set var properties on class level{Colors.ENDC}")
                            except Exception as e:
                                _dbg(f"   {Colors.WARNING}⚠️  Var class-level setting failed: {e}{Colors.ENDC}")
                                # Final fallback: set as instance attributes
                                try:
                                    setattr(var_obj, 'index', _var_index_property(var_obj))
                                    setattr(var_obj, 'columns', _var_columns_property(var_obj))
                                    setattr(var_obj, 'shape', _var_shape_property(var_obj))
                                    setattr(var_obj, 'info', MethodType(_var_info, var_obj))
                                except Exception:
                                    pass
                                
                        except Exception as e:
                            _dbg(f"   {Colors.WARNING}⚠️  Failed to set var properties: {e}{Colors.ENDC}")
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
            return adata_obj

        if backend == 'python':
            adata = sc.read_h5ad(path, **kwargs)
            # Ensure pandas obs index matches obs_names
            
            return adata

        elif backend == 'rust':
            try:
                import snapatac2 as snap
            except ImportError:
                raise ImportError('snapatac2 is not installed. `pip install snapatac2`')

            print(f'{Colors.GREEN}Using anndata-rs to read h5ad file{Colors.ENDC}')
            print(f'{Colors.WARNING}You should run adata.close() after analysis{Colors.ENDC}')
            print(f'{Colors.WARNING}Not all function support Rust backend{Colors.ENDC}')
            adata = snap.read(path, **kwargs)
            _patch_ann_compat(adata)  # 兼容补丁
            _patch_vector_api(adata)
            # Force display alignment: obs.index mirrors obs_names
            try:
                _sync_obs_index_to_names(adata)
            except Exception:
                pass
            return adata

        else:
            raise ValueError("backend must be 'python' or 'rust'")

    # 其它纯表格：pandas 会自动识别 gz 压缩，不必手动区分
    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')


# ------------------ 兼容补丁 ------------------

# 简化的 DataFrame 显示补丁，不再需要全局映射

def _patch_ann_compat(adata):
    """
    给 anndata-rs(Rust) AnnData 类/实例补齐：
    - is_view (property, False)
    - obsm_keys()/varm_keys()（返回键列表）
    - raw (property, None)  —— 只在确实缺失时补
    - strings_to_categoricals(df=None) / _sanitize() —— 兼容 pandas/Polars
    - obs/var 的表格显示补丁
    """
    import numpy as np
    from types import MethodType
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

    # 1) is_view → property(False)
    _try_set_class_attr("is_view", property(lambda self: False))

    # 2) obsm_keys / varm_keys
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

    # 3) raw → property(None)（只有缺失时才补；避免覆盖已有实现）
    if not hasattr(adata, "raw"):
        _try_set_class_attr("raw", property(lambda self: None))
        # 类不可写就算了，访问 getattr(adata,"raw",None) 做兜底

    # 4) strings_to_categoricals / _sanitize（pandas/Polars 兼容）
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
            # pandas DataFrame
            if pd is not None and mod.startswith("pandas"):
                string_cols = [
                    k for k in _df.columns
                    if (infer_dtype(_df[k]) in ("string", "unicode")) or (_df[k].dtype == "object")
                ]
                for key in string_cols:
                    c = pd.Categorical(_df[key])
                    # 仅当类别数 < 行数时才转分类（与 anndata 行为一致）
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

            # polars DataFrame
            elif pl is not None and mod.startswith("polars"):
                string_cols = [k for k, dt in _df.schema.items() if dt == pl.Utf8]
                for key in string_cols:
                    s = _df[key]
                    # 与 pandas 近似：当存在重复（n_unique < n_rows）才转分类
                    try:
                        n_null = s.is_null().sum()
                    except Exception:
                        n_null = s.null_count()
                    if dont_modify:
                        raise RuntimeError(
                            "Call `.strings_to_categoricals()` on full AnnData, not on a backed view."
                        )
                    if s.n_unique(approx=False) < s.len():
                        _df[key] = s.cast(pl.Categorical)

            # 其它 DataFrame 实现：跳过
        return None

    # 尝试挂到类
    ok1 = _try_set_class_attr("strings_to_categoricals", _strings_to_categoricals)
    ok2 = _try_set_class_attr("_sanitize", _strings_to_categoricals)

    # 类不可写则绑定到实例
    if not ok1 and not hasattr(adata, "strings_to_categoricals"):
        adata.strings_to_categoricals = MethodType(_strings_to_categoricals, adata)
    if not ok2 and not hasattr(adata, "_sanitize"):
        # 与 anndata 旧 API 对齐
        adata._sanitize = adata.strings_to_categoricals

    # 5) var_names_make_unique / obs_names_make_unique 方法补丁
    def _safe_get_index(df):
        """Safely get index from both pandas DataFrame and Rust PyDataFrameElem."""
        if hasattr(df, 'index'):
            # pandas DataFrame
            return df.index
        else:
            # Rust PyDataFrameElem - try different methods to get the index
            try:
                # Method 1: Try to get index names directly
                if hasattr(df, 'index_names') and df.index_names:
                    return pd.Index(df.index_names)
                # Method 2: Try to get the first column if it's the index
                if hasattr(df, 'columns') and hasattr(df, '__getitem__'):
                    # This is a fallback - we'll use var_names/obs_names from the parent object
                    return None
                # Method 3: Convert to pandas and get index
                if hasattr(df, 'to_pandas'):
                    return df.to_pandas().index
            except Exception:
                pass
            return None

    def _safe_set_names(adata, attr_name, new_names):
        """Safely set var_names or obs_names for both pandas and Rust backends."""
        if hasattr(adata, attr_name):
            setattr(adata, attr_name, new_names)
        else:
            # Fallback for Rust backends
            if attr_name == 'var_names' and hasattr(adata, 'var'):
                # Try to set the index of the var dataframe
                try:
                    if hasattr(adata.var, 'index'):
                        adata.var.index = new_names
                except Exception:
                    pass
            elif attr_name == 'obs_names' and hasattr(adata, 'obs'):
                # Try to set the index of the obs dataframe
                try:
                    if hasattr(adata.obs, 'index'):
                        adata.obs.index = new_names
                except Exception:
                    pass

    def make_index_unique(index_or_names, join: str = "-"):
        """
        Makes the index unique by appending a number string to each duplicate index element:
        '1', '2', etc.

        If a tentative name created by the algorithm already exists in the index, it tries
        the next integer in the sequence.

        The first occurrence of a non-unique value is ignored.

        Parameters
        ----------
        index_or_names
             A pandas Index object or array-like of names
        join
             The connecting string between name and integer.
        """
        # Convert to pandas Index if needed
        if not hasattr(index_or_names, 'is_unique'):
            index_or_names = pd.Index(index_or_names)
            
        if index_or_names.is_unique:
            return index_or_names
            
        from collections import Counter
        import warnings

        values = index_or_names.values.copy()
        indices_dup = index_or_names.duplicated(keep="first")
        values_dup = values[indices_dup]
        values_set = set(values)
        counter = Counter()
        issue_interpretation_warning = False
        example_colliding_values = []
        for i, v in enumerate(values_dup):
            while True:
                counter[v] += 1
                tentative_new_name = v + join + str(counter[v])
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
                "There values with a similar suffixes in the index. "
                "Consider using a different delimiter by passing `join={delimiter}`. "
                "Example key collisions generated by the make_index_unique algorithm: "
                f"{example_colliding_values}"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        values[indices_dup] = values_dup
        index = pd.Index(values, name=getattr(index_or_names, 'name', None))
        return index

    def var_names_make_unique(self, join: str = "-"):
        """Make variable names unique by appending numbers to duplicates."""
        # Get current var_names safely
        current_names = getattr(self, 'var_names', None)
        if current_names is None:
            # Try to get from var.index
            var_index = _safe_get_index(self.var)
            if var_index is not None:
                current_names = var_index
            else:
                # Last resort - get from var_names attribute or create default
                try:
                    current_names = list(range(self.n_vars))
                except:
                    return  # Can't proceed without names
        
        # Make unique
        new_names = make_index_unique(current_names, join)
        
        # Set the new names safely
        _safe_set_names(self, 'var_names', new_names)

    def obs_names_make_unique(self, join: str = "-"):
        """Make observation names unique by appending numbers to duplicates."""
        # Get current obs_names safely
        current_names = getattr(self, 'obs_names', None)
        if current_names is None:
            # Try to get from obs.index
            obs_index = _safe_get_index(self.obs)
            if obs_index is not None:
                current_names = obs_index
            else:
                # Last resort - get from obs_names attribute or create default
                try:
                    current_names = list(range(self.n_obs))
                except:
                    return  # Can't proceed without names
        
        # Make unique
        new_names = make_index_unique(current_names, join)
        
        # Set the new names safely
        _safe_set_names(self, 'obs_names', new_names)

    # 尝试挂到类
    ok3 = _try_set_class_attr("var_names_make_unique", var_names_make_unique)
    ok4 = _try_set_class_attr("obs_names_make_unique", obs_names_make_unique)

    # 类不可写则绑定到实例
    if not ok3 and not hasattr(adata, "var_names_make_unique"):
        adata.var_names_make_unique = MethodType(var_names_make_unique, adata)
    if not ok4 and not hasattr(adata, "obs_names_make_unique"):
        adata.obs_names_make_unique = MethodType(obs_names_make_unique, adata)

    # 简化的显示补丁 - 不再需要复杂的全局映射
    _dbg(f"   {Colors.CYAN}💡 Simplified display patching without global mapping{Colors.ENDC}")


def _patch_vector_api(adata):
    import numpy as np, warnings
    from types import MethodType
    try:
        import scipy.sparse as sp
        _issparse = sp.issparse
    except Exception:
        _issparse = lambda x: False

    cls = type(adata)

    # ---------- helpers ----------
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
        """同时兼容 pandas/Polars/代理对象；返回 (found, series/array-like)。"""
        # 1) 最常见：支持 df[key]
        try:
            s = df[key]
            return True, s
        except Exception:
            pass
        # 2) Polars: get_column
        if hasattr(df, "get_column"):
            try:
                return True, df.get_column(key)
            except Exception:
                pass
        # 3) Polars: select 返回 DataFrame，再取第一列
        if hasattr(df, "select"):
            try:
                tmp = df.select(key)
                # polars 0.20+: tmp is DataFrame
                if hasattr(tmp, "to_series"):
                    return True, tmp.to_series()  # type: ignore[attr-defined]
                # 兜底：取第一列
                if hasattr(tmp, "columns") and tmp.columns:
                    return True, tmp[tmp.columns[0]]
            except Exception:
                pass
        return False, None

    def _series_to_np(s):
        for attr in ("to_numpy", "to_ndarray", "to_list"):
            if hasattr(s, attr):
                arr = getattr(s, attr)()
                return np.asarray(arr)
        return np.asarray(s)

    # ---------- _get_X ----------
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
                FutureWarning, stacklevel=2,
            )
            return self.X
        return self.layers[layer]

    # ---------- _make_slice / _normalize_indices ----------
    def _make_slice(self, key, selected_dim):
        if selected_dim == 1:  # along var
            j = _find_pos(self.var_names, key)
            return (slice(None), [j])
        else:                  # along obs
            i = _find_pos(self.obs_names, key)
            return ([i], slice(None))

    def _normalize_indices(self, idx):
        return idx  # 我们生成的已可直接用于 numpy/scipy

    # ---------- get_vector：不再依赖 .columns ----------
    def _get_vector(self, k, coldim, idxdim, layer=None):
        dims = ("obs", "var")
        obj = getattr(self, coldim)
        idx_names = getattr(self, f"{idxdim}_names")

        in_col, col_series = _try_get_column(obj, k)
        in_idx = (k in idx_names)

        if (in_col + in_idx) == 2:
            raise ValueError(f"Key {k} could be found in both .{idxdim}_names and .{coldim}.columns")
        elif (in_col + in_idx) == 0:
            raise KeyError(f"Could not find key {k} in .{idxdim}_names or .{coldim}.columns.")
        elif in_col:
            return _series_to_arraylike(col_series)
        else:
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

    # ---------- bind ----------
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

def _series_to_arraylike(s):
    import numpy as np
    try:
        import pandas as pd
        if s.__class__.__module__.startswith("pandas"):
            return s.values   # 分类列会得到 pandas.Categorical
    except Exception:
        pass
    try:
        import polars as pl, pandas as pd
        if isinstance(s, pl.Series):
            if s.dtype == pl.Categorical:
                return pd.Categorical(s.to_list())
            return np.asarray(s.to_list())
    except Exception:
        pass
    for attr in ("to_numpy", "to_ndarray", "to_list"):
        if hasattr(s, attr):
            return np.asarray(getattr(s, attr)())
    return np.asarray(s)


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
        "# Now you can use pandas methods",
        "filtered = obs_df[obs_df['n_genes'] > 1000]"
    ],
    related=["utils.read", "pp.preprocess", "utils.store_layers"]
)
def convert_to_pandas(df_obj):
    """
    Convert PyDataFrameElem or similar objects to pandas DataFrame.

    This is a utility function to convert Rust-based DataFrame objects
    (like PyDataFrameElem from anndata-rs/SnapATAC2) to pandas DataFrames.

    Arguments:
        df_obj: PyDataFrameElem or similar DataFrame-like object

    Returns:
        pandas.DataFrame: Converted DataFrame

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.read('data.h5ad', backend='rust')
        >>> obs_df = ov.utils.convert_to_pandas(adata.obs)
        >>> print(obs_df)  # Now displays as pandas DataFrame
    """
    import pandas as pd

    try:
        # 如果对象已经有 to_pandas 方法，直接使用
        if hasattr(df_obj, 'to_pandas'):
            return df_obj.to_pandas()
    except Exception:
        pass

    try:
        # 方法1: 使用切片获取整个 DataFrame（SnapATAC2 风格）
        import polars as pl
        df_slice = df_obj[:]

        # 检查返回的是否是 polars DataFrame
        if hasattr(df_slice, 'to_pandas'):
            return df_slice.to_pandas()
        elif isinstance(df_slice, pl.DataFrame):
            return df_slice.to_pandas()
        else:
            # 已经是 pandas DataFrame
            return df_slice
    except Exception:
        pass

    try:
        # 方法2: 通过列名构建 DataFrame
        if hasattr(df_obj, '__getitem__'):
            import polars as pl
            data = {}
            # 尝试获取列名
            if hasattr(df_obj, 'columns'):
                columns = df_obj.columns
            else:
                return pd.DataFrame()

            for col in columns:
                try:
                    series = df_obj[col]
                    # 检查是否是 polars Series
                    if hasattr(series, 'to_pandas'):
                        data[col] = series.to_pandas()
                    elif isinstance(series, pl.Series):
                        data[col] = series.to_pandas()
                    else:
                        data[col] = series
                except:
                    pass

            if data:
                return pd.DataFrame(data)
    except Exception:
        pass

    # 如果都失败了，返回空 DataFrame
    return pd.DataFrame()


class PyDataFrameElemWrapper:
    """
    A wrapper class that provides pandas DataFrame-like interface for PyDataFrameElem.

    This class wraps PyDataFrameElem objects and provides familiar pandas methods
    like head(), tail(), shape, columns, index, etc.
    """

    def __init__(self, df_obj):
        self._df_obj = df_obj
        self._pandas_cache = None

    def _get_pandas(self):
        """Get pandas DataFrame, with caching."""
        if self._pandas_cache is None:
            self._pandas_cache = convert_to_pandas(self._df_obj)
        return self._pandas_cache

    def head(self, n=5):
        """Return first n rows."""
        return self._get_pandas().head(n)

    def tail(self, n=5):
        """Return last n rows."""
        return self._get_pandas().tail(n)

    @property
    def shape(self):
        """Return shape of DataFrame."""
        return self._get_pandas().shape

    @property
    def columns(self):
        """Return column labels."""
        return self._get_pandas().columns

    @property
    def index(self):
        """Return row index."""
        return self._get_pandas().index

    @property
    def dtypes(self):
        """Return data types."""
        return self._get_pandas().dtypes

    def info(self, *args, **kwargs):
        """Print info about DataFrame."""
        return self._get_pandas().info(*args, **kwargs)

    def describe(self, *args, **kwargs):
        """Generate descriptive statistics."""
        return self._get_pandas().describe(*args, **kwargs)

    def to_pandas(self):
        """Convert to pandas DataFrame."""
        return self._get_pandas()

    def __getitem__(self, key):
        """Support indexing like df['column'] or df[0:5]."""
        return self._get_pandas()[key]

    def __repr__(self):
        """Display as pandas DataFrame."""
        return repr(self._get_pandas())

    def __str__(self):
        """Display as pandas DataFrame."""
        return str(self._get_pandas())

    def __len__(self):
        """Return number of rows."""
        return len(self._get_pandas())

    # Delegate attribute access to the original object
    def __getattr__(self, name):
        return getattr(self._df_obj, name)


@register_function(
    aliases=["包装PyDataFrame", "wrap_dataframe", "pandas_wrapper", "DataFrame包装器"],
    category="utils",
    description="Wrap PyDataFrameElem to provide pandas DataFrame-like interface",
    examples=[
        "# Wrap PyDataFrameElem for pandas-like usage",
        "adata = ov.read('data.h5ad', backend='rust')",
        "obs_wrapper = ov.utils.wrap_dataframe(adata.obs)",
        "print(obs_wrapper.head())",
        "print(obs_wrapper.shape)",
        "print(obs_wrapper.columns)"
    ],
    related=["utils.convert_to_pandas", "utils.read"]
)
def wrap_dataframe(df_obj):
    """
    Wrap PyDataFrameElem to provide pandas DataFrame-like interface.

    Arguments:
        df_obj: PyDataFrameElem or similar DataFrame-like object

    Returns:
        PyDataFrameElemWrapper: Wrapped object with pandas-like methods

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.read('data.h5ad', backend='rust')
        >>> obs = ov.utils.wrap_dataframe(adata.obs)
        >>> print(obs.head())
        >>> print(obs.shape)
    """
    return PyDataFrameElemWrapper(df_obj)


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
        "# Now you can use pandas methods",
        "filtered = obs_df[obs_df['n_genes'] > 1000]"
    ],
    related=["utils.read", "pp.preprocess", "utils.store_layers"]
)
def convert_to_pandas(df_obj):
    """
    Convert PyDataFrameElem or similar objects to pandas DataFrame.

    This is a utility function to convert Rust-based DataFrame objects
    (like PyDataFrameElem from anndata-rs/SnapATAC2) to pandas DataFrames.

    Arguments:
        df_obj: PyDataFrameElem or similar DataFrame-like object

    Returns:
        pandas.DataFrame: Converted DataFrame

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.read('data.h5ad', backend='rust')
        >>> obs_df = ov.utils.convert_to_pandas(adata.obs)
        >>> print(obs_df)  # Now displays as pandas DataFrame
    """
    import pandas as pd

    try:
        # 如果对象已经有 to_pandas 方法，直接使用
        if hasattr(df_obj, 'to_pandas'):
            return df_obj.to_pandas()
    except Exception:
        pass

    try:
        # 方法1: 使用切片获取整个 DataFrame（SnapATAC2 风格）
        import polars as pl
        df_slice = df_obj[:]

        # 检查返回的是否是 polars DataFrame
        if hasattr(df_slice, 'to_pandas'):
            return df_slice.to_pandas()
        elif isinstance(df_slice, pl.DataFrame):
            return df_slice.to_pandas()
        else:
            # 已经是 pandas DataFrame
            return df_slice
    except Exception:
        pass

    try:
        # 方法2: 通过列名构建 DataFrame
        if hasattr(df_obj, '__getitem__'):
            import polars as pl
            data = {}
            # 尝试获取列名
            if hasattr(df_obj, 'columns'):
                columns = df_obj.columns
            else:
                return pd.DataFrame()

            for col in columns:
                try:
                    series = df_obj[col]
                    # 检查是否是 polars Series
                    if hasattr(series, 'to_pandas'):
                        data[col] = series.to_pandas()
                    elif isinstance(series, pl.Series):
                        data[col] = series.to_pandas()
                    else:
                        data[col] = series
                except:
                    pass

            if data:
                return pd.DataFrame(data)
    except Exception:
        pass

    # 如果都失败了，返回空 DataFrame
    return pd.DataFrame()


# 替换 get_vector 内 in_col 分支那一行：
# 旧：return _series_to_np(col_series)
# 新：



    
def read_csv(**kwargs):
    return pd.read_csv(**kwargs)

def read_10x_mtx(**kwargs):
    return sc.read_10x_mtx(**kwargs)

def read_h5ad(**kwargs):
    return sc.read_h5ad(**kwargs)

def read_10x_h5(**kwargs):
    return sc.read_10x_h5(**kwargs)


def data_downloader(url,path,title):
    r"""Download datasets from URL.
    
    Arguments:
        url: The download url of datasets
        path: The save path of datasets
        title: The name of datasets
    
    Returns:
        path: The save path of datasets
    """
    if os.path.isfile(path):
        print("......Loading dataset from {}".format(path))
        return path
    else:
        print("......Downloading dataset save to {}".format(path))
        
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))
    
    start = time.time()
    size = 0
    res = requests.get(url, stream=True)

    chunk_size = 1024000
    content_size = int(res.headers["content-length"]) 
    if res.status_code == 200:
        print('......[%s Size of file]: %0.2f MB' % (title, content_size/chunk_size/10.24))
        with open(path, 'wb') as f:
            for data in res.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data) 
                print('\r'+ '......[Downloader]: %s%.2f%%' % ('>'*int(size*50/content_size), float(size/content_size*100)), end='')
        end = time.time()
        print('\n' + ".......Finish！%s.2f s" % (end - start))
    
    return path

def download_CaDRReS_model():
    r"""load CaDRReS_model
    
    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'cadrres-wo-sample-bias_output_dict_all_genes':'https://figshare.com/ndownloader/files/39753568',
        'cadrres-wo-sample-bias_output_dict_prism':'https://figshare.com/ndownloader/files/39753571',
        'cadrres-wo-sample-bias_param_dict_all_genes':'https://figshare.com/ndownloader/files/39753574',
        'cadrres-wo-sample-bias_param_dict_prism':'https://figshare.com/ndownloader/files/39753577',
    }
    for datasets_name in _datasets.keys():
        print('......CaDRReS model download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='models/{}.pickle'.format(datasets_name),title=datasets_name)
    print('......CaDRReS model download finished!')

def download_GDSC_data():
    r"""load GDSC_data
    
    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'masked_drugs':'https://figshare.com/ndownloader/files/39753580',
        'GDSC_exp':'https://figshare.com/ndownloader/files/39744025',
    }
    for datasets_name in _datasets.keys():
        print('......GDSC data download start:',datasets_name)
        if datasets_name == 'masked_drugs':
            data_downloader(url=_datasets[datasets_name],path='models/{}.csv'.format(datasets_name),title=datasets_name)
        elif datasets_name == 'GDSC_exp':
            data_downloader(url=_datasets[datasets_name],path='models/{}.tsv.gz'.format(datasets_name),title=datasets_name)
    print('......GDSC data download finished!')

@register_function(
    aliases=["下载通路数据库", "download_pathway_database", "download_genesets", "通路数据下载"],
    category="utils",
    description="Download pathway and gene set databases for enrichment analysis",
    examples=[
        "ov.utils.download_pathway_database()",
        "# Downloads the following databases:",
        "# - GO_Biological_Process_2021",
        "# - GO_Cellular_Component_2021", 
        "# - GO_Molecular_Function_2021",
        "# - WikiPathway_2021_Human",
        "# - WikiPathways_2019_Mouse",
        "# - Reactome_2022"
    ],
    related=["utils.geneset_prepare", "bulk.geneset_enrichment", "bulk.pyGSEA"]
)
def download_pathway_database():
    r"""Download pathway and gene set databases for enrichment analysis.

    Arguments:
        None

    Returns:
        None: The function downloads pathway databases to the genesets/ directory including GO_Biological_Process_2021, GO_Cellular_Component_2021, GO_Molecular_Function_2021, WikiPathway_2021_Human, WikiPathways_2019_Mouse, and Reactome_2022.
    """
    _datasets = {
        'GO_Biological_Process_2021':'https://figshare.com/ndownloader/files/39820720',
        'GO_Cellular_Component_2021':'https://figshare.com/ndownloader/files/39820714',
        'GO_Molecular_Function_2021':'https://figshare.com/ndownloader/files/39820711',
        'WikiPathway_2021_Human':'https://figshare.com/ndownloader/files/39820705',
        'WikiPathways_2019_Mouse':'https://figshare.com/ndownloader/files/39820717',
        'Reactome_2022':'https://figshare.com/ndownloader/files/39820702',
    }
     
    for datasets_name in _datasets.keys():
        print('......Pathway Geneset download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.txt'.format(datasets_name),title=datasets_name)
    print('......Pathway Geneset download finished!')
    print('......Other Genesets can be dowload in `https://maayanlab.cloud/Enrichr/#libraries`')

@register_function(
    aliases=["下载基因ID注释", "download_geneid_annotation_pair", "download_gene_mapping", "基因ID映射下载"],
    category="utils",
    description="Download gene ID annotation mapping files for various organisms",
    examples=[
        "ov.utils.download_geneid_annotation_pair()",
        "# Files downloaded to genesets/ directory:",
        "# - pair_GRCm39.tsv (Mouse)",
        "# - pair_GRCh38.tsv (Human)",
        "# - pair_GRCh37.tsv (Human legacy)",
        "# - pair_danRer11.tsv (Zebrafish)"
    ],
    related=["bulk.Matrix_ID_mapping", "utils.geneset_prepare"]
)
def download_geneid_annotation_pair():
    r"""Download gene ID annotation mapping files for various organisms.

    Arguments:
        None

    Returns:
        None: The function downloads mapping files to the genesets/ directory including pair_GRCm39.tsv (Mouse), pair_GRCh38.tsv (Human), pair_GRCh37.tsv (Human legacy), and pair_danRer11.tsv (Zebrafish).
    """
    _datasets = {
        'pair_GRCm39':'https://figshare.com/ndownloader/files/39820684',
        'pair_T2TCHM13':'https://figshare.com/ndownloader/files/39820687',
        'pair_GRCh38':'https://figshare.com/ndownloader/files/39820690',
        'pair_GRCh37':'https://figshare.com/ndownloader/files/39820693',
        'pair_danRer11':'https://figshare.com/ndownloader/files/39820696',
        'pair_danRer7':'https://figshare.com/ndownloader/files/39820699',
        'pair_hgnc_all':'https://github.com/Starlitnightly/omicverse/files/14664966/pair_hgnc_all.tsv.tar.gz',
    }
     
    for datasets_name in _datasets.keys():
        print('......Geneid Annotation Pair download start:',datasets_name)
        if datasets_name == 'pair_hgnc_all':
            # Handle the tar.gz file for HGNC mapping
            import tarfile
            tar_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.tar.gz'.format(datasets_name),title=datasets_name)
            # Extract the TSV file from tar.gz
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path='genesets/')
            print('......Extracted pair_hgnc_all.tsv from tar.gz')
        else:
            model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.tsv'.format(datasets_name),title=datasets_name)
    print('......Geneid Annotation Pair download finished!')

@register_function(
    aliases=["GTF转换", "gtf_to_pair_tsv", "gtf_to_mapping", "GTF基因映射", "convert_gtf"],
    category="utils",
    description="Convert GTF file to gene ID mapping pairs TSV format for Matrix_ID_mapping",
    examples=[
        "# Convert GTF to mapping pairs",
        "gene_count = ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv')",
        "# Keep version numbers in gene IDs",
        "ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv', gene_id_version=True)",
        "# Remove version numbers from gene IDs", 
        "ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv', gene_id_version=False)",
        "# Use converted file for gene mapping",
        "data = ov.bulk.Matrix_ID_mapping(data, 'gene_pairs.tsv')"
    ],
    related=["bulk.Matrix_ID_mapping", "utils.download_geneid_annotation_pair", "utils.read_gtf"]
)
def gtf_to_pair_tsv(gtf_path, output_path, gene_id_version=True):
    r"""Convert GTF file to gene ID mapping pairs TSV format.

    Arguments:
        gtf_path: Path to input GTF file.
        output_path: Path for output TSV file.
        gene_id_version: Whether to keep version numbers in gene IDs. Default: True.

    Returns:
        gene_count: Number of genes processed and written to the output file.

    Examples:
        >>> import omicverse as ov
        >>> # Convert GTF to mapping pairs
        >>> gene_count = ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv')
        >>> # Use converted file for gene mapping
        >>> data = ov.bulk.Matrix_ID_mapping(data, 'gene_pairs.tsv')
    """
    import pandas as pd
    from ._genomics import read_gtf
    
    print(f'......Reading GTF file: {gtf_path}')
    
    # Read GTF file using existing reader
    gtf = read_gtf(gtf_path)
    
    # Filter for gene features only
    gene_features = gtf[gtf['feature'] == 'gene'].copy()
    print(f'......Found {len(gene_features)} gene features')
    
    if len(gene_features) == 0:
        raise ValueError("No gene features found in GTF file!")
    
    # Split attributes to extract gene_id and gene_name
    gene_features = gene_features.split_attribute()
    
    # Check required columns
    if 'gene_id' not in gene_features.columns:
        raise ValueError("gene_id not found in GTF attributes!")
    
    # Extract gene_id and gene_name
    gene_pairs = []
    for idx, row in gene_features.iterrows():
        gene_id = str(row['gene_id'])
        
        # Handle version numbers in gene IDs
        if not gene_id_version and '.' in gene_id:
            gene_id = gene_id.split('.')[0]
        
        # Use gene_name if available, otherwise use gene_id as symbol
        if 'gene_name' in gene_features.columns and pd.notna(row['gene_name']) and str(row['gene_name']) != '.':
            symbol = str(row['gene_name'])
        else:
            symbol = gene_id
            
        gene_pairs.append([gene_id, symbol])
    
    # Create DataFrame and remove duplicates
    df = pd.DataFrame(gene_pairs, columns=['gene_id', 'symbol'])
    df = df.drop_duplicates(subset=['gene_id'], keep='first')
    
    print(f'......Processed {len(df)} unique genes')
    
    # Save to TSV
    df.to_csv(output_path, sep='\t', index=False)
    print(f'......Gene mapping pairs saved to: {output_path}')
    
    return len(df)

def download_tosica_gmt():
    r"""load TOSICA gmt dataset

    """
    _datasets = {
        'GO_bp':'https://figshare.com/ndownloader/files/41460072',
        'TF':'https://figshare.com/ndownloader/files/41460066',
        'reactome':'https://figshare.com/ndownloader/files/41460051',
        'm_GO_bp':'https://figshare.com/ndownloader/files/41460060',
        'm_TF':'https://figshare.com/ndownloader/files/41460057',
        'm_reactome':'https://figshare.com/ndownloader/files/41460054',
        'immune':'https://figshare.com/ndownloader/files/41460063',
    }
     
    for datasets_name in _datasets.keys():
        print('......TOSICA gmt dataset download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.gmt'.format(datasets_name),title=datasets_name)
    print('......TOSICA gmt dataset download finished!')

@register_function(
    aliases=["基因集准备", "geneset_prepare", "pathway_prepare", "基因集加载", "load_geneset"],
    category="utils",
    description="Load and prepare gene sets from GMT/TXT files for enrichment analysis",
    examples=[
        "# Load human gene sets",
        "geneset_dict = ov.utils.geneset_prepare('KEGG_pathways.gmt', organism='Human')",
        "# Load mouse gene sets",
        "geneset_dict = ov.utils.geneset_prepare('GO_biological_process.txt', organism='Mouse')",
        "# Use with enrichment analysis",
        "geneset_dict = ov.utils.geneset_prepare('c2.cp.kegg.v7.4.symbols.gmt')",
        "enrich_res = ov.bulk.geneset_enrichment(gene_list, geneset_dict)"
    ],
    related=["bulk.geneset_enrichment", "utils.download_tosica_gmt", "single.pathway_enrichment"]
)
def geneset_prepare(geneset_path,organism='Human',):
    r"""Load and prepare gene sets from GMT/TXT files for enrichment analysis.

    Arguments:
        geneset_path: Path of geneset file.
        organism: Organism of geneset file. Default: 'Human'.

    Returns:
        go_bio_dict: A dictionary of geneset where keys are pathway names and values are lists of gene symbols.
    """
    result_dict = {}
    file_path=geneset_path
    with open(file_path, 'r', encoding='utf-8') as file:
        for idx,line in enumerate(file):
            line = line.strip()
            if not line:
                continue

            # 自动检测第一个分隔符
            if idx==0:
                first_delimiter=None
                delimiters = ['\t\t',',', '\t', ';', ' ',]
                for delimiter in delimiters:
                    if delimiter in line:
                        first_delimiter = delimiter
                        break
            
            if first_delimiter is None:
                # 如果找不到分隔符，跳过这行
                continue
            
            # 使用第一个分隔符分割行
            parts = line.split(first_delimiter, 1)
            if len(parts) != 2:
                continue
            
            key = parts[0].strip()
            # 使用剩余部分的第一个字符作为分隔符来分割
            value = parts[1].strip().split()

            # 将键值对添加到字典中
            result_dict[key] = value
    go_bio_dict=result_dict

    if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
        for key in go_bio_dict:
            go_bio_dict[key]=[i.lower().capitalize() for i in go_bio_dict[key]]
    elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
        for key in go_bio_dict:
            go_bio_dict[key]=[i.upper() for i in go_bio_dict[key]]
    else:
        for key in go_bio_dict:
            go_bio_dict[key]=[i for i in go_bio_dict[key]]

    return go_bio_dict

def geneset_prepare_old(geneset_path,organism='Human'):
    r"""load geneset

    Parameters
    ----------
    - geneset_path: `str`
        Path of geneset file.
    - organism: `str`
        Organism of geneset file. Default: 'Human'

    Returns
    -------
    - go_bio_dict: `dict`
        A dictionary of geneset.
    """
    go_bio_geneset=pd.read_csv(geneset_path,sep='\t\t',header=None)
    go_bio_dict={}
    if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.lower().capitalize() for i in go_bio_geneset.loc[i,1].split('\t')]
    elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.upper() for i in go_bio_geneset.loc[i,1].split('\t')]
    else:
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i for i in go_bio_geneset.loc[i,1].split('\t')]
    return go_bio_dict

def get_gene_annotation(
        adata: anndata.AnnData, var_by: str = None,
        gtf: os.PathLike = None, gtf_by: str = None,
        by_func: Optional[Callable] = None
) -> None:
    r"""
    Get genomic annotation of genes by joining with a GTF file.
    It was writed by scglue, and I just copy it.

    Arguments:
        adata: Input dataset.
        var_by: Specify a column in ``adata.var`` used to merge with GTF attributes, 
            otherwise ``adata.var_names`` is used by default.
        gtf: Path to the GTF file.
        gtf_by: Specify a field in the GTF attributes used to merge with ``adata.var``,
            e.g. "gene_id", "gene_name".
        by_func: Specify an element-wise function used to transform merging fields,
            e.g. removing suffix in gene IDs.

    Note:
        The genomic locations are converted to 0-based as specified
        in bed format rather than 1-based as specified in GTF format.

    """
    if gtf is None:
        raise ValueError("Missing required argument `gtf`!")
    if gtf_by is None:
        raise ValueError("Missing required argument `gtf_by`!")
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = read_gtf(gtf).query("feature == 'gene'").split_attribute()
    if by_func:
        by_func = np.vectorize(by_func)
        var_by = by_func(var_by)
        gtf[gtf_by] = by_func(gtf[gtf_by])  # Safe inplace modification
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_by], keep="last"
    )  # Typically, scaffolds come first, chromosomes come last
    merge_df = pd.concat([
        pd.DataFrame(gtf.to_bed(name=gtf_by)),
        pd.DataFrame(gtf).drop(columns=Gtf.COLUMNS)  # Only use the splitted attributes
    ], axis=1).set_index(gtf_by).reindex(var_by).set_index(adata.var.index)
    adata.var = adata.var.assign(**merge_df)
    
from importlib import resources

predefined_signatures = dict(
    cell_cycle_human=resources.files("omicverse").joinpath("data_files/cell_cycle_human.gmt").__fspath__(),
    cell_cycle_mouse=resources.files("omicverse").joinpath("data_files/cell_cycle_mouse.gmt").__fspath__(),
    gender_human=resources.files("omicverse").joinpath("data_files/gender_human.gmt").__fspath__(),
    gender_mouse=resources.files("omicverse").joinpath("data_files/gender_mouse.gmt").__fspath__(),
    mitochondrial_genes_human=resources.files("omicverse").joinpath("data_files/mitochondrial_genes_human.gmt").__fspath__(),
    mitochondrial_genes_mouse=resources.files("omicverse").joinpath("data_files/mitochondrial_genes_mouse.gmt").__fspath__(),
    ribosomal_genes_human=resources.files("omicverse").joinpath("data_files/ribosomal_genes_human.gmt").__fspath__(),
    ribosomal_genes_mouse=resources.files("omicverse").joinpath("data_files/ribosomal_genes_mouse.gmt").__fspath__(),
    apoptosis_human=resources.files("omicverse").joinpath("data_files/apoptosis_human.gmt").__fspath__(),
    apoptosis_mouse=resources.files("omicverse").joinpath("data_files/apoptosis_mouse.gmt").__fspath__(),
    human_lung=resources.files("omicverse").joinpath("data_files/human_lung.gmt").__fspath__(),
    mouse_lung=resources.files("omicverse").joinpath("data_files/mouse_lung.gmt").__fspath__(),
    mouse_brain=resources.files("omicverse").joinpath("data_files/mouse_brain.gmt").__fspath__(),
    mouse_liver=resources.files("omicverse").joinpath("data_files/mouse_liver.gmt").__fspath__(),
    emt_human=resources.files("omicverse").joinpath("data_files/emt_human.gmt").__fspath__(),
)

def load_signatures_from_file(input_file: str) -> Dict[str, List[str]]:
    signatures = {}
    with open(input_file) as fin:
        for line in fin:
            items = line.strip().split('\t')
            signatures[items[0]] = list(set(items[2:]))
    print(f"Loaded signatures from GMT file {input_file}.")
    return signatures

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Literal,
    TypeVar,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence,
)

class TestMethod(ModeEnum):  # noqa
    FISHER = "fisher"
    PERM_TEST = "perm_test"



def _mat_mat_corr_sparse(
    X: csr_matrix,
    Y: np.ndarray,
) -> np.ndarray:
    n = X.shape[1]

    X_bar = np.reshape(np.array(X.mean(axis=1)), (-1, 1))
    X_std = np.reshape(
        np.sqrt(np.array(X.power(2).mean(axis=1)) - (X_bar**2)), (-1, 1)
    )

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)

def correlation_pseudotime(
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    method: TestMethod = TestMethod.FISHER,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the correlation between rows in matrix ``X`` columns of matrix ``Y``.

    Parameters
    ----------
    X
        Array or matrix of `(M, N)` elements.
    Y
        Array of `(N, K)` elements.
    method
        Method for p-value calculation.
    n_perms
        Number of permutations if ``method='perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    kwargs
        Keyword arguments for :func:`cellrank._utils._parallelize.parallelize`.

    Returns
    -------
        Correlations, p-values, corrected p-values, lower and upper bound of 95% confidence interval.
        Each array if of shape ``(n_genes, n_lineages)``.
    """

    def perm_test_extractor(
        res: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pvals, corr_bs = zip(*res)
        pvals = np.sum(pvals, axis=0) / float(n_perms)

        corr_bs = np.concatenate(corr_bs, axis=0)
        corr_ci_low, corr_ci_high = np.quantile(corr_bs, q=ql, axis=0), np.quantile(
            corr_bs, q=qh, axis=0
        )

        return pvals, corr_ci_low, corr_ci_high

    if not (0 <= confidence_level <= 1):
        raise ValueError(
            f"Expected `confidence_level` to be in interval `[0, 1]`, found `{confidence_level}`."
        )

    n = X.shape[1]  # genes x cells
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    if issparse(X) and not isspmatrix_csr(X):
        X = csr_matrix(X)

    corr = _mat_mat_corr_sparse(X, Y) if issparse(X) else _mat_mat_corr_dense(X, Y)

    if method == TestMethod.FISHER:
        # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
        mean, se = np.arctanh(corr), 1.0 / np.sqrt(n - 3)
        z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

        z = norm.ppf(qh)
        corr_ci_low = np.tanh(mean - z * se)
        corr_ci_high = np.tanh(mean + z * se)
        pvals = 2 * norm.cdf(-np.abs(z_score))
    else:
        raise NotImplementedError(method)
    '''
    elif method == TestMethod.PERM_TEST:
        if not isinstance(n_perms, int):
            raise TypeError(
                f"Expected `n_perms` to be an integer, found `{type(n_perms).__name__}`."
            )
        if n_perms <= 0:
            raise ValueError(f"Expcted `n_perms` to be positive, found `{n_perms}`.")


        pvals, corr_ci_low, corr_ci_high = parallelize(
            _perm_test,
            np.arange(n_perms),
            as_array=False,
            unit="permutation",
            extractor=perm_test_extractor,
            **kwargs,
        )(corr, X, Y, seed=seed)
    '''
    

    return corr, pvals, corr_ci_low, corr_ci_high

def _np_apply_along_axis(func1d, axis: int, arr: np.ndarray) -> np.ndarray:
    """
    Apply a reduction function over a given axis.

    Parameters
    ----------
    func1d
        Reduction function that operates only on 1 dimension.
    axis
        Axis over which to apply the reduction.
    arr
        The array to be reduced.

    Returns
    -------
    The reduced array.
    """

    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
        return result

    result = np.empty(arr.shape[0])
    for i in range(len(result)):
        result[i] = func1d(arr[i, :])

    return result

def np_mean(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.mean, axis, array)

def np_std(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.std, axis, array)

def _mat_mat_corr_dense(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    #from cellrank.kernels._utils import np_std, np_mean

    n = X.shape[1]

    X_bar = np.reshape(np_mean(X, axis=1), (-1, 1))
    X_std = np.reshape(np_std(X, axis=1), (-1, 1))

    y_bar = np.reshape(np_mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np_std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _perm_test(
    ixs: np.ndarray,
    corr: np.ndarray,
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    seed: Optional[int] = None,
    queue=None,
) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(None if seed is None else seed + ixs[0])
    cell_ixs = np.arange(X.shape[1])
    pvals = np.zeros_like(corr, dtype=np.float64)
    corr_bs = np.zeros((len(ixs), X.shape[0], Y.shape[1]))  # perms x genes x lineages

    mmc = _mat_mat_corr_sparse if issparse(X) else _mat_mat_corr_dense

    for i, _ in enumerate(ixs):
        rs.shuffle(cell_ixs)
        corr_i = mmc(X, Y[cell_ixs, :])
        pvals += np.abs(corr_i) >= np.abs(corr)

        bootstrap_ixs = rs.choice(cell_ixs, replace=True, size=len(cell_ixs))
        corr_bs[i, :, :] = mmc(X[:, bootstrap_ixs], Y[bootstrap_ixs, :])

        if queue is not None:
            queue.put(1)

    if queue is not None:
        queue.put(None)

    return pvals, corr_bs

def anndata_sparse(adata):
    """
    Set adata.X to csr_matrix

    Arguments:
        adata: AnnData

    Returns:
        adata: AnnData

    """

    from scipy.sparse import csr_matrix
    x = csr_matrix(adata.X.copy())
    adata.X=x
    return adata

@register_function(
    aliases=["存储层数据", "store_layers", "save_layers", "层数据存储", "保存层"],
    category="utils",
    description="Store the X matrix of AnnData in adata.uns for later retrieval",
    examples=[
        "# Store current X matrix as 'counts'",
        "ov.utils.store_layers(adata, layers='counts')",
        "# Store normalized data",
        "ov.utils.store_layers(adata, layers='normalized')",
        "# Use with preprocessing pipeline",
        "ov.utils.store_layers(adata, layers='raw')",
        "adata = ov.pp.preprocess(adata)",
        "ov.utils.retrieve_layers(adata, layers='raw')"
    ],
    related=["utils.retrieve_layers", "pp.preprocess", "pp.scale"]
)
def store_layers(adata,layers='counts'):
    """Store the X matrix of AnnData in adata.uns for later retrieval.

    Arguments:
        adata: AnnData object containing single-cell data.
        layers: The layers name to store. Default: 'counts'.

    Returns:
        None: The function modifies adata.uns in place by storing the X matrix.

    Examples:
        >>> import omicverse as ov
        >>> # Store original counts before preprocessing
        >>> ov.utils.store_layers(adata, layers='raw_counts')
        >>> # Apply preprocessing
        >>> adata = ov.pp.preprocess(adata)
        >>> # Retrieve original data if needed
        >>> ov.utils.retrieve_layers(adata, layers='raw_counts')
    """


    if issparse(adata.X) and not isspmatrix_csr(adata.X):
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif issparse(adata.X):
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in {}'.format(layers))

@register_function(
    aliases=["检索层数据", "retrieve_layers", "get_layers", "层数据检索", "获取层"],
    category="utils",
    description="Retrieve previously stored X matrix from adata.uns and restore to adata.X",
    examples=[
        "# Retrieve stored counts data",
        "ov.utils.retrieve_layers(adata, layers='counts')",
        "# Retrieve raw data after preprocessing",
        "ov.utils.retrieve_layers(adata, layers='raw')",
        "# Complete workflow example",
        "ov.utils.store_layers(adata, layers='original')",
        "adata = ov.pp.preprocess(adata)",
        "ov.utils.retrieve_layers(adata, layers='original')"
    ],
    related=["utils.store_layers", "pp.preprocess", "pp.scale"]
)
def retrieve_layers(adata,layers='counts'):
    """Retrieve previously stored X matrix from adata.uns and restore to adata.X.

    Arguments:
        adata: AnnData object containing single-cell data.
        layers: The layers name to retrieve. Default: 'counts'.

    Returns:
        None: The function modifies adata.X in place by restoring the stored matrix.

    Examples:
        >>> import omicverse as ov
        >>> # Store original data before preprocessing
        >>> ov.utils.store_layers(adata, layers='raw_counts')
        >>> # Apply preprocessing
        >>> adata = ov.pp.preprocess(adata)
        >>> # Retrieve original data
        >>> ov.utils.retrieve_layers(adata, layers='raw_counts')
    """

    adata_test=adata.uns['layers_{}'.format(layers)].copy()
    adata_test=adata_test[adata.obs.index,adata.var.index]
    
    if issparse(adata.X) and not isspmatrix_csr(adata.X):
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif issparse(adata.X):
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in raw')
    adata.X=adata_test.X.copy()
    print('......The layers {} of adata have been retreved'.format(layers))
    del adata_test


class easter_egg(object):

    def __init__(self,):
        print('Easter egg is ready to be hatched!')

    def O(self):
        print('尊嘟假嘟')


def save(file, path,):
    """Save object to file using pickle or cloudpickle."""
    print(f"{Colors.HEADER}{Colors.BOLD}💾 Save Operation:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Target path: {Colors.BOLD}{path}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Object type: {Colors.BOLD}{type(file).__name__}{Colors.ENDC}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        import pickle
        print(f"   {Colors.GREEN}Using: {Colors.BOLD}pickle{Colors.ENDC}")
        with open(path, 'wb') as f:
            pickle.dump(file, f)
        print(f"   {Colors.GREEN}✅ Successfully saved!{Colors.ENDC}")
    except:
        import cloudpickle
        print(f"   {Colors.WARNING}Pickle failed, switching to: {Colors.BOLD}cloudpickle{Colors.ENDC}")
        with open(path, 'wb') as f:
            cloudpickle.dump(file, f)
        print(f"   {Colors.GREEN}✅ Successfully saved using cloudpickle!{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")

def load(path,backend=None):
    """Load object from file using pickle or cloudpickle."""
    print(f"{Colors.HEADER}{Colors.BOLD}📂 Load Operation:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Source path: {Colors.BOLD}{path}{Colors.ENDC}")
    if backend is None:
        try:
            import pickle
            print(f"   {Colors.GREEN}Using: {Colors.BOLD}pickle{Colors.ENDC}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"   {Colors.GREEN}✅ Successfully loaded!{Colors.ENDC}")
            print(f"   {Colors.BLUE}Loaded object type: {Colors.BOLD}{type(data).__name__}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
            return data
        except:
            import cloudpickle
            print(f"   {Colors.WARNING}Pickle failed, switching to: {Colors.BOLD}cloudpickle{Colors.ENDC}")
            with open(path, 'rb') as f:
                data = cloudpickle.load(f)
            print(f"   {Colors.GREEN}✅ Successfully loaded using cloudpickle!{Colors.ENDC}")
            print(f"   {Colors.BLUE}Loaded object type: {Colors.BOLD}{type(data).__name__}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
            return data
    else:
        if backend=='pickle':
            import pickle
            print(f"   {Colors.GREEN}Using: {Colors.BOLD}pickle{Colors.ENDC}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"   {Colors.GREEN}✅ Successfully loaded!{Colors.ENDC}")
            print(f"   {Colors.BLUE}Loaded object type: {Colors.BOLD}{type(data).__name__}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
            return data
        elif backend=='cloudpickle':
            import cloudpickle
            print(f"   {Colors.GREEN}Using: {Colors.BOLD}cloudpickle{Colors.ENDC}")
            with open(path, 'rb') as f:
                data = cloudpickle.load(f)
            print(f"   {Colors.GREEN}✅ Successfully loaded!{Colors.ENDC}")
            print(f"   {Colors.BLUE}Loaded object type: {Colors.BOLD}{type(data).__name__}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
            return data
        else:
            raise ValueError(f"Invalid backend: {backend}")


import os
import requests
from tqdm import tqdm
from typing import Optional

def download_data(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download data with headers and progress bar."""
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = os.path.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        print(f"File {file_path} already exists.")
        return file_path

    print(f"Downloading data to {file_path}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://cf.10xgenomics.com/",
    }

    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('Content-Length', 0))
            chunk_size = 8192
            with open(file_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=file_name, ncols=80
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        print(f"Download failed: {e}")
        raise

    return file_path


@register_function(
    aliases=["AnnData兼容转换", "convert_adata_for_rust", "fix_adata_compatibility", "修复兼容性", "rust_compatibility"],
    category="utils", 
    description="Convert old Python-backend h5ad AnnData to be compatible with Rust backend requirements using snapatac2.AnnData",
    examples=[
        "# Convert for Rust backend using snapatac2",
        "adata_rust = ov.utils.convert_adata_for_rust(adata, output_file='fixed_data.h5ad')",
        "# Now you can safely read with Rust backend",
        "adata = ov.read('fixed_data.h5ad', backend='rust')",
        "# Access obs without errors",
        "print(adata.obs['cell_type'])"
    ],
    related=["utils.read", "utils.convert_to_pandas", "pp.preprocess"]
)
def convert_adata_for_rust(adata, output_file=None, verbose=True, close_file=True):
    """Convert AnnData object to be compatible with Rust backend using snapatac2.AnnData.
    
    This function creates a new backed AnnData object using snapatac2.AnnData constructor,
    ensuring full compatibility with Rust backend requirements. It handles:
    - Proper sparse matrix formatting
    - DataFrame compatibility
    - Data type consistency
    - Automatic unique name generation
    
    Arguments:
        adata: AnnData object to be converted (from Python backend)
        output_file: Output h5ad file path. If None, uses temp file. Default: None
        verbose: Whether to print conversion progress. Default: True
        close_file: Whether to close the snapatac2 AnnData after creation. Default: True
        
    Returns:
        output_file: Path to the converted h5ad file compatible with Rust backend
        
    Examples:
        >>> import omicverse as ov
        >>> # Load old h5ad file with Python backend
        >>> adata = ov.read('old_data.h5ad', backend='python') 
        >>> # Convert for Rust compatibility
        >>> output_path = ov.utils.convert_adata_for_rust(adata, 'fixed_data.h5ad')
        >>> # Now read with Rust backend
        >>> adata_rust = ov.read(output_path, backend='rust')
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.sparse import csr_matrix, issparse
    import tempfile
    import os
    
    # Import snapatac2 for creating backed AnnData
    try:
        import snapatac2 as snap
    except ImportError:
        raise ImportError("snapatac2 is required for Rust backend conversion. Install with: pip install snapatac2")
    
    if output_file is None:
        # Create a temporary file if no output specified
        fd, output_file = tempfile.mkstemp(suffix='.h5ad')
        os.close(fd)  # Close the file descriptor
    
    if verbose:
        print(f"{Colors.HEADER}{Colors.BOLD}🔧 Converting AnnData for Rust Backend using anndata-rs{Colors.ENDC}")
        print(f"   {Colors.CYAN}Original shape: {adata.shape}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Output file: {output_file}{Colors.ENDC}")
    
    # Make sure names are unique
    if verbose:
        print(f"   {Colors.BLUE}📝 Ensuring unique names...{Colors.ENDC}")
    
    adata_copy = adata.copy()
    adata_copy.var_names_make_unique()
    adata_copy.obs_names_make_unique()
    
    # Prepare clean data for snapatac2
    def _clean_matrix(X):
        """Clean matrix for snapatac2 compatibility."""
        if X is None:
            return None
        if issparse(X):
            X = X.tocsr(copy=True)
            X.sum_duplicates()
            X.sort_indices()
            # Clean NaN/Inf values
            if np.isnan(X.data).any() or np.isinf(X.data).any():
                X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
                X.eliminate_zeros()
            return X
        else:
            # Clean dense matrices and ensure compatible dtype
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure the matrix has a SnapATAC2-compatible dtype
            if X_clean.dtype.kind == 'V':  # structured/void arrays not supported
                if verbose:
                    print(f"   {Colors.WARNING}⚠️  Converting void array to float32{Colors.ENDC}")
                X_clean = X_clean.astype(np.float32)
            elif X_clean.dtype == np.float64:
                # Convert float64 to float32 for efficiency
                X_clean = X_clean.astype(np.float32)
            elif X_clean.dtype.kind in ['U', 'S']:  # string arrays
                if verbose:
                    print(f"   {Colors.WARNING}⚠️  Converting string array to float32{Colors.ENDC}")
                # Try to convert strings to numbers, fallback to zeros
                try:
                    X_clean = pd.to_numeric(X_clean.flatten(), errors='coerce').values.reshape(X_clean.shape).astype(np.float32)
                    X_clean = np.nan_to_num(X_clean, nan=0.0)
                except:
                    X_clean = np.zeros(X_clean.shape, dtype=np.float32)
            
            return X_clean
    
    def _clean_dataframe(df):
        """Clean DataFrame for snapatac2 compatibility."""
        if df is None or df.empty:
            return df
        
        df_clean = df.copy()
        
        # Reset index to RangeIndex to avoid issues with SnapATAC2
        # The actual obs_names/var_names will be set separately
        df_clean = df_clean.reset_index(drop=True)
        
        # Remove any columns that are completely empty or problematic
        cols_to_drop = []
        for col in df_clean.columns:
            # Check for completely empty columns
            if df_clean[col].isna().all():
                cols_to_drop.append(col)
                continue
                
            # Handle categorical columns
            if pd.api.types.is_categorical_dtype(df_clean[col]):
                try:
                    # Ensure categories are sorted and handle empty categories
                    cat_data = df_clean[col]
                    if len(cat_data.cat.categories) == 0:
                        # Convert empty categorical to string
                        df_clean[col] = df_clean[col].astype(str)
                    elif not cat_data.cat.ordered:
                        try:
                            from natsort import natsorted
                            new_categories = natsorted(cat_data.cat.categories.astype(str))
                        except ImportError:
                            new_categories = sorted(cat_data.cat.categories.astype(str))
                        df_clean[col] = cat_data.cat.reorder_categories(new_categories)
                except Exception:
                    # If categorical handling fails, convert to string
                    df_clean[col] = df_clean[col].astype(str)
            
            # Handle object columns
            elif df_clean[col].dtype == 'object':
                try:
                    # Fill NaN values with empty string first
                    df_clean[col] = df_clean[col].fillna('')
                    # Convert all to string to ensure consistency
                    df_clean[col] = df_clean[col].astype(str)
                except Exception:
                    # If conversion fails, drop the column
                    cols_to_drop.append(col)
                    continue
            
            # Handle numeric columns with NaN/Inf
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                try:
                    if df_clean[col].dtype.kind == 'f':  # float columns
                        col_data = df_clean[col].values
                        if np.isnan(col_data).any() or np.isinf(col_data).any():
                            col_clean = np.nan_to_num(col_data, nan=0.0, 
                                                    posinf=np.finfo(col_data.dtype).max,
                                                    neginf=np.finfo(col_data.dtype).min)
                            df_clean[col] = col_clean
                    elif df_clean[col].dtype.kind in ['i', 'u']:  # integer columns
                        # Fill NaN in integer columns with 0
                        if df_clean[col].isna().any():
                            df_clean[col] = df_clean[col].fillna(0).astype(df_clean[col].dtype)
                except Exception:
                    # If numeric processing fails, try to convert to float
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
                    except Exception:
                        cols_to_drop.append(col)
                        continue
            
            # Handle boolean columns
            elif df_clean[col].dtype == 'bool':
                # Fill NaN boolean values with False
                df_clean[col] = df_clean[col].fillna(False)
        
        # Drop problematic columns
        if cols_to_drop:
            if verbose:
                print(f"   {Colors.WARNING}⚠️  Dropping problematic columns: {cols_to_drop}{Colors.ENDC}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Ensure the DataFrame is not empty after cleaning
        if df_clean.empty:
            # Create a minimal DataFrame with at least one column
            df_clean = pd.DataFrame({'placeholder': [''] * len(df)})
        
        return df_clean
    
    # Clean main data
    if verbose:
        print(f"   {Colors.BLUE}📊 Cleaning data matrices...{Colors.ENDC}")
    
    X_clean = _clean_matrix(adata_copy.X)
    obs_clean = _clean_dataframe(adata_copy.obs)
    var_clean = _clean_dataframe(adata_copy.var)
    
    # Clean obsm
    obsm_clean = {}
    if hasattr(adata_copy, 'obsm') and adata_copy.obsm:
        for key, value in adata_copy.obsm.items():
            obsm_clean[key] = _clean_matrix(value)
    
    # Clean varm  
    varm_clean = {}
    if hasattr(adata_copy, 'varm') and adata_copy.varm:
        for key, value in adata_copy.varm.items():
            varm_clean[key] = _clean_matrix(value)
    
    # Clean uns
    def _clean_uns(uns_dict):
        """Recursively clean uns dictionary."""
        if not isinstance(uns_dict, dict):
            return uns_dict
        
        cleaned = {}
        for key, value in uns_dict.items():
            if value is None:
                cleaned[key] = value
            elif isinstance(value, np.ndarray):
                if value.dtype.kind == 'f':
                    cleaned[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    cleaned[key] = value
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                # Convert numpy scalars to Python native types
                cleaned[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                # Convert 0-dimensional numpy arrays to scalars
                cleaned[key] = value.item()
            elif issparse(value):
                cleaned[key] = _clean_matrix(value)
            elif isinstance(value, pd.DataFrame):
                cleaned[key] = _clean_dataframe(value)
            elif isinstance(value, dict):
                cleaned[key] = _clean_uns(value)
            elif isinstance(value, list):
                try:
                    # Convert numpy types in lists to Python native types
                    converted_list = []
                    for item in value:
                        if isinstance(item, (np.bool_, np.integer, np.floating)):
                            converted_list.append(item.item())
                        elif isinstance(item, np.ndarray) and item.ndim == 0:
                            converted_list.append(item.item())
                        else:
                            converted_list.append(item)
                    
                    # Check if all items are numeric for array conversion
                    if all(isinstance(x, (int, float)) for x in converted_list):
                        arr = np.array(converted_list)
                        if arr.dtype.kind == 'f':
                            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                        cleaned[key] = arr
                    else:
                        cleaned[key] = converted_list
                except Exception:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned
    
    uns_clean = _clean_uns(adata_copy.uns) if hasattr(adata_copy, 'uns') else {}
    
    # Create snapatac2 AnnData object
    if verbose:
        print(f"   {Colors.BLUE}🔧 Creating anndata-rs AnnData object...{Colors.ENDC}")
    
    try:
        # Log the cleaned data types for debugging
        if verbose:
            print(f"   {Colors.BLUE}📋 Data summary before anndata-rs creation:{Colors.ENDC}")
            print(f"      X: {type(X_clean)} {X_clean.shape if X_clean is not None else 'None'}")
            print(f"      obs: {type(obs_clean)} {obs_clean.shape if obs_clean is not None else 'None'}")
            print(f"      var: {type(var_clean)} {var_clean.shape if var_clean is not None else 'None'}")
            if obs_clean is not None and not obs_clean.empty:
                print(f"      obs columns: {list(obs_clean.columns)}")
                print(f"      obs dtypes: {dict(obs_clean.dtypes)}")
            if var_clean is not None and not var_clean.empty:
                print(f"      var columns: {list(var_clean.columns)}")
                print(f"      var dtypes: {dict(var_clean.dtypes)}")
        
        # Create SnapATAC2 AnnData step by step to isolate issues
        # First create with minimal data, then add others
        if verbose:
            print(f"   {Colors.BLUE}🔧 Creating anndata-rs with basic data first...{Colors.ENDC}")
        
        # Create with minimal required data first
        adata_snap = snap.AnnData(
            filename=output_file,
            X=X_clean,
        )
        
        # Add obs and var separately if they exist
        if obs_clean is not None and not obs_clean.empty:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding obs data...{Colors.ENDC}")
            # Make sure obs data is completely clean
            for col in obs_clean.columns:
                if obs_clean[col].dtype == 'object':
                    obs_clean[col] = obs_clean[col].astype(str)
                elif pd.api.types.is_categorical_dtype(obs_clean[col]):
                    # Convert categorical to string to avoid issues
                    obs_clean[col] = obs_clean[col].astype(str)
            # Create new anndata-rs with obs data
            adata_snap.close()
            adata_snap = snap.AnnData(
                filename=output_file,
                X=X_clean,
                obs=obs_clean,
            )
        
        if var_clean is not None and not var_clean.empty:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding var data...{Colors.ENDC}")
            # Make sure var data is completely clean
            for col in var_clean.columns:
                if var_clean[col].dtype == 'object':
                    var_clean[col] = var_clean[col].astype(str)
                elif pd.api.types.is_categorical_dtype(var_clean[col]):
                    var_clean[col] = var_clean[col].astype(str)
            # Recreate with both obs and var
            adata_snap.close()
            adata_snap = snap.AnnData(
                filename=output_file,
                X=X_clean,
                obs=obs_clean,
                var=var_clean,
            )
        
        # Set obs_names and var_names explicitly
        if verbose:
            print(f"   {Colors.BLUE}📝 Setting obs_names and var_names...{Colors.ENDC}")
        
        # Convert names to list of strings
        obs_names_list = [str(name) for name in adata_copy.obs_names]
        var_names_list = [str(name) for name in adata_copy.var_names]
        
        adata_snap.obs_names = obs_names_list
        adata_snap.var_names = var_names_list
        
        # Add obsp if exists (has to be done after creation)
        if hasattr(adata_copy, 'obsp') and adata_copy.obsp:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding obsp matrices...{Colors.ENDC}")
            for key, value in adata_copy.obsp.items():
                if value is not None:
                    adata_snap.obsp[key] = _clean_matrix(value)
        
        # Add varp if exists
        if hasattr(adata_copy, 'varp') and adata_copy.varp:
            if verbose:
                print(f"   {Colors.BLUE}📊 Adding varp matrices...{Colors.ENDC}")
            for key, value in adata_copy.varp.items():
                if value is not None:
                    adata_snap.varp[key] = _clean_matrix(value)
        
        # Add layers if exists
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
        # Clean up the file if creation failed
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
        raise


def _fix_dataframe_for_rust(df, df_type="dataframe"):
    """Fix DataFrame for Rust backend compatibility."""
    if df is None or df.empty:
        return df
        
    import pandas as pd
    import numpy as np
    
    # Create a copy to avoid modifying original
    df_fixed = df.copy()
    
    # 1. Ensure index is consistent
    if not isinstance(df_fixed.index, pd.RangeIndex):
        # Keep the current index but ensure it's proper pandas Index
        df_fixed.index = pd.Index(df_fixed.index, name=df_fixed.index.name)
    
    # 2. Fix categorical columns
    for col in df_fixed.columns:
        if pd.api.types.is_categorical_dtype(df_fixed[col]):
            cat_data = df_fixed[col]
            # Ensure categories are properly ordered
            if not cat_data.cat.ordered:
                # Sort categories naturally if possible
                try:
                    from natsort import natsorted
                    new_categories = natsorted(cat_data.cat.categories.astype(str))
                except ImportError:
                    new_categories = sorted(cat_data.cat.categories.astype(str))
                
                df_fixed[col] = cat_data.cat.reorder_categories(new_categories)
        
        # 3. Handle object columns that might contain mixed types
        elif df_fixed[col].dtype == 'object':
            # Try to convert to string if they're not already
            try:
                # Check if all non-null values are strings
                non_null_values = df_fixed[col].dropna()
                if len(non_null_values) > 0:
                    if not all(isinstance(x, str) for x in non_null_values):
                        df_fixed[col] = df_fixed[col].astype(str)
            except Exception:
                pass
        
        # 4. Handle numeric columns with NaN/Inf
        elif pd.api.types.is_numeric_dtype(df_fixed[col]):
            if df_fixed[col].dtype.kind == 'f':  # float columns
                col_data = df_fixed[col].values
                if np.isnan(col_data).any() or np.isinf(col_data).any():
                    # Replace NaN with 0, Inf with finite values
                    col_clean = np.nan_to_num(col_data, nan=0.0, posinf=np.finfo(col_data.dtype).max, neginf=np.finfo(col_data.dtype).min)
                    df_fixed[col] = col_clean
    
    return df_fixed


def _fix_uns_for_rust(uns_dict):
    """Fix uns dictionary for Rust backend compatibility."""
    if not isinstance(uns_dict, dict):
        return uns_dict
    
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse
    
    uns_fixed = {}
    
    for key, value in uns_dict.items():
        if value is None:
            uns_fixed[key] = value
        elif isinstance(value, np.ndarray):
            # Clean numpy arrays
            if value.dtype.kind == 'f':  # float arrays
                value_clean = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                uns_fixed[key] = value_clean
            else:
                uns_fixed[key] = value
        elif issparse(value):
            # Fix sparse matrices in uns
            uns_fixed[key] = _to_sorted_csr(value)
        elif isinstance(value, pd.DataFrame):
            # Fix DataFrames in uns
            uns_fixed[key] = _fix_dataframe_for_rust(value, f"uns[{key}]")
        elif isinstance(value, dict):
            # Recursively fix nested dictionaries
            uns_fixed[key] = _fix_uns_for_rust(value)
        elif isinstance(value, list):
            # Handle lists
            try:
                # Convert to numpy array if all elements are numeric
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
