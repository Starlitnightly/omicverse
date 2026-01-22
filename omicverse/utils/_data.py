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
from .._settings import Colors, EMOJI  # Import Colors and EMOJI from settings
from .registry import register_function
from ..datasets import download_data_requests


DATA_DOWNLOAD_LINK_DICT = {
    'cadrres-wo-sample-bias_output_dict_all_genes':{
        'figshare':'https://figshare.com/ndownloader/files/39753568',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_output_dict_all_genes.pickle',
    },
    'cadrres-wo-sample-bias_output_dict_prism':{
        'figshare':'https://figshare.com/ndownloader/files/39753571',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_output_dict_prism.pickle',
    },
    'cadrres-wo-sample-bias_param_dict_all_genes':{
        'figshare':'https://figshare.com/ndownloader/files/39753574',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_param_dict_all_genes.pickle',
    },
    'cadrres-wo-sample-bias_param_dict_prism':{
        'figshare':'https://figshare.com/ndownloader/files/39753577',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_param_dict_prism.pickle',
    },
    'GDSC_exp':{
        'figshare':'https://figshare.com/ndownloader/files/39753580',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GDSC_exp.tsv.gz',
    },
    'masked_drugs':{
        'figshare':'https://figshare.com/ndownloader/files/39753583',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/masked_drugs.csv',
    },
    'GO_Biological_Process_2021':{
        'figshare':'https://figshare.com/ndownloader/files/39820720',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_Biological_Process_2021.txt',
    },
    'GO_Cellular_Component_2021':{
        'figshare':'https://figshare.com/ndownloader/files/39820714',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_Cellular_Component_2021.txt',
    },
    'GO_Molecular_Function_2021':{
        'figshare':'https://figshare.com/ndownloader/files/39820711',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_Molecular_Function_2021.txt',
    },
    'WikiPathway_2021_Human':{
        'figshare':'https://figshare.com/ndownloader/files/39820705',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/WikiPathway_2021_Human.txt',
    },
    'WikiPathways_2019_Mouse':{
        'figshare':'https://figshare.com/ndownloader/files/39820717',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/WikiPathways_2019_Mouse.txt',
    },
    'Reactome_2022':{
        'figshare':'https://figshare.com/ndownloader/files/39820702',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/Reactome_2022.txt',
    },
    'pair_GRCm39':{
        'figshare':'https://figshare.com/ndownloader/files/39820684',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_GRCm39.tsv',
    },
    'pair_T2TCHM13':{
        'figshare':'https://figshare.com/ndownloader/files/39820687',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_T2TCHM13.tsv',
    },
    'pair_GRCh38':{
        'figshare':'https://figshare.com/ndownloader/files/39820690',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_GRCh38.tsv',
    },
    'pair_GRCh37':{
        'figshare':'https://figshare.com/ndownloader/files/39820693',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_GRCh37.tsv',
    },
    'pair_danRer11':{
        'figshare':'https://figshare.com/ndownloader/files/39820696',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_danRer11.tsv',
    },
    'pair_danRer7':{
        'figshare':'https://figshare.com/ndownloader/files/39820699',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_danRer7.tsv',
    },
    'GO_bp':{
        'figshare':'https://figshare.com/ndownloader/files/41460072',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_bp.gmt',
    },
    'TF':{
        'figshare':'https://figshare.com/ndownloader/files/41460066',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/TF.gmt',
    },
    'reactome':{
        'figshare':'https://figshare.com/ndownloader/files/41460051',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/reactome.gmt',
    },
    'm_GO_bp':{
        'figshare':'https://figshare.com/ndownloader/files/41460060',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/m_GO_bp.gmt',
    },
    'm_TF':{
        'figshare':'https://figshare.com/ndownloader/files/41460057',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/m_TF.gmt',
    },
    'm_reactome':{
        'figshare':'https://figshare.com/ndownloader/files/41460054',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/m_reactome.gmt',
    },
    'immune':{
        'figshare':'https://figshare.com/ndownloader/files/41460049',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/immune.gmt',
    },

}


def get_utils_dataset_url(dataset_name: str, prefer_stanford: bool = True) -> str:
    """Get URL for a dataset by name, preferring Stanford over Figshare.

    Args:
        dataset_name: Name of the dataset (e.g., 'GO_bp', 'GDSC_exp').
        prefer_stanford: Whether to prefer Stanford links over Figshare (default: True).

    Returns:
        URL string for the dataset.

    Raises:
        ValueError: If dataset name is not found.
    """
    if dataset_name not in DATA_DOWNLOAD_LINK_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATA_DOWNLOAD_LINK_DICT")

    dataset_urls = DATA_DOWNLOAD_LINK_DICT[dataset_name]

    if prefer_stanford and 'stanford' in dataset_urls:
        print(f"{Colors.CYAN}Using Stanford mirror for {dataset_name}{Colors.ENDC}")
        return dataset_urls['stanford']
    elif 'figshare' in dataset_urls:
        if prefer_stanford:
            print(f"{Colors.WARNING}{EMOJI['warning']} Stanford link not available for {dataset_name}, using Figshare{Colors.ENDC}")
        return dataset_urls['figshare']
    else:
        raise ValueError(f"No valid URL found for dataset '{dataset_name}'")


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
            return adata

        else:
            raise ValueError("backend must be 'python' or 'rust'")

    # 其它纯表格：pandas 会自动识别 gz 压缩，不必手动区分
    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')


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


# Deprecated: data_downloader has been replaced by download_data_requests from omicverse.datasets
# All download functions now use download_data_requests for better error handling and progress display

def download_CaDRReS_model():
    r"""load CaDRReS_model

    Parameters
    ---------

    Returns
    -------

    """
    _datasets = [
        'cadrres-wo-sample-bias_output_dict_all_genes',
        'cadrres-wo-sample-bias_output_dict_prism',
        'cadrres-wo-sample-bias_param_dict_all_genes',
        'cadrres-wo-sample-bias_param_dict_prism',
    ]
    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......CaDRReS model download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        model_path = download_data_requests(url=url, file_path=f'{datasets_name}.pickle', dir='./models')
    print(f'{Colors.GREEN}{EMOJI["done"]} CaDRReS model download finished!{Colors.ENDC}')

def download_GDSC_data():
    r"""load GDSC_data

    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'masked_drugs': '.csv',
        'GDSC_exp': '.tsv.gz',
    }
    for datasets_name, ext in _datasets.items():
        print(f'{Colors.CYAN}......GDSC data download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}{ext}', dir='./models')
    print(f'{Colors.GREEN}{EMOJI["done"]} GDSC data download finished!{Colors.ENDC}')

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
    _datasets = [
        'GO_Biological_Process_2021',
        'GO_Cellular_Component_2021',
        'GO_Molecular_Function_2021',
        'WikiPathway_2021_Human',
        'WikiPathways_2019_Mouse',
        'Reactome_2022',
    ]

    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......Pathway Geneset download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}.txt', dir='./genesets')
    print(f'{Colors.GREEN}{EMOJI["done"]} Pathway Geneset download finished!{Colors.ENDC}')
    print(f'{Colors.CYAN}......Other Genesets can be downloaded from https://maayanlab.cloud/Enrichr/#libraries{Colors.ENDC}')

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
    _datasets = [
        'pair_GRCm39',
        'pair_T2TCHM13',
        'pair_GRCh38',
        'pair_GRCh37',
        'pair_danRer11',
        'pair_danRer7',
    ]

    # Add special handling for pair_hgnc_all
    _special_datasets = {
        'pair_hgnc_all': 'https://github.com/Starlitnightly/omicverse/files/14664966/pair_hgnc_all.tsv.tar.gz'
    }

    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......Geneid Annotation Pair download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}.tsv', dir='./genesets')

    # Handle special datasets not in DATA_DOWNLOAD_LINK_DICT
    for datasets_name, url in _special_datasets.items():
        print(f'{Colors.CYAN}......Geneid Annotation Pair download start: {datasets_name}{Colors.ENDC}')
        import tarfile
        tar_path = download_data_requests(url=url, file_path=f'{datasets_name}.tar.gz', dir='./genesets')
        # Extract the TSV file from tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path='genesets/')
        print(f'{Colors.GREEN}......Extracted {datasets_name}.tsv from tar.gz{Colors.ENDC}')

    print(f'{Colors.GREEN}{EMOJI["done"]} Geneid Annotation Pair download finished!{Colors.ENDC}')

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
    _datasets = [
        'GO_bp',
        'TF',
        'reactome',
        'm_GO_bp',
        'm_TF',
        'm_reactome',
        'immune',
    ]

    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......TOSICA gmt dataset download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}.gmt', dir='./genesets')
    print(f'{Colors.GREEN}{EMOJI["done"]} TOSICA gmt dataset download finished!{Colors.ENDC}')

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


# Note: download_data function has been removed.
# Please use download_data_requests from omicverse.datasets instead.


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
