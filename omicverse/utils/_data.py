r"""
Pyomic data (Pyomic.utils._data)
"""


import time
import requests
import os
import pandas as pd
import scanpy as sc
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
def read(path,backend='python',**kwargs):
    if path.split('.')[-1]=='h5ad':
        if backend=='python':
            return sc.read_h5ad(path,**kwargs)
        elif backend=='rust':
            try:
                import snapatac2 as snap
            except ImportError:
                raise ImportError('snapatac2 is not installed. Please install it with `pip install snapatac2`')
            print(f'{Colors.GREEN}Using snapatac2 to read h5ad file{Colors.ENDC}')
            print(f'{Colors.WARNING}You should run adata.close() after analysis{Colors.ENDC}')
            return snap.read(path,**kwargs)
    elif path.split('.')[-1]=='csv':
        return pd.read_csv(path,**kwargs)
    elif path.split('.')[-1]=='tsv' or path.split('.')[-1]=='txt':
        return pd.read_csv(path,sep='\t',**kwargs)
    elif path.split('.')[-1]=='gz':
        if path.split('.')[-2]=='csv':
            return pd.read_csv(path,**kwargs)
        elif path.split('.')[-2]=='tsv' or path.split('.')[-2]=='txt':
            return pd.read_csv(path,sep='\t',**kwargs)
    else:
        raise ValueError('The type is not supported.')
    
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


def save(file, path):
    """Save object to file using pickle or cloudpickle."""
    print(f"{Colors.HEADER}{Colors.BOLD}💾 Save Operation:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Target path: {Colors.BOLD}{path}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Object type: {Colors.BOLD}{type(file).__name__}{Colors.ENDC}")
    
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

def load(path):
    """Load object from file using pickle or cloudpickle."""
    print(f"{Colors.HEADER}{Colors.BOLD}📂 Load Operation:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Source path: {Colors.BOLD}{path}{Colors.ENDC}")
    
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


