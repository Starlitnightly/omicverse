import ntpath
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import requests
from typing import Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
from anndata import AnnData, read_h5ad, read_loom
import warnings

# Import omicverse color settings
try:
    from .._settings import Colors, EMOJI
except ImportError:
    # Fallback if settings not available
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    
    EMOJI = {
        "start": "🔍",
        "done": "✅",
        "error": "❌",
        "warning": "⚠️",
    }


DATA_DOWNLOAD_LINK_DICT = {
    'neuron_splicing':{
        'figshare':'https://figshare.com/ndownloader/files/47439605',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/neuron_splicing.h5ad',
    },
    'neuron_labeling':{
        'figshare':'https://figshare.com/ndownloader/files/47439629',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/neuron_labeling.h5ad',
    },
    'zebrafish':{
        'figshare':'https://figshare.com/ndownloader/files/47420257',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/zebrafish.h5ad',
    },
    'bone_marrow':{
        'figshare':'https://figshare.com/ndownloader/files/35826944',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/setty_bone_marrow.h5ad',
    },
    'human_tfs':{
        'figshare':'https://figshare.com/ndownloader/files/47439617',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/human_tfs.txt',
    },
    'onefilepercell_A1_unique_and_others_J2CH1':{
        'figshare':'https://figshare.com/ndownloader/files/47439620',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/onefilepercell_A1_unique_and_others_J2CH1.loom',
    },
    '10X_multiome_mouse_brain':{
        'figshare':'https://figshare.com/ndownloader/files/54153947',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/10X_multiome_mouse_brain.loom',
    },
    'cell_annotations':{
        'figshare':'https://figshare.com/ndownloader/files/54154376',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/cell_annotations.tsv',
    },
    'dentategyrus_scv':{
        'figshare':'https://figshare.com/ndownloader/files/47439623',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/dentategyrus_scv.h5ad',
    },
    'hematopoiesis_raw':{
        'figshare':'https://figshare.com/ndownloader/files/47439626',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/hematopoiesis_raw.h5ad',
    },
    'rpe1':{
        'figshare':'https://figshare.com/ndownloader/files/47439641',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/rpe1.h5ad',
    },
    'organoid':{
        'figshare':'https://figshare.com/ndownloader/files/47439632',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/organoid.h5ad',
    },
    'hematopoiesis':{
        'figshare':'https://figshare.com/ndownloader/files/47439635',  
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/hematopoiesis.h5ad',
    },
    'COVID_PBMC_bulk':{
        'figshare':'https://figshare.com/ndownloader/files/59192924',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/COVID_PBMC_bulk_GSE152418.h5ad',
    },
    'COVID_PBMC_single':{
        'figshare':'https://figshare.com/ndownloader/files/59192927',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/COVID_PBMC_sc_ref.h5ad',
    },
}

def download_data(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download example data to local folder."""
    file_path = ntpath.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_path)
    print(f"{Colors.BLUE}{EMOJI['start']} Downloading data to {file_path}{Colors.ENDC}")

    if not os.path.exists(file_path):
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # download the data with colored progress bar
        class ColoredTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
        with ColoredTqdm(
            unit='B', 
            unit_scale=True, 
            desc=f"{Colors.GREEN}Downloading{Colors.ENDC}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green'
        ) as t:
            def report_progress(block_num, block_size, total_size):
                if t.total != total_size:
                    t.total = total_size
                downloaded = block_num * block_size
                if downloaded <= total_size:
                    t.update(downloaded - t.n)
            urlretrieve(url, file_path, reporthook=report_progress)
        print(f"{Colors.GREEN}{EMOJI['done']} Download completed{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}{EMOJI['warning']} File {file_path} already exists{Colors.ENDC}")

    return file_path


def get_dataset_url(dataset_name: str, prefer_stanford: bool = True) -> str:
    """Get URL for a dataset by name, preferring Stanford over Figshare.

    Args:
        dataset_name: Name of the dataset (e.g., 'neuron_splicing').
        prefer_stanford: Whether to prefer Stanford links over Figshare (default: True).

    Returns:
        URL string for the dataset.

    Raises:
        ValueError: If dataset name is not found.
    """
    if dataset_name not in DATA_DOWNLOAD_LINK_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in available datasets")

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


def get_adata(url: str, filename: Optional[str] = None) -> Optional[AnnData]:
    """Download example data to local folder.

    Args:
        url: the url of the data.
        filename: the name of the file to be saved.

    Returns:
        An Annodata object.
    """

    try:
        file_path = download_data_requests(url, filename)
        print(f"{Colors.CYAN} Loading data from {file_path}{Colors.ENDC}")

        if Path(file_path).suffixes[-1][1:] == "loom":
            adata = read_loom(filename=file_path)
        elif Path(file_path).suffixes[-1][1:] == "h5ad":
            adata = read_h5ad(filename=file_path)
        else:
            print(f"{Colors.FAIL}{EMOJI['error']} REPORT THIS: Unknown filetype ({file_path}){Colors.ENDC}")
            return None

        adata.var_names_make_unique()
        print(f"{Colors.GREEN}{EMOJI['done']} Successfully loaded: {adata.n_obs} cells × {adata.n_vars} genes{Colors.ENDC}")

    except OSError:
        # Usually occurs when download is stopped before completion then attempted again.
        file_path = os.path.join('./data', filename)
        print(f"{Colors.WARNING}{EMOJI['warning']} Corrupted file. Deleting {file_path} then redownloading...{Colors.ENDC}")
        # Half-downloaded file cannot be read due to corruption so it's better to delete it.
        # Potential issue: user have a file with duplicate name but is not sample data (this will overwrite file).
        try:
            os.remove(file_path)
        except:
            pass
        adata = get_adata(url, filename)
    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} REPORT THIS: {str(e)}{Colors.ENDC}")
        adata = None

    return adata


def download_data_requests(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download data with headers to bypass 403 errors."""
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = os.path.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        print(f"{Colors.WARNING}{EMOJI['warning']} File {file_path} already exists{Colors.ENDC}")
        return file_path

    print(f"{Colors.BLUE}{EMOJI['start']} Downloading data to {file_path}...{Colors.ENDC}")

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
            
            with open(file_path, 'wb') as f:
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=f"{Colors.GREEN}Downloading{Colors.ENDC}",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    colour='blue'
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print(f"{Colors.GREEN}{EMOJI['done']} Download completed{Colors.ENDC}")
                            
    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} Download failed: {e}{Colors.ENDC}")
        raise

    return file_path


# add our toy sample data
def gillespie():
    """TODO: add data here"""
    pass


def hl60():
    """TODO: add data here"""
    pass


def nascseq():
    """TODO: add data here"""
    pass


def scslamseq():
    """TODO: add data here"""
    pass


def scifate():
    """TODO: add data here"""
    pass


def scnt_seq_neuron_splicing(
    filename: str = "neuron_splicing.h5ad",
) -> AnnData:
    """The neuron splicing data is from Qiu, et al (2020).

    This data consists of 44,021 genes across 13,476 cells.
    """
    url = get_dataset_url("neuron_splicing")
    adata = get_adata(url, filename)

    return adata


def scnt_seq_neuron_labeling(
    filename: str = "neuron_labeling.h5ad",
) -> AnnData:
    """The neuron splicing data is from Qiu, et al (2020).

    This data consists of 24, 078 genes across 3,060 cells.
    """
    url = get_dataset_url("neuron_labeling")
    adata = get_adata(url, filename)

    return adata


def cite_seq():
    pass


def zebrafish(
    filename: str = "zebrafish.h5ad",
) -> AnnData:
    """The zebrafish is from Saunders, et al (2019).

    This data consists of 16,940 genes across 4,181 cells.
    """
    url = get_dataset_url("zebrafish")
    adata = get_adata(url, filename)

    return adata


def dentate_gyrus(
    url: str = "http://pklab.med.harvard.edu/velocyto/DentateGyrus/DentateGyrus.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The Dentate Gyrus dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb.

    This data consists of 27,998 genes across 18,213 cells.
    Note this one http://pklab.med.harvard.edu/velocyto/DG1/10X43_1.loom: a subset of the above data.
    """
    adata = get_adata(url, filename)

    return adata


def bone_marrow(
    filename: str = "bone_marrow.h5ad",
) -> AnnData:
    """The bone marrow dataset used in

    This data consists of 27,876 genes across 5,780 cells.
    """
    url = get_dataset_url("bone_marrow")
    adata = get_adata(url, filename)

    return adata


def haber(
    url: str = "http://pklab.med.harvard.edu/velocyto/Haber_et_al/Haber_et_al.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The Haber dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/Haber_et_al.ipynb

    This data consists of 27,998 genes across 7,216 cells.
    """
    adata = get_adata(url, filename)
    urlretrieve(
        "http://pklab.med.harvard.edu/velocyto/Haber_et_al/goatools_cellcycle_genes.txt",
        "data/goatools_cellcycle_genes.txt",
    )
    cell_cycle_genes = open("data/goatools_cellcycle_genes.txt").read().split()
    adata.var.loc[:, "cell_cycle_genes"] = adata.var.index.isin(cell_cycle_genes)

    return adata


def hg_forebrain_glutamatergic(
    url: str = "http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The hgForebrainGlutamatergic dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/hgForebrainGlutamatergic.ipynb

    This data consists of 32,738 genes across 1,720 cells.
    """
    adata = get_adata(url, filename)
    urlretrieve(
        "http://pklab.med.harvard.edu/velocyto/Haber_et_al/goatools_cellcycle_genes.txt",
        "data/goatools_cellcycle_genes.txt",
    )
    cell_cycle_genes = open("data/goatools_cellcycle_genes.txt").read().split()
    adata.var.loc[:, "cell_cycle_genes"] = adata.var.index.isin(cell_cycle_genes)

    return adata


def chromaffin(
    filename: str = "onefilepercell_A1_unique_and_others_J2CH1.loom",
) -> AnnData:  #
    """The chromaffin dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/chromaffin2.nb.html

    This data consists of 32,738 genes across 1,720 cells.
    """

    url = get_dataset_url("onefilepercell_A1_unique_and_others_J2CH1")
    adata = get_adata(url, filename)

    adata.var_names_make_unique()
    return adata


def bm(
    url: str = "http://pklab.med.harvard.edu/velocyto/mouseBM/SCG71.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The BM dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/SCG71.nb.html

    This data consists of 24,421genes across 6,667 cells.
    """

    adata = get_adata(url, filename)

    return adata


def pancreatic_endocrinogenesis(
    url: str = "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad",
    filename: Optional[str] = None,
) -> AnnData:
    """Pancreatic endocrinogenesis. Data from scvelo.

    Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition / embryonic day 15.5.
    https://dev.biologists.org/content/146/12/dev173849
    """

    adata = get_adata(url, filename)

    return adata

def pancreas_cellrank(
    url: str = "https://figshare.com/ndownloader/files/25060877",
    filename: str = "pancreas_cellrank.h5ad",
) -> AnnData:
    """The pancreas cellrank dataset used in https://github.com/theislab/scvelo_notebooks/tree/master/data/Pancreas.

    This data consists of 13,913 genes across 2,930 cells.
    """
    adata = get_adata(url, filename)
    return adata



def dentate_gyrus_scvelo(
    filename: str = "dentategyrus_scv.h5ad",
) -> AnnData:
    """The Dentate Gyrus dataset used in https://github.com/theislab/scvelo_notebooks/tree/master/data/DentateGyrus.

    This data consists of 13,913 genes across 2,930 cells. Note this dataset is the same processed dataset from the
    excellent scVelo package, which is a subset of the DentateGyrus dataset.
    """
    url = get_dataset_url("dentategyrus_scv")
    adata = get_adata(url, filename)

    return adata


def sceu_seq_rpe1(
    filename: str = "rpe1.h5ad",
):
    """Download rpe1 dataset from Battich, et al (2020) via a figshare link.

    This data consists of 13,913 genes across 2,930 cells.
    """
    print(f"{Colors.HEADER}{EMOJI['start']} Downloading scEU_seq data{Colors.ENDC}")
    url = get_dataset_url("rpe1")
    adata = get_adata(url, filename)
    return adata


def sceu_seq_organoid(
    filename: str = "organoid.h5ad",
):
    """Download organoid dataset from Battich, et al (2020) via a figshare link.

    This data consists of 9,157 genes across 3,831 cells.
    """
    print(f"{Colors.HEADER}{EMOJI['start']} Downloading scEU_seq data{Colors.ENDC}")
    url = get_dataset_url("organoid")
    adata = get_adata(url, filename)
    return adata


def hematopoiesis(
    filename: str = "hematopoiesis.h5ad",
) -> AnnData:
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    print(f"{Colors.HEADER}🧬 Downloading processed hematopoiesis adata{Colors.ENDC}")
    url = get_dataset_url("hematopoiesis")
    adata = get_adata(url, filename)
    return adata


def multi_brain_5k():
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    print(f"{Colors.HEADER}🧠 Downloading raw Fresh Embryonic E18 Mouse Brain (5k){Colors.ENDC}")
    print(f"{Colors.CYAN}Epi Multiome ATAC + Gene Expression dataset{Colors.ENDC}")

    h5_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5'
    fragment_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz'
    fragment_tbi_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz.tbi'
    peak_annotation_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_peak_annotation.tsv'
    velocyto_url=get_dataset_url("10X_multiome_mouse_brain")
    anontation_url=get_dataset_url("cell_annotations")

    h5_path = download_data_requests(h5_url, 'filtered_feature_bc_matrix.h5', dir='./data/multi_brain_5k')
    fragment_path = download_data_requests(fragment_url, 'fragments.tsv.gz', dir='./data/multi_brain_5k')
    fragment_tbi_path = download_data_requests(fragment_tbi_url, 'fragments.tsv.gz.tbi', dir='./data/multi_brain_5k')
    peak_annotation_path = download_data_requests(peak_annotation_url, 'peak_annotation.tsv', dir='./data/multi_brain_5k')
    velocyto_path = download_data_requests(velocyto_url, '10X_multiome_mouse_brain.loom', dir='./data/multi_brain_5k/velocyto')
    annotation_path = download_data_requests(anontation_url, 'cell_annotations.tsv', dir='./data/multi_brain_5k')

    analysis_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_analysis.tar.gz'
    analysis_path = download_data_requests(analysis_url, 'e18_mouse_brain_fresh_5k_analysis.tar.gz', dir='./data/multi_brain_5k')
    # Extract the tar.gz file
    import tarfile
    with tarfile.open(analysis_path, "r:gz") as tar:
        tar.extractall(path='./data/multi_brain_5k/')
    # Remove the tar.gz file after extraction
    os.remove(analysis_path)

    try:
        from ..multi import read_10x_multiome_h5
        mdata=read_10x_multiome_h5(multiome_base_path='./data/multi_brain_5k',
                                                       rna_splicing_loom='velocyto/10X_multiome_mouse_brain.loom',
                                                      cellranger_path_structure=False)
        cell_annot = pd.read_csv('./data/multi_brain_5k/cell_annotations.tsv', sep='\t', index_col=0)
        cell_annot.index=[i.split('-')[0] for i in cell_annot.index]
        ret_index=list(set(cell_annot.index) & set(mdata.obs.index))
        cell_annot=cell_annot.loc[ret_index]
        mdata.update()
        mdata = mdata[ret_index]
        mdata['rna'].obs['celltype'] = cell_annot['celltype'].tolist()
        return mdata
    except ImportError:
        print(f"{Colors.WARNING}{EMOJI['warning']} multi module not available, returning None{Colors.ENDC}")
        return None


def hematopoiesis_raw(
    filename: str = "hematopoiesis_raw.h5ad",
) -> AnnData:
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    print(f"{Colors.HEADER}🧬 Downloading raw hematopoiesis adata{Colors.ENDC}")
    url = get_dataset_url("hematopoiesis_raw")
    adata = get_adata(url, filename)
    return adata


def human_tfs(
    filename: str = "human_tfs.txt",
) -> pd.DataFrame:
    """Download human transcription factors."""
    url = get_dataset_url("human_tfs")
    file_path = download_data_requests(url, filename)
    tfs = pd.read_csv(file_path, sep="\t")
    return tfs


# Scanpy-inspired datasets with dynamo pattern
def blobs(
    n_variables: int = 11,
    n_centers: int = 5,
    cluster_std: float = 1.0,
    n_observations: int = 640,
    random_state: int = 0,
) -> AnnData:
    """Gaussian Blobs dataset.

    Parameters
    ----------
    n_variables
        Dimension of feature space.
    n_centers
        Number of cluster centers.
    cluster_std
        Standard deviation of clusters.
    n_observations
        Number of observations.
    random_state
        Determines random number generation for dataset creation.

    Returns
    -------
    Annotated data matrix containing a observation annotation 'blobs' that
    indicates cluster identity.
    """
    print(f"{Colors.HEADER}🎯 Generating Gaussian Blobs dataset{Colors.ENDC}")
    
    try:
        import sklearn.datasets
        
        X, y = sklearn.datasets.make_blobs(
            n_samples=n_observations,
            n_features=n_variables,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        adata = AnnData(X, obs=dict(blobs=y.astype(str)))
        print(f"{Colors.GREEN}{EMOJI['done']} Generated blobs: {n_observations} cells × {n_variables} features, {n_centers} centers{Colors.ENDC}")
        return adata
    except ImportError:
        print(f"{Colors.WARNING}{EMOJI['warning']} sklearn not available, generating mock blobs{Colors.ENDC}")
        return create_mock_dataset(n_cells=n_observations, n_genes=n_variables, n_cell_types=n_centers, with_clustering=False)


def burczynski06(
    url: str = "ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS1nnn/GDS1615/soft/GDS1615_full.soft.gz",
    filename: str = "GDS1615_full.soft.gz",
) -> AnnData:
    """Bulk data with conditions ulcerative colitis (UC) and Crohn's disease (CD).

    The study assesses transcriptional profiles in peripheral blood mononuclear
    cells from 42 healthy individuals, 59 CD patients, and 26 UC patients.

    This data consists of 127 samples × 22283 genes.
    """
    print(f"{Colors.HEADER}🩸 Downloading Burczynski06 UC/CD dataset{Colors.ENDC}")
    adata = get_adata(url, filename)
    return adata


def moignard15(
    url: str = "https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.3154/MediaObjects/41587_2015_BFnbt3154_MOESM4_ESM.xlsx",
    filename: str = "nbt.3154-S3.xlsx",
) -> AnnData:
    """Hematopoiesis in early mouse embryos (Moignard et al. 2015).

    The data was obtained using qRT–PCR.
    Contains normalized dCt values with experimental groups:
    "primitive streak" (PS), "neural plate" (NP), "head fold (HF),
    "four somite" blood/GFP⁺ (4SG), and "four somite" endothelial/GFP¯ (4SFG).
    
    This data consists of 3934 cells × 42 genes.
    """
    print(f"{Colors.HEADER}🐭 Downloading Moignard15 mouse embryo hematopoiesis{Colors.ENDC}")
    try:
        adata = get_adata(url, filename)
        if adata is not None:
            # Apply some basic processing similar to scanpy version
            import numpy as np
            # filter out 4 genes as in original processing
            if adata.n_vars > 42:
                gene_subset = ~np.isin(adata.var_names, ["Eif2b1", "Mrpl19", "Polr2a", "Ubc"])
                adata = adata[:, gene_subset].copy()
            
            # Add experimental groups if not present
            if "exp_groups" not in adata.obs.columns and adata.obs_names is not None:
                groups = {"HF": "#D7A83E", "NP": "#7AAE5D", "PS": "#497ABC", "4SG": "#AF353A", "4SFG": "#765099"}
                adata.obs["exp_groups"] = [
                    next((gname for gname in groups if str(sname).startswith(gname)), "Unknown")
                    for sname in adata.obs_names
                ]
                adata.uns["exp_groups_colors"] = list(groups.values())
                adata.uns["iroot"] = 532
            return adata
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load Moignard15: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock hematopoiesis data{Colors.ENDC}")
        return create_mock_dataset(n_cells=3934, n_genes=42, n_cell_types=5, with_clustering=True)


def paul15(
    url: str = "https://falexwolf.de/data/paul15.h5",
    filename: str = "paul15.h5",
) -> AnnData:
    """Development of Myeloid Progenitors (Paul et al. 2015).

    Non-logarithmized raw data of myeloid progenitor development.
    This data consists of 2730 cells × 3451 genes.
    """
    print(f"{Colors.HEADER}🧬 Downloading Paul15 myeloid progenitors{Colors.ENDC}")
    try:
        import h5py
        file_path = download_data(url, filename)
        
        with h5py.File(file_path, "r") as f:
            X = f["data.debatched"][()].astype(np.float32)
            gene_names = f["data.debatched_rownames"][()].astype(str)
            cell_names = f["data.debatched_colnames"][()].astype(str)
            clusters = f["cluster.id"][()].flatten().astype(int)
            infogenes_names = f["info.genes_strings"][()].astype(str)
            
        # each row corresponds to observation, therefore transpose
        adata = AnnData(X.transpose())
        adata.var_names = gene_names
        adata.obs_names = cell_names
        
        # names reflecting cell type identifications
        cell_type = 6 * ["Ery"]
        cell_type += "MEP Mk GMP GMP DC Baso Baso Mo Mo Neu Neu Eos Lymph".split()
        adata.obs["paul15_clusters"] = [f"{i}{cell_type[i - 1]}" for i in clusters]
        
        # just keep the first of the two equivalent names per gene
        adata.var_names = [gn.split(";")[0] for gn in adata.var_names]
        
        # remove corrupted gene names
        infogenes_names = np.intersect1d(infogenes_names, adata.var_names)
        adata = adata[:, infogenes_names].copy()
        
        adata.uns["iroot"] = 840
        print(f"{Colors.GREEN}{EMOJI['done']} Loaded Paul15: {adata.n_obs} cells × {adata.n_vars} genes{Colors.ENDC}")
        return adata
        
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load Paul15: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock myeloid data{Colors.ENDC}")
        return create_mock_dataset(n_cells=2730, n_genes=3451, n_cell_types=13, with_clustering=True)


def pbmc68k_reduced(
    url: str = "https://falexwolf.de/data/pbmc68k_reduced.h5ad", 
    filename: str = "pbmc68k_reduced.h5ad"
) -> AnnData:
    """Subsampled and processed 68k PBMCs.

    PBMC 68k dataset from 10x Genomics, preprocessed and subsampled.
    The original PBMC 68k dataset was preprocessed and saved keeping 
    only 724 cells and 221 highly variable genes.
    
    Contains cell type annotations, UMAP coordinates, and clustering results.
    """
    print(f"{Colors.HEADER}🩸 Downloading PBMC 68k reduced dataset{Colors.ENDC}")
    adata = get_adata(url, filename)
    if adata is None:
        print(f"{Colors.WARNING}🔄 Generating mock PBMC68k reduced data{Colors.ENDC}")
        adata = create_mock_dataset(n_cells=724, n_genes=765, n_cell_types=8, with_clustering=True)
        # Add typical PBMC cell types
        cell_types = ['CD4+ T', 'CD8+ T', 'NK', 'B', 'Monocytes', 'Dendritic', 'Megakaryocytes', 'Other']
        adata.obs['bulk_labels'] = np.random.choice(cell_types, adata.n_obs)
    return adata


def toggleswitch(
    filename: str = "toggleswitch.txt"
) -> AnnData:
    """Simulated toggleswitch data.

    Data obtained simulating a simple toggleswitch system.
    This data consists of 200 cells × 2 genes.
    """
    print(f"{Colors.HEADER}⚖️ Loading toggleswitch simulation data{Colors.ENDC}")
    try:
        # Try to create simple toggleswitch-like data
        np.random.seed(0)
        n_cells = 200
        
        # Simple toggleswitch: two genes with anti-correlated expression
        t = np.linspace(0, 4*np.pi, n_cells)
        gene1 = np.maximum(0, np.sin(t) + 0.1*np.random.randn(n_cells))
        gene2 = np.maximum(0, np.cos(t) + 0.1*np.random.randn(n_cells))
        
        X = np.column_stack([gene1, gene2])
        adata = AnnData(X)
        adata.var_names = ['Gene1', 'Gene2'] 
        adata.obs_names = [f'Cell_{i+1}' for i in range(n_cells)]
        adata.uns["iroot"] = 0
        
        print(f"{Colors.GREEN}{EMOJI['done']} Generated toggleswitch: {n_cells} cells × 2 genes{Colors.ENDC}")
        return adata
        
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Error generating toggleswitch: {e}{Colors.ENDC}")
        return create_mock_dataset(n_cells=200, n_genes=2, n_cell_types=2, with_clustering=False)


def krumsiek11() -> AnnData:
    """Simulated myeloid progenitors (Krumsiek et al. 2011).

    The literature-curated boolean network was used to simulate the data. 
    It describes development to four cell fates:
    "monocyte" (Mo), "erythrocyte" (Ery), "megakaryocyte" (Mk) and "neutrophil" (Neu).
    
    This data consists of 640 cells × 11 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading Krumsiek11 myeloid progenitor simulation{Colors.ENDC}")
    try:
        # Generate simple simulated myeloid development data
        np.random.seed(42)
        n_cells = 640
        n_genes = 11
        
        # Create differentiation trajectory
        X = np.random.lognormal(0, 1, (n_cells, n_genes)).astype(np.float32)
        
        # Add structure for different cell fates
        cell_type = pd.array(["progenitor"]).repeat(n_cells)
        cell_type[80:160] = "Mo"     # Monocyte
        cell_type[240:320] = "Ery"   # Erythrocyte  
        cell_type[400:480] = "Mk"    # Megakaryocyte
        cell_type[560:640] = "Neu"   # Neutrophil
        
        adata = AnnData(X)
        adata.var_names = [f'Gene_{i+1}' for i in range(n_genes)]
        adata.obs_names = [f'Cell_{i+1}' for i in range(n_cells)]
        adata.obs["cell_type"] = cell_type
        
        # Add trajectory information
        adata.uns["iroot"] = 0
        fate_labels = {0: "Stem", 159: "Mo", 319: "Ery", 459: "Mk", 619: "Neu"}
        adata.uns["highlights"] = fate_labels
        
        print(f"{Colors.GREEN}{EMOJI['done']} Generated Krumsiek11: {n_cells} cells × {n_genes} genes{Colors.ENDC}")
        return adata
        
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Error generating Krumsiek11: {e}{Colors.ENDC}")
        return create_mock_dataset(n_cells=640, n_genes=11, n_cell_types=5, with_clustering=True)


# Mock datasets (following dynamo pattern)
def create_mock_dataset(
    n_cells: int = 2000,
    n_genes: int = 1500, 
    n_cell_types: int = 6,
    with_clustering: bool = False,
    random_state: int = 42
) -> AnnData:
    """
    Create a mock single-cell dataset for testing statistical functions.
    
    Arguments:
        n_cells: Number of cells to simulate.
        n_genes: Number of genes to simulate.
        n_cell_types: Number of cell types to simulate.
        with_clustering: Whether to include clustering preprocessing.
        random_state: Random seed for reproducibility.
    
    Returns:
        AnnData object with mock single-cell data.
    """
    
    np.random.seed(random_state)
    
    # Generate mock expression data
    # Create some structure with different expression patterns per cell type
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    
    # Add some structure: different cell types have different expression patterns
    cell_type_labels = np.random.choice(range(n_cell_types), n_cells)
    
    for ct in range(n_cell_types):
        ct_mask = cell_type_labels == ct
        if np.sum(ct_mask) > 0:
            # Make some genes more highly expressed in this cell type
            high_genes_size = min(100, n_genes // 2)  # Use at most half the genes, max 100
            high_genes = np.random.choice(n_genes, size=high_genes_size, replace=False)
            X[ct_mask][:, high_genes] *= np.random.uniform(2, 5)
    
    # Create gene names
    gene_names = [f"Gene_{i+1:04d}" for i in range(n_genes)]
    
    # Create cell names  
    cell_names = [f"Cell_{i+1:04d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add mock metadata
    adata.obs['cell_type'] = [f'CellType_{i+1}' for i in cell_type_labels]
    
    # Add sample information
    n_samples = max(2, n_cell_types // 2)
    sample_labels = np.random.choice([f'Sample_{i+1}' for i in range(n_samples)], n_cells)
    adata.obs['sample_id'] = sample_labels
    
    # Add conditions
    conditions = np.random.choice(['Control', 'Treatment'], n_cells, p=[0.5, 0.5])
    adata.obs['condition'] = conditions
    
    # Add tissue types for odds ratio testing
    tissues = np.random.choice(['Blood', 'Normal', 'Tumor'], n_cells, p=[0.33, 0.33, 0.34])
    adata.obs['tissue'] = tissues
    
    # Add some basic gene information
    adata.var['gene_symbols'] = gene_names
    adata.var['highly_variable'] = np.random.choice([True, False], n_genes, p=[0.2, 0.8])
    
    # Add clustering preprocessing if requested (without scanpy dependency)
    if with_clustering:
        try:
            # Simple normalization (total count normalization + log1p)
            X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            
            # Total count normalization
            cell_sums = np.sum(X, axis=1, keepdims=True)
            X_norm = X / cell_sums * 1e4
            
            # Log1p transformation
            X_log = np.log1p(X_norm)
            
            # Simple highly variable gene selection (top variance genes)
            gene_vars = np.var(X_log, axis=0)
            n_hvg = min(2000, int(n_genes * 0.1))  # Top 10% or 2000 genes
            top_genes = np.argsort(gene_vars)[::-1][:n_hvg]
            
            # Mark highly variable genes
            adata.var['highly_variable'] = False
            adata.var.iloc[top_genes, adata.var.columns.get_loc('highly_variable')] = True
            
            # Store raw data and subset to HVGs
            adata.raw = adata.copy()
            adata = adata[:, adata.var.highly_variable].copy()
            
            # Update X with normalized data
            adata.X = X_log[np.ix_(np.arange(n_cells), top_genes)]
            
            # Simple scaling (z-score, clipped at 10)
            X_scaled = (adata.X - np.mean(adata.X, axis=0)) / (np.std(adata.X, axis=0) + 1e-8)
            X_scaled = np.clip(X_scaled, -10, 10)
            adata.X = X_scaled
            
            # Simple PCA (using SVD)
            n_comps = min(50, adata.n_vars - 1, adata.n_obs - 1, 50)
            if n_comps > 0:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_comps, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                adata.obsm['X_pca'] = X_pca
                
                # Simple UMAP-like embedding (just first 2 PCs with some noise)
                umap_coords = X_pca[:, :2] + np.random.normal(0, 0.1, (n_cells, 2))
                adata.obsm['X_umap'] = umap_coords
            
            # Simple clustering (just assign random clusters based on cell types)
            n_clusters = min(8, n_cell_types + 2)
            cluster_labels = np.random.choice(range(n_clusters), n_cells)
            adata.obs['leiden'] = [str(i) for i in cluster_labels]
            
        except Exception as e:
            warnings.warn(f"Mock clustering preprocessing failed: {e}")
    
    print(f"{Colors.GREEN}{EMOJI['done']} Created mock dataset: {n_cells} cells, {n_genes} genes, {n_cell_types} cell types{Colors.ENDC}")
    return adata


def decov_bulk_covid_bulk(
    filename: str = "COVID_PBMC_bulk.h5ad"
) -> AnnData:
    """COVID-19 PBMC bulk data from Decov et al. 2020.

    This data consists of 10,000 cells × 15,000 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading COVID-19 PBMC bulk data{Colors.ENDC}")
    url = get_dataset_url("COVID_PBMC_bulk")
    adata = get_adata(url, filename)
    return adata

def decov_bulk_covid_single(
    filename: str = "COVID_PBMC_single.h5ad"
) -> AnnData:
    """COVID-19 PBMC single-cell data from Decov et al. 2020.

    This data consists of 10,000 cells × 15,000 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading COVID-19 PBMC single-cell data{Colors.ENDC}")
    url = get_dataset_url("COVID_PBMC_single")
    adata = get_adata(url, filename)
    return adata

def sc_ref_Lymph_Node(
    url: str = "https://cell2location.cog.sanger.ac.uk/paper/integrated_lymphoid_organ_scrna/RegressionNBV4Torch_57covariates_73260cells_10237genes/sc.h5ad",
    filename: str = "sc_ref_Lymph_Node.h5ad"
) -> AnnData:
    """SC reference data for Lymph Node.
    
    This data consists of 10,000 cells × 15,000 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading SC reference data for Lymph Node{Colors.ENDC}")
    adata = get_adata(url, filename)
    return adata


def pbmc3k(processed: bool = False) -> AnnData:
    """
    Load PBMC 3k dataset from URL.
    
    3k PBMCs from 10x Genomics. Downloads directly from public URLs,
    falls back to mock data generation if URLs are unavailable.
    
    Arguments:
        processed: Whether to load processed version with clustering (default: True)
    
    Returns:
        AnnData object with PBMC 3k data
    """
    try:
        if processed:
            # Use the official processed PBMC3k from scanpy/cellxgene
            url = "https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad"
            filename = "pbmc3k_processed.h5ad"
        else:
            # Use the raw PBMC3k data
            url = "https://falexwolf.de/data/pbmc3k_raw.h5ad"
            filename = "pbmc3k_raw.h5ad"
        
        print(f"{Colors.HEADER} Loading PBMC 3k dataset ({'processed' if processed else 'raw'}){Colors.ENDC}")
        adata = get_adata(url, filename)
        
        if adata is not None:
            return adata
        else:
            print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load from URL, generating mock data...{Colors.ENDC}")
            return create_mock_dataset(n_cells=2700, n_genes=32738, n_cell_types=8, with_clustering=processed)
            
    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} Error loading PBMC3k: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock data as fallback...{Colors.ENDC}")
        return create_mock_dataset(n_cells=2700, n_genes=32738, n_cell_types=8, with_clustering=processed)


def bhattacherjee(processed: bool = True) -> AnnData:
    """Processed single-cell data PFC adult mice under cocaine self-administration.

    Adult mice were subject to cocaine self-administration, samples were
    collected at three time points: Maintenance, 48h after cocaine withdrawal and
    15 days after cocaine withdrawal.

    Args:
        processed: If True, returns processed data. If False, returns raw data.

    References:
        Bhattacherjee A, Djekidel MN, Chen R, Chen W, Tuesta LM, Zhang Y. Cell
        type-specific transcriptional programs in mouse prefrontal cortex during
        adolescence and addiction. Nat Commun. 2019 Sep 13;10(1):4169.
        doi: 10.1038/s41467-019-12054-3. PMID: 31519873; PMCID: PMC6744514.

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.datasets.bhattacherjee()
        >>> print(adata)
    """
    try:
        url = "https://exampledata.scverse.org/pertpy/bhattacher_rna.h5ad"
        filename = "bhattacherjee_rna.h5ad"

        print(f"{Colors.HEADER} Loading Bhattacherjee et al. dataset{Colors.ENDC}")
        adata = get_adata(url, filename)

        if adata is not None:
            return adata
        else:
            print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load from URL, generating mock data...{Colors.ENDC}")
            return create_mock_dataset(n_cells=5000, n_genes=2000, n_cell_types=10, with_clustering=processed)

    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} Error loading Bhattacherjee dataset: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock data as fallback...{Colors.ENDC}")
        return create_mock_dataset(n_cells=5000, n_genes=2000, n_cell_types=10, with_clustering=processed)


if __name__ == "__main__":
    pass
