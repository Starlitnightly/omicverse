import numpy as np
import pandas as pd
import os.path
import sys
import scanpy as sc
from scipy.io import mmread
import anndata
from typing import Optional, List, Union


def _clean_up(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows and columns with all zeros from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with rows and columns containing all zeros removed.
    """
    df = df.loc[df.sum(axis=1) > 0, :]
    df = df.loc[:, df.sum(axis=0) > 0]
    return df


def from_csv(counts_csv_file: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Read gene expression data from a CSV file.
    
    Parameters
    ----------
    counts_csv_file : str
        Path to the CSV file containing gene expression data.
    delimiter : str, optional
        Delimiter used in the CSV file. Default is ','.
        
    Returns
    -------
    pd.DataFrame
        Gene expression data with rows as cells and columns as genes.
        Cells and genes with zero counts are removed.
    """
    # Read in csv file
    df = pd.read_csv(counts_csv_file, sep=delimiter, index_col=0)
    clean_df = _clean_up(df)
    return clean_df


def from_mtx(mtx_file: str, gene_name_file: str) -> pd.DataFrame:
    """
    Read gene expression data from a Matrix Market format file.
    
    Parameters
    ----------
    mtx_file : str
        Path to the Matrix Market file containing gene expression data.
    gene_name_file : str
        Path to the file containing gene names, one per line.
        
    Returns
    -------
    pd.DataFrame
        Gene expression data with rows as cells and columns as genes.
        Cells and genes with zero counts are removed.
    """
    # Read in mtx file
    count_matrix = mmread(mtx_file)

    gene_names = np.loadtxt(gene_name_file, dtype=np.dtype("S"))
    gene_names = np.array([gene.decode("utf-8") for gene in gene_names])

    # Convert to dense format
    df = pd.DataFrame(count_matrix.todense(), columns=gene_names)

    return _clean_up(df)


def from_10x(data_dir: Optional[str], use_ensemble_id: bool = True) -> pd.DataFrame:
    """
    Load data from 10X Genomics format.
    
    Parameters
    ----------
    data_dir : Optional[str]
        Directory containing the 10X Genomics output files:
        matrix.mtx, genes.tsv, and barcodes.tsv.
        If None, the current directory is used.
    use_ensemble_id : bool, optional
        If True, use Ensembl IDs as gene identifiers. 
        If False, use gene symbols. Default is True.
        
    Returns
    -------
    pd.DataFrame
        Gene expression data with rows as cells and columns as genes.
        Cells and genes with zero counts are removed.
    """
    # loads 10x sparse format data
    if data_dir is None:
        data_dir = "./"
    elif data_dir[len(data_dir) - 1] != "/":
        data_dir = data_dir + "/"

    filename_dataMatrix = os.path.expanduser(data_dir + "matrix.mtx")
    filename_genes = os.path.expanduser(data_dir + "genes.tsv")
    filename_cells = os.path.expanduser(data_dir + "barcodes.tsv")

    # Read in gene expression matrix (sparse matrix)
    # Rows = genes, columns = cells
    dataMatrix = mmread(filename_dataMatrix)

    # Read in row names (gene names / IDs)
    gene_names = np.loadtxt(filename_genes, delimiter="\t", dtype=bytes).astype(str)
    if use_ensemble_id:
        gene_names = [gene[0] for gene in gene_names]
    else:
        gene_names = [gene[1] for gene in gene_names]
    cell_names = np.loadtxt(filename_cells, delimiter="\t", dtype=bytes).astype(str)

    dataMatrix = pd.DataFrame(dataMatrix.todense(), columns=cell_names, index=gene_names)

    # combine duplicate genes
    if not use_ensemble_id:
        dataMatrix = dataMatrix.groupby(dataMatrix.index).sum()
    dataMatrix = dataMatrix.transpose()

    return _clean_up(dataMatrix)


def from_10x_HDF5(filename: str, genome: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from 10X Genomics HDF5 format.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing 10X Genomics data.
    genome : Optional[str], optional
        Name of the genome to load. If None, the first genome is used.
        
    Returns
    -------
    pd.DataFrame
        Gene expression data with rows as cells and columns as genes.
        Cells and genes with zero counts are removed.
    """
    ad = sc.read_10x_h5(filename, genome=genome, gex_only=True)

    dataMatrix = pd.DataFrame(ad.X.todense(), columns=ad.var_names, index=ad.obs_names)

    return _clean_up(dataMatrix)


def from_fcs(
    cls,
    fcs_file: str,
    cofactor: float = 5,
    metadata_channels: List[str] = [
        "Time",
        "Event_length",
        "DNA1",
        "DNA2",
        "Cisplatin",
        "beadDist",
        "bead1",
    ],
) -> pd.DataFrame:
    """
    Load data from Flow Cytometry Standard (FCS) format.
    
    Parameters
    ----------
    cls : object
        Class instance (unused, kept for compatibility).
    fcs_file : str
        Path to the FCS file to load.
    cofactor : float, optional
        Cofactor for arcsinh transformation. Default is 5.
    metadata_channels : List[str], optional
        List of metadata channel names to exclude from the returned data.
        
    Returns
    -------
    pd.DataFrame
        Processed cytometry data with metadata channels removed and
        optionally transformed using arcsinh.
        
    Notes
    -----
    This function requires the fcsparser package to be installed.
    If not installed, it will raise an ImportError with instructions.
    """
    try:
        import fcsparser
    except ImportError:
        raise ImportError(
            "The fcsparser package is required for reading FCS files. "
            "Please install it with: pip install fcsparser"
        )
    # Parse the fcs file
    text, data = fcsparser.parse(fcs_file)
    # Use view instead of newbyteorder for NumPy 2.0 compatibility
    data = data.astype(np.float64, copy=False)

    # Metadata and data
    metadata_channels = data.columns.intersection(metadata_channels)
    data_channels = data.columns.difference(metadata_channels)
    # metadata = data[metadata_channels]
    data = data[data_channels]

    # Transform if necessary
    if cofactor is not None or cofactor > 0:
        data = np.arcsinh(np.divide(data, cofactor))

    return data