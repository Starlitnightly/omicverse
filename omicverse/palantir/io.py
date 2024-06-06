import numpy as np
import pandas as pd
import os.path

import scanpy as sc
from scipy.io import mmread


def _clean_up(df):
    df = df.loc[df.index[df.sum(axis=1) > 0], :]
    df = df.loc[:, df.columns[df.sum() > 0]]
    return df


def from_csv(counts_csv_file, delimiter=","):
    # Read in csv file
    df = pd.read_csv(counts_csv_file, sep=delimiter, index_col=0)
    clean_df = _clean_up(df)
    return clean_df


def from_mtx(mtx_file, gene_name_file):
    # Read in mtx file
    count_matrix = mmread(mtx_file)

    gene_names = np.loadtxt(gene_name_file, dtype=np.dtype("S"))
    gene_names = np.array([gene.decode("utf-8") for gene in gene_names])

    # remove todense
    df = pd.DataFrame(count_matrix.todense(), columns=gene_names)

    return _clean_up(df)


def from_10x(data_dir, use_ensemble_id=True):
    # loads 10x sparse format data
    # data_dir is dir that contains matrix.mtx, genes.tsv and barcodes.tsv
    # return_sparse=True -- returns data matrix in sparse format (default = False)

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

    dataMatrix = pd.DataFrame(
        dataMatrix.todense(), columns=cell_names, index=gene_names
    )

    # combine duplicate genes
    if not use_ensemble_id:
        dataMatrix = dataMatrix.groupby(dataMatrix.index).sum()
    dataMatrix = dataMatrix.transpose()

    return _clean_up(dataMatrix)


def from_10x_HDF5(filename, genome=None):
    ad = sc.read_10x_h5(filename, genome, True)

    dataMatrix = pd.DataFrame(ad.X.todense(), columns=ad.var_names, index=ad.obs_names)

    return _clean_up(dataMatrix)


def from_fcs(
    cls,
    fcs_file,
    cofactor=5,
    metadata_channels=[
        "Time",
        "Event_length",
        "DNA1",
        "DNA2",
        "Cisplatin",
        "beadDist",
        "bead1",
    ],
):
    # Parse the fcs file
    import fcsparser
    text, data = fcsparser.parse(fcs_file)
    data = data.astype(np.float64)

    # Extract the S and N features (Indexing assumed to start from 1)
    # Assumes channel names are in S
    no_channels = text["$PAR"]
    channel_names = [""] * no_channels
    for i in range(1, no_channels + 1):
        # S name
        try:
            channel_names[i - 1] = text["$P%dS" % i]
        except KeyError:
            channel_names[i - 1] = text["$P%dN" % i]
    data.columns = channel_names

    # Metadata and data
    metadata_channels = data.columns.intersection(metadata_channels)
    data_channels = data.columns.difference(metadata_channels)
    # metadata = data[metadata_channels]
    data = data[data_channels]

    # Transform if necessary
    if cofactor is not None or cofactor > 0:
        data = np.arcsinh(np.divide(data, cofactor))

    return data
