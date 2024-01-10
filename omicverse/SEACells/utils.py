import os

import numpy as np
import pandas as pd
import scanpy as sc
from IPython.display import display

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    """Get absolute path relative to package installation for loading static files.

    :param path: (str) relative path to items
    :return: (str) absolute path to items.
    """
    return os.path.join(_ROOT, "data", path)


def get_Rscript(path):
    """Get absolute path for loading Rscripts.

    :param path: (str) relative path to items
    :return: (str) absolute path to items.
    """
    return os.path.join(_ROOT, "Rscripts", path)


def load_data():
    """Get absolute path for loading sample data.

    :return: (anndata.AnnData object) sample dataset.
    """
    p = get_data("sample_data.h5ad")
    return sc.read(p)


def chromVAR_R(outdir):
    """Run chromVAR R scripts.

    Given an output directory containing:
        - peaks.bed
        - sampling_depth.txt
        - peak_names.txt
        - cell_names.txt
        - counts.txt
    Executes chromVAR R script and writes output files to same directory:
        - deviations.csv
        - variability.csv
        - deviationScores.csv.

    Loads and returns
    (1) pd.Dataframe containing deviations and
    (2) pd.Dataframe containing variability
    """
    import subprocess

    result = subprocess.run(
        ["Rscript", get_Rscript("chromVAR.R"), outdir], stdout=subprocess.PIPE
    )
    print("Executing command:", " ".join(["Rscript", "run_chromVAR.R", outdir]))
    display(result.stdout.decode("utf-8"))

    deviations = pd.read_csv(outdir + "deviations.csv", index_col=[0])
    variability = pd.read_csv(outdir + "variability.csv", index_col=[0])

    return deviations, variability


def run_chromVAR(ad, outdir):
    """Run chromVAR Rscript on anndata object and output deviations to outdir.

    Writes the following files in outdir:
        - peaks.bed
        - sampling_depth.txt
        - peak_names.txt
        - cell_names.txt
        - counts.txt.

    Executes chromVAR_R() - runs chromVAR R script and writes output files to same directory:
        - deviations.csv
        - variability.csv
        - deviationScores.csv

    Loads and returns
    (1) pd.Dataframe containing deviations and
    (2) pd.Dataframe containing variability
    """
    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Convert ad to counts matrix
    atac_df = ad.to_df()

    # Drop zero-count peaks
    atac_df = atac_df.T[atac_df.T.sum(1) > 0]
    ad[:, atac_df.index].var[["seqnames", "start", "end"]].to_csv(
        outdir + "peaks.bed", index=False, sep="\t", header=None
    )
    atac_df.to_csv(outdir + "counts.txt", index=True, sep="\t")
    atac_df.sum(0).to_csv(outdir + "sampling_depth.txt", index=True, sep="\t")

    np.savetxt(outdir + "peak_names.txt", atac_df.index.values, fmt="%s")
    np.savetxt(outdir + "cell_names.txt", atac_df.columns.values, fmt="%s")

    print("Finished writing input files to chromVAR...")

    return chromVAR_R(outdir)


def tanay_metacells():
    """TOOD."""
    raise NotImplementedError


def run_tanay():
    """TODO."""
    raise NotImplementedError
