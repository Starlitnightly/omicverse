import numpy as np
import pandas as pd
import pyranges as pr
import scanpy as sc
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from . import core


def prepare_multiome_anndata(
    atac_ad, rna_ad, SEACells_label="SEACell", n_bins_for_gc=50
):
    """Function to create metacell Anndata objects from single-cell Anndata objects for multiome data.

    :param atac_ad: (Anndata) ATAC Anndata object with raw peak counts in `X`. These anndata objects should be constructed
     using the example notebook available in
    :param rna_ad: (Anndata) RNA Anndata object with raw gene expression counts in `X`. Note: RNA and ATAC anndata objects
     should contain the same set of cells
    :param SEACells_label: (str) `atac_ad.obs` field for constructing metacell matrices. Same field will be used for
      summarizing RNA and ATAC metacells.
    :param n_bins_gc: (int) Number of bins for creating GC bins of ATAC peaks.
    :return: ATAC metacell Anndata object and RNA metacell Anndata object.
    """
    # Subset of cells common to ATAC and RNA
    common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
    if len(common_cells) != atac_ad.shape[0]:
        print(
            "Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used."
        )
    atac_mod_ad = atac_ad[common_cells, :]
    rna_mod_ad = rna_ad[common_cells, :]

    # #################################################################################
    # Generate metacell matrices

    # Set of metacells
    metacells = atac_mod_ad.obs[SEACells_label].astype(str).unique()
    metacells = metacells[atac_mod_ad.obs[SEACells_label].value_counts()[metacells] > 1]

    print("Generating Metacell matrices...")
    print(" ATAC")
    atac_meta_ad = core.summarize_by_SEACell(
        atac_mod_ad, SEACells_label=SEACells_label, summarize_layer="X"
    )
    atac_meta_ad = atac_meta_ad[metacells, :]
    # ATAC - Summarize SVD representation

    svd = pd.DataFrame(atac_mod_ad.obsm["X_svd"], index=atac_mod_ad.obs_names)
    summ_svd = svd.groupby(atac_mod_ad.obs[SEACells_label]).mean()
    atac_meta_ad.obsm["X_svd"] = summ_svd.loc[atac_meta_ad.obs_names, :].values

    # ATAC - Normalize
    _add_atac_meta_data(atac_meta_ad, atac_mod_ad, n_bins_for_gc)
    sc.pp.filter_genes(atac_meta_ad, min_cells=1)
    _normalize_ad(atac_meta_ad)

    # RNA summaries using ATAC SEACells
    print(" RNA")
    rna_mod_ad.obs["temp"] = atac_mod_ad.obs[SEACells_label]
    rna_meta_ad = core.summarize_by_SEACell(
        rna_mod_ad, SEACells_label="temp", summarize_layer="X"
    )
    rna_meta_ad = rna_meta_ad[metacells, :]
    _normalize_ad(rna_meta_ad)

    return atac_meta_ad, rna_meta_ad


def _normalize_ad(meta_ad, save_raw=True):
    if save_raw:
        # Save in raw
        meta_ad.raw = meta_ad.copy()

    # Normalize
    sc.pp.normalize_total(meta_ad, key_added="n_counts")
    sc.pp.log1p(meta_ad)


def _add_atac_meta_data(atac_meta_ad, atac_ad, n_bins_for_gc):
    atac_ad.var["log_n_counts"] = np.ravel(np.log10(atac_ad.X.sum(axis=0)))

    atac_meta_ad.var["GC_bin"] = np.digitize(
        atac_ad.var["GC"], np.linspace(0, 1, n_bins_for_gc)
    )
    atac_meta_ad.var["counts_bin"] = np.digitize(
        atac_ad.var["log_n_counts"],
        np.linspace(
            atac_ad.var["log_n_counts"].min(),
            atac_ad.var["log_n_counts"].max(),
            n_bins_for_gc,
        ),
    )


def _pyranges_from_strings(pos_list):
    """Function to create pyranges for a `pd.Series` of strings."""
    # Chromosome and positions
    chr = pos_list.str.split(":").str.get(0)
    start = pd.Series(pos_list.str.split(":").str.get(1)).str.split("-").str.get(0)
    end = pd.Series(pos_list.str.split(":").str.get(1)).str.split("-").str.get(1)

    # Create ranges
    gr = pr.PyRanges(chromosomes=chr, starts=start, ends=end)
    return gr


def _pyranges_to_strings(peaks):
    """Function to convert pyranges to `pd.Series` of strings of format 'chr:start-end'."""
    # Chromosome and positions
    chr = peaks.Chromosome.astype(str).values
    start = peaks.Start.astype(str).values
    end = peaks.End.astype(str).values

    # Create ranges
    gr = chr + ":" + start + "-" + end

    return gr


def load_transcripts(path_to_gtf):
    """Load transcripts from GTF File. `chr` is preprended to each entry."""
    gtf = pr.read_gtf(path_to_gtf)
    gtf.Chromosome = "chr" + gtf.Chromosome.astype(str)
    transcripts = gtf[gtf.Feature == "transcript"]
    return transcripts


def _peaks_correlations_per_gene(
    gene,
    atac_exprs,
    rna_exprs,
    atac_meta_ad,
    peaks_pr,
    transcripts,
    span,
    n_rand_sample=100,
):
    # Gene transcript - use the longest transcript
    gene_transcripts = transcripts[transcripts.gene_name == gene]
    if len(gene_transcripts) == 0:
        return 0
    longest_transcript = gene_transcripts[
        np.arange(len(gene_transcripts))
        == np.argmax(gene_transcripts.End - gene_transcripts.Start)
    ]
    start = longest_transcript.Start.values[0] - span
    end = longest_transcript.End.values[0] + span

    # Gene span
    gene_pr = pr.from_dict(
        {
            "Chromosome": [longest_transcript.Chromosome.values[0]],
            "Start": [start],
            "End": [end],
        }
    )
    gene_peaks = peaks_pr.overlap(gene_pr)
    if len(gene_peaks) == 0:
        return 0
    gene_peaks_str = _pyranges_to_strings(gene_peaks)

    # Compute correlations
    X = atac_exprs.loc[:, gene_peaks_str].T
    cors = 1 - np.ravel(
        pairwise_distances(
            np.apply_along_axis(rankdata, 1, X.values),
            rankdata(rna_exprs[gene].T.values).reshape(1, -1),
            metric="correlation",
        )
    )
    cors = pd.Series(cors, index=gene_peaks_str)

    # Random background
    df = pd.DataFrame(1.0, index=cors.index, columns=["cor", "pval"])
    df["cor"] = cors
    for p in df.index:
        # TODO: Handle exception properly
        try:
            # Try random sampling without replacement
            rand_peaks = np.random.choice(
                atac_meta_ad.var_names[
                    (atac_meta_ad.var["GC_bin"] == atac_meta_ad.var["GC_bin"][p])
                    & (
                        atac_meta_ad.var["counts_bin"]
                        == atac_meta_ad.var["counts_bin"][p]
                    )
                ],
                n_rand_sample,
                False,
            )
        except:  # noqa: E722
            rand_peaks = np.random.choice(
                atac_meta_ad.var_names[
                    (atac_meta_ad.var["GC_bin"] == atac_meta_ad.var["GC_bin"][p])
                    & (
                        atac_meta_ad.var["counts_bin"]
                        == atac_meta_ad.var["counts_bin"][p]
                    )
                ],
                n_rand_sample,
                True,
            )

        if type(atac_exprs) is sc.AnnData:
            X = pd.DataFrame(atac_exprs[:, rand_peaks].X.todense().T)
        else:
            X = atac_exprs.loc[:, rand_peaks].T

        rand_cors = 1 - np.ravel(
            pairwise_distances(
                np.apply_along_axis(rankdata, 1, X.values),
                rankdata(rna_exprs[gene].T.values).reshape(1, -1),
                metric="correlation",
            )
        )

        m = np.mean(rand_cors)
        v = np.std(rand_cors)

        from scipy.stats import norm

        df.loc[p, "pval"] = 1 - norm.cdf(cors[p], m, v)

    return df


def get_gene_peak_correlations(
    atac_meta_ad,
    rna_meta_ad,
    path_to_gtf,
    gene_ranges=None,
    span=100000,
    n_jobs=1,
    gene_set=None,
):
    """Function to compute  correlations between gene expression and peak accessibility.

    :param atac_meta_ad: (Anndata) ATAC metacell Anndata created using `prepare_multiome_anndata`
    :param rna_meta_ad: (Anndata) RNA metacell Anndata created using `prepare_multiome_anndata`
    :param path_to_gtf: (str or None) Path to ENSEMBL GTF file OR None if using pyranges object as input
    :param gene_ranges: (pyranges or None) Pyranges object containing regions corresponding to custom annotation sets. Only used if path_to_gtf is None.
    :param span: (int) Genomic window around the gene body to identify for which correlations with expression are computed
    :param n_jobs: (int) Number of jobs for parallel processing
    :param gene_set: (pd.Series) Subset of genes for which to compute correlations. All genes are used by default

    :return: `pd.Series` with a dataframe of correlation and p-value for each gene. Note that p-value is one-sided assuming positive correlations
    """
    # #################################################################################
    print("Loading transcripts per gene...")
    if path_to_gtf is None:
        transcripts = gene_ranges
    else:
        transcripts = load_transcripts(path_to_gtf)

    print("Preparing matrices for gene-peak associations")
    atac_exprs = pd.DataFrame(
        atac_meta_ad.X.todense(),
        index=atac_meta_ad.obs_names,
        columns=atac_meta_ad.var_names,
    )
    rna_exprs = pd.DataFrame(
        rna_meta_ad.X.todense(),
        index=rna_meta_ad.obs_names,
        columns=rna_meta_ad.var_names,
    )
    peaks_pr = _pyranges_from_strings(atac_meta_ad.var_names)

    print("Computing peak-gene correlations")
    if gene_set is None:
        use_genes = rna_meta_ad.var_names
    else:
        use_genes = gene_set
    from joblib import Parallel, delayed

    gene_peak_correlations = Parallel(n_jobs=n_jobs)(
        delayed(_peaks_correlations_per_gene)(
            gene, atac_exprs, rna_exprs, atac_meta_ad, peaks_pr, transcripts, span
        )
        for gene in tqdm(use_genes)
    )
    gene_peak_correlations = pd.Series(gene_peak_correlations, index=use_genes)
    return gene_peak_correlations


def get_gene_peak_assocations(gene_peak_correlations, pval_cutoff=1e-1, cor_cutoff=0.1):
    """Determine the number of significantly correlated peaks per gene.

    :param gene_peak_correlations: (pd.Series) Output of `get_gene_peak_correlations` function
    :param p_val_cutoff: (float) Nominal p-value cutoff for test of significance of correlation
    :param cor_cutoff: (float) Correlation cutoff

    :return: `pd.Series` with number of significantly positive correlated peaks with each gene
    """
    peak_counts = pd.Series(0, index=gene_peak_correlations.index)
    for gene in tqdm(peak_counts.index):
        df = gene_peak_correlations[gene]
        if type(df) is int:
            continue
        gene_peaks = df.index[(df["pval"] < pval_cutoff) & (df["cor"] > cor_cutoff)]
        peak_counts[gene] = len(gene_peaks)

    return peak_counts


def get_gene_scores(
    atac_meta_ad, gene_peak_correlations, pval_cutoff=1e-1, cor_cutoff=0.1
):
    """Compute the aggregate accessibility of all peaks associated with each gene.

    Gene scores are computed as the aggregate accessibility of all the signficantly correlated peaks associated with a gene.

    :param atac_meta_ad: (Anndata) ATAC metacell Anndata created using `prepare_multiome_anndata`
    :param gene_peak_correlations: (pd.Series) Output of `get_gene_peak_correlations` function
    :param p_val_cutoff: (float) Nominal p-value cutoff for test of significance of correlation
    :param cor_cutoff: (float) Correlation cutoff

    :return: `pd.DataFrame` of ATAC gene scores (cells X genes)
    """
    gene_scores = pd.DataFrame(
        0.0, index=atac_meta_ad.obs_names, columns=gene_peak_correlations.index
    )

    for gene in tqdm(gene_scores.columns):
        df = gene_peak_correlations[gene]
        if type(df) is int:
            continue
        gene_peaks = df.index[(df["pval"] < pval_cutoff) & (df["cor"] > cor_cutoff)]
        gene_scores[gene] = np.ravel(
            np.dot(atac_meta_ad[:, gene_peaks].X.todense(), df.loc[gene_peaks, "cor"])
        )
    gene_scores = gene_scores.loc[:, (gene_scores.sum() >= 0)]
    return gene_scores
