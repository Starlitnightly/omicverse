from __future__ import annotations

import logging
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import jit

if TYPE_CHECKING:
    from ctxcore.genesig import GeneSignature
    from ctxcore.rnkdb import RankingDatabase

__all__ = [
    "recovery",
    "aucs",
    "enrichment4features",
    "enrichment4cells",
    "leading_edge4row",
]


LOGGER = logging.getLogger(__name__)


def derive_rank_cutoff(
    auc_threshold: float, total_genes: int, rank_threshold: int | None = None
) -> int:
    """
    Get rank cutoff.

    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :param total_genes: The total number of genes ranked.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :return Rank cutoff.
    """
    if not rank_threshold:
        rank_threshold = total_genes - 1

    assert 0 < rank_threshold < total_genes, (
        f"Rank threshold must be an integer between 1 and {total_genes:d}."
    )
    assert 0.0 < auc_threshold <= 1.0, (
        "AUC threshold must be a fraction between 0.0 and 1.0."
    )

    # In the R implementation the cutoff is rounded.
    rank_cutoff = round(auc_threshold * total_genes)
    assert 0 < rank_cutoff <= rank_threshold, (
        f"An AUC threshold of {auc_threshold:f} corresponds to {rank_cutoff:d} top "
        f"ranked genes/regions in the database. Please increase the rank threshold "
        "or decrease the AUC threshold."
    )
    # Make sure we have exactly the same AUC values as the R-SCENIC pipeline.
    # In the latter the rank threshold is not included in AUC calculation.
    rank_cutoff -= 1
    return rank_cutoff


# Do not use numba as it dwarfs the performance.
def rcc2d(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int) -> np.ndarray:
    """
    Calculate recovery curves.

    :param rankings: The features rankings for a gene signature (n_features, n_genes).
    :param weights: The weights of these genes.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :return: Recovery curves (n_features, rank_threshold).
    """
    n_features = rankings.shape[0]
    rccs = np.empty(shape=(n_features, rank_threshold))  # Pre-allocation.
    for row_idx in range(n_features):
        curranking = rankings[row_idx, :]
        rccs[row_idx, :] = np.cumsum(
            np.bincount(curranking, weights=weights)[:rank_threshold]
        )
    return rccs


def recovery(
    rnk: pd.DataFrame,
    total_genes: int,
    weights: np.ndarray,
    rank_threshold: int,
    auc_threshold: float,
    no_auc: bool = False,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate recovery curves and AUCs. This is the workhorse of the recovery algorithm.

    :param rnk: A dataframe containing the rank number of genes of interest.
        Columns correspond to genes.
    :param total_genes: The total number of genes ranked.
    :param weights: the weights associated with the selected genes.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :param no_auc: Do not calculate AUCs.
    :return: A tuple of numpy arrays. The first array contains the recovery curves
        (n_features/n_cells x rank_threshold),
        the second array the AUC values (n_features/n_cells).
    """
    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes, rank_threshold)
    features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    weights = np.insert(weights, len(weights), 0.0)
    n_features = len(features)
    rankings = np.append(
        rankings, np.full(shape=(n_features, 1), fill_value=total_genes), axis=1
    )

    # Calculate recovery curves.
    rccs = rcc2d(rankings, weights, rank_threshold)
    if no_auc:
        return rccs, np.array([])

    # Calculate AUC.
    # For reason of generating the same results as in R we introduce an error by adding
    # one to the rank_cutoff for calculating the maximum AUC.
    maxauc = float((rank_cutoff + 1) * weights.sum())
    assert maxauc > 0
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation. The maximum AUC takes this into account.
    aucs = rccs[:, :rank_cutoff].sum(axis=1) / maxauc

    return rccs, aucs


def enrichment4cells(
    rnk_mtx: pd.DataFrame, regulon: GeneSignature, auc_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Calculate the enrichment of the regulon for the cells in the ranking dataframe.

    :param rnk_mtx: The ranked expression matrix (n_cells, n_genes).
    :param regulon: The regulon the assess for enrichment
    :param auc_threshold: The fraction of the ranked genome to take into account for the
        calculation of the Area Under the recovery Curve.
    :return:
    """
    total_genes = len(rnk_mtx.columns)
    index = pd.MultiIndex.from_tuples(
        list(zip(rnk_mtx.index.values, repeat(regulon.name))), names=["Cell", "Regulon"]
    )
    rnk = rnk_mtx.iloc[:, rnk_mtx.columns.isin(regulon.genes)]
    if rnk.empty or (float(len(rnk.columns)) / float(len(regulon))) < 0.80:
        LOGGER.warning(
            f"Less than 80% of the genes in {regulon.name} are present in the "
            "expression matrix."
        )
        return pd.DataFrame(
            index=index,
            data={"AUC": np.zeros(shape=(rnk_mtx.shape[0]), dtype=np.float64)},
        )
    else:
        weights = np.asarray(
            [
                regulon[gene] if gene in regulon.genes else 1.0
                for gene in rnk.columns.values
            ]
        )
        return pd.DataFrame(
            index=index, data={"AUC": aucs(rnk, total_genes, weights, auc_threshold)}
        )


def enrichment4features(
    rnkdb: RankingDatabase,
    gs: GeneSignature,
    rank_threshold: int = 5000,
    auc_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate AUC and NES for all regulatory features in the supplied database using
    the genes of the given signature.

    :param rnkdb: The database.
    :param gs: The gene signature to assess for enrichment.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :return: A dataframe containing all information.
    """  # noqa: D205
    assert rnkdb, "A database must be supplied"
    assert gs, "A gene signature must be supplied"

    # Load rank of genes from database.
    df = rnkdb.load(gs)
    features = df.index.values
    genes = df.columns.values
    rankings = df.values
    weights = np.asarray([gs[gene] for gene in genes])

    rccs, aucs = recovery(df, rnkdb.total_genes, weights, rank_threshold, auc_threshold)
    ness = (aucs - aucs.mean()) / aucs.std()

    # The creation of a dataframe is a severe performance penalty.
    df_nes = pd.DataFrame(
        index=features, data={("Enrichment", "AUC"): aucs, ("Enrichment", "NES"): ness}
    )
    df_rnks = pd.DataFrame(
        index=features,
        columns=pd.MultiIndex.from_tuples(list(zip(repeat("Ranking"), genes))),
        data=rankings,
    )
    df_rccs = pd.DataFrame(
        index=features,
        columns=pd.MultiIndex.from_tuples(
            list(zip(repeat("Recovery"), np.arange(rank_threshold)))
        ),
        data=rccs,
    )
    return pd.concat([df_nes, df_rccs, df_rnks], axis=1)


def leading_edge(
    rcc: np.ndarray,
    avg2stdrcc: np.ndarray,
    ranking: np.ndarray,
    genes: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[list[tuple[str, float]], int]:
    """
    Calculate the leading edge for a given recovery curve.

    :param rcc: The recovery curve.
    :param avg2stdrcc: The average + 2 standard deviation recovery curve.
    :param ranking: The rank numbers of the gene signature for a given regulatory
        feature.
    :param genes: The genes corresponding to the ranking available in the
         aforementioned parameter.
    :param weights: The weights for these genes.
    :return: The leading edge returned as a list of tuple. Each tuple associates a
        gene part of the leading edge with its rank or with its importance (if gene
        signature supplied). In addition, the rank at maximum difference is returned.
    """

    def critical_point() -> tuple[int, int]:
        """Returns (rank_at_max, max_recovery)."""
        rank_at_max = np.argmax(rcc - avg2stdrcc)
        return rank_at_max, rcc[rank_at_max]

    def get_genes(rank_at_max: int) -> list[tuple[str, float]]:
        sorted_idx = np.argsort(ranking)
        sranking = ranking[sorted_idx]
        gene_ids = genes[sorted_idx]
        # Make sure to include the gene at the leading edge itself. This is different
        # from the i-cisTarget implementation but is inline with the RcisTarget
        # implementation.
        filtered_idx = sranking <= rank_at_max
        filtered_gene_ids = gene_ids[filtered_idx]
        return list(
            zip(
                filtered_gene_ids,
                weights[sorted_idx][filtered_idx]
                if weights is not None
                else sranking[filtered_idx],
            )
        )

    rank_at_max, _n_recovered_genes = critical_point()
    # noinspection PyTypeChecker
    return get_genes(rank_at_max), rank_at_max


def leading_edge4row(
    row: pd.Series,
    avg2stdrcc: np.ndarray,
    genes: np.ndarray,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Calculate the leading edge for a row of a dataframe.

    Should be used with partial function application to make this
    function amenable to the apply idiom common for dataframes.

    :param row: The row of the dataframe to calculate the leading edge for.
    :param avg2stdrcc: The average + 2 standard deviation recovery curve.
    :param genes: The genes corresponding to the ranking available in the supplied row.
    :param weights: The weights for these genes.
    :return: The leading edge returned as a list of tuple. Each tuple associates a gene
        part of the leading edge with its rank or with its importance (if gene signature
        supplied).
    """
    return pd.Series(
        data=leading_edge(
            row["Recovery"].values, avg2stdrcc, row["Ranking"].values, genes, weights
        )
    )


# Giving numba a signature makes the code marginally faster but with losing flexibility
# (only being able to use one type of integers used in rankings).
# @jit(signature_or_function=float64(int16[:], int_, float64), nopython=True)
@jit(nopython=True)
def auc1d(ranking: np.ndarray, rank_cutoff: int, max_auc: float) -> float:
    """
    Calculate the AUC of the recovery curve of a single ranking. [DEPRECATED].

    :param ranking: The rank numbers of the genes.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUC.
    """
    # Using concatenate and full constructs required by numba.
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation.
    x = np.concatenate(
        (
            np.sort(ranking[ranking < rank_cutoff]),
            np.full((1,), rank_cutoff, dtype=np.int_),
        )
    )
    y = np.arange(1, x.size, dtype=np.float64)
    return np.sum(np.diff(x) * y) / max_auc


@jit(nopython=True)
def weighted_auc1d(
    ranking: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUC of the weighted recovery curve of a single ranking.

    :param ranking: The rank numbers of the genes.
    :param weights: The associated weights.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUC.
    """
    # Using concatenate and full constructs required by numba.
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation.
    filter_idx = ranking < rank_cutoff
    x = ranking[filter_idx]
    y = weights[filter_idx]
    sort_idx = np.argsort(x)
    x = np.concatenate((x[sort_idx], np.full((1,), rank_cutoff, dtype=np.int_)))
    y = y[sort_idx].cumsum()
    return np.sum(np.diff(x) * y) / max_auc


def auc2d(
    rankings: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUCs of multiple rankings.

    :param rankings: The rankings.
    :param weights: The weights associated with the selected genes.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUCs.
    """
    n_features = rankings.shape[0]
    aucs = np.empty(shape=(n_features,), dtype=np.float64)  # Pre-allocation.
    for row_idx in range(n_features):
        aucs[row_idx] = weighted_auc1d(
            rankings[row_idx, :], weights, rank_cutoff, max_auc
        )
    return aucs


def aucs(
    rnk: pd.DataFrame, total_genes: int, weights: np.ndarray, auc_threshold: float
) -> np.ndarray:
    """
    Calculate AUCs (implementation without calculating recovery curves first).

    :param rnk: A dataframe containing the rank number of genes of interest. Columns
        correspond to genes.
    :param total_genes: The total number of genes ranked.
    :param weights: The weights associated with the selected genes.
    :param auc_threshold: The fraction of the ranked genome to take into account for the
        calculation of the Area Under the recovery Curve.
    :return: An array with the AUCs.
    """
    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes)
    _features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    y_max = weights.sum()
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation.
    # The maximum AUC takes this into account.
    # For reason of generating the same results as in R we introduce an error by adding
    # one to the rank_cutoff for calculating the maximum AUC.
    maxauc = float((rank_cutoff + 1) * y_max)
    assert maxauc > 0
    return auc2d(rankings, weights, rank_cutoff, maxauc)
