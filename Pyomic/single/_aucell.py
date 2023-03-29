# -*- coding: utf-8 -*-

import logging
from ctypes import c_uint32
from math import ceil
from multiprocessing import Array, Process, cpu_count
from multiprocessing.sharedctypes import RawArray
from operator import attrgetter, mul
from typing import Sequence, Type

import numpy as np
import pandas as pd
from boltons.iterutils import chunked
from ctxcore.genesig import GeneSignature
from ctxcore.recovery import enrichment4cells
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
# To reduce the memory footprint of a ranking matrix we use unsigned 32bit integers which provides a range from 0
# through 4,294,967,295. This should be sufficient even for region-based approaches.
DTYPE = "uint32"
DTYPE_C = c_uint32


def create_rankings(ex_mtx: pd.DataFrame, seed=None) -> pd.DataFrame:
    """
    Create a whole genome rankings dataframe from a single cell expression profile dataframe.

    :param ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different
        genes (n_cells x n_genes).
    :return: A genome rankings dataframe (n_cells x n_genes).
    """
    # Do a shuffle would be nice for exactly similar behaviour as R implementation.
    # 1. Ranks are assigned in the range of 1 to n, therefore we need to subtract 1.
    # 2. In case of a tie the 'first' method is used, i.e. we keep the order in the original array. The remove any
    #    bias we shuffle the dataframe before ranking it. This introduces a performance penalty!
    # 3. Genes are ranked according to gene expression in descending order, i.e. from highly expressed (0) to low expression (n).
    # 3. NAs should be given the highest rank numbers. Documentation is bad, so tested implementation via code snippet:
    #
    #    import pandas as pd
    #    import numpy as np
    #    df = pd.DataFrame(data=[4, 1, 3, np.nan, 2, 3], columns=['values'])
    #    # Run below statement multiple times to see effect of shuffling in case of a tie.
    #    df.sample(frac=1.0, replace=False).rank(ascending=False, method='first', na_option='bottom').sort_index() - 1
    #
    return (
        ex_mtx.sample(frac=1.0, replace=False, axis=1, random_state=seed)
        .rank(axis=1, ascending=False, method="first", na_option="bottom")
        .astype(DTYPE)
        - 1
    )


def derive_auc_threshold(ex_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    Derive AUC thresholds for an expression matrix.

    It is important to check that most cells have a substantial fraction of expressed/detected genes in the calculation of
    the AUC.

    :param ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different
        genes (n_cells x n_genes).
    :return: A dataframe with AUC threshold for different quantiles over the number cells: a fraction of 0.01 designates
        that when using this value as the AUC threshold for 99% of the cells all ranked genes used for AUC calculation will
        have had a detected expression in the single-cell experiment.
    """
    return (
        pd.Series(np.count_nonzero(ex_mtx, axis=1)).quantile(
            [0.01, 0.05, 0.10, 0.50, 1]
        )
        / ex_mtx.shape[1]
    )


enrichment = enrichment4cells


def _enrichment(
    shared_ro_memory_array, modules, genes, cells, auc_threshold, auc_mtx, offset
):
    # The rankings dataframe is properly reconstructed (checked this).
    df_rnk = pd.DataFrame(
        data=np.frombuffer(shared_ro_memory_array, dtype=DTYPE).reshape(
            len(cells), len(genes)
        ),
        columns=genes,
        index=cells,
    )
    # To avoid additional memory burden de resulting AUCs are immediately stored in the output sync. array.
    result_mtx = np.frombuffer(auc_mtx.get_obj(), dtype="d")
    inc = len(cells)
    for idx, module in enumerate(modules):
        result_mtx[
            offset + (idx * inc) : offset + ((idx + 1) * inc)
        ] = enrichment4cells(df_rnk, module, auc_threshold).values.ravel(order="C")


def aucell4r(
    df_rnk: pd.DataFrame,
    signatures: Sequence[Type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.

    :param df_rnk: The rank matrix (n_cells x n_genes).
    :param signatures: The gene signatures or regulons.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
    :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
    :param num_workers: The number of cores to use.
    :return: A dataframe with the AUCs (n_cells x n_modules).
    """
    if num_workers == 1:
        # Show progress bar ...
        aucs = pd.concat(
            [
                enrichment4cells(
                    df_rnk,
                    module.noweights() if noweights else module,
                    auc_threshold=auc_threshold,
                )
                for module in tqdm(signatures)
            ]
        ).unstack("Regulon")
        aucs.columns = aucs.columns.droplevel(0)
    else:
        # Decompose the rankings dataframe: the index and columns are shared with the child processes via pickling.
        genes = df_rnk.columns.values
        cells = df_rnk.index.values
        # The actual rankings are shared directly. This is possible because during a fork from a parent process the child
        # process inherits the memory of the parent process. A RawArray is used instead of a synchronize Array because
        # these rankings are read-only.
        shared_ro_memory_array = RawArray(DTYPE_C, mul(*df_rnk.shape))
        array = np.frombuffer(shared_ro_memory_array, dtype=DTYPE)
        # Copy the contents of df_rank into this shared memory block using row-major ordering.
        array[:] = df_rnk.values.ravel(order="C")

        # The resulting AUCs are returned via a synchronize array.
        auc_mtx = Array("d", len(cells) * len(signatures))  # Double precision floats.

        # Convert the modules to modules with uniform weights if necessary.
        if noweights:
            signatures = list(map(lambda m: m.noweights(), signatures))

        # Do the analysis in separate child processes.
        chunk_size = ceil(float(len(signatures)) / num_workers)
        processes = [
            Process(
                target=_enrichment,
                args=(
                    shared_ro_memory_array,
                    chunk,
                    genes,
                    cells,
                    auc_threshold,
                    auc_mtx,
                    (chunk_size * len(cells)) * idx,
                ),
            )
            for idx, chunk in enumerate(chunked(signatures, chunk_size))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Reconstitute the results array. Using C or row-major ordering.
        aucs = pd.DataFrame(
            data=np.ctypeslib.as_array(auc_mtx.get_obj()).reshape(
                len(signatures), len(cells)
            ),
            columns=pd.Index(data=cells, name="Cell"),
            index=pd.Index(
                data=list(map(attrgetter("name"), signatures)), name="Regulon"
            ),
        ).T
    return aucs / aucs.max(axis=0) if normalize else aucs


def aucell(
    exp_mtx: pd.DataFrame,
    signatures: Sequence[Type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    seed=None,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.

    :param exp_mtx: The expression matrix (n_cells x n_genes).
    :param signatures: The gene signatures or regulons.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
    :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
    :param num_workers: The number of cores to use.
    :return: A dataframe with the AUCs (n_cells x n_modules).
    """
    return aucell4r(
        create_rankings(exp_mtx, seed),
        signatures,
        auc_threshold,
        noweights,
        normalize,
        num_workers,
    )
