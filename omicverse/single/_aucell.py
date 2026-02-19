# -*- coding: utf-8 -*-

import logging
from ctypes import c_uint32
from math import ceil
import os
from multiprocessing import cpu_count
from operator import attrgetter, mul
from typing import Sequence, Type

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.sparse import csr_matrix

LOGGER = logging.getLogger(__name__)
# To reduce the memory footprint of a ranking matrix we use unsigned 32bit integers which provides a range from 0
# through 4,294,967,295. This should be sufficient even for region-based approaches.
DTYPE = "uint32"
DTYPE_C = c_uint32

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def create_rankings(ex_mtx: pd.DataFrame, seed=None) -> pd.DataFrame:
    r"""Create genome-wide gene rankings from single-cell expression data.

    This function ranks genes for each cell based on expression levels,
    creating rankings suitable for AUCell gene set enrichment analysis.

    Arguments:
        ex_mtx: Expression matrix with cells as rows and genes as columns (n_cells x n_genes)
        seed (int): Random seed for reproducible ranking (default: None)
    
    Returns:
        pd.DataFrame: Gene rankings matrix with same dimensions as input (n_cells x n_genes)

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

def _rank_sparse_row(row_sparse, n_cols):
    row = row_sparse.toarray().ravel()
    sort_idx = np.lexsort((np.arange(n_cols), -row))
    rank_vals = np.empty(n_cols, dtype=np.int32)
    rank_vals[sort_idx] = np.arange(n_cols)
    return rank_vals

def fast_rank(X_sparse, seed=42):

    rng = np.random.default_rng(seed)
    n_rows, n_cols = X_sparse.shape

    shuffle_order = rng.permutation(n_cols)
    X_shuffled = X_sparse[:, shuffle_order]

    # Handle both sparse matrix and dense array
    results = []
    for i in tqdm(range(n_rows)):
        if hasattr(X_shuffled, 'getrow'):
            # Sparse matrix
            row_result = _rank_sparse_row(X_shuffled.getrow(i), n_cols)
        else:
            # Dense array - create a sparse-like object
            from scipy.sparse import csr_matrix
            row_result = _rank_sparse_row(csr_matrix(X_shuffled[i:i+1]), n_cols)
        results.append(row_result)
    ranks = np.stack(results)

    # Undo column shuffle
    unshuffled_ranks = np.empty_like(ranks)
    unshuffled_ranks[:, shuffle_order] = ranks

    return unshuffled_ranks



def derive_auc_threshold(ex_mtx: csr_matrix, AUC_threshold: float = None) -> pd.DataFrame:
    """
    Derive AUC thresholds for an expression matrix.

    It is important to check that most cells have a substantial fraction of expressed/detected genes in the calculation of
    the AUC.
    
    Arguments:
        ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different genes (n_cells x n_genes).
        AUC_threshold: Specific AUC threshold to include in the quantile calculation. If None, returns default quantiles.
    
    Returns:
        A dataframe with AUC threshold for different quantiles over the number cells: a fraction of 0.01 designates that when using this value as the AUC threshold for 99% of the cells all ranked genes used for AUC calculation will have had a detected expression in the single-cell experiment.

    """
    quantiles = [0.01, 0.05, 0.10, 0.50, 1]
    if AUC_threshold is not None and AUC_threshold not in quantiles:
        quantiles.append(AUC_threshold)
        quantiles.sort()

    # Handle both sparse matrix and dense array
    if hasattr(ex_mtx, 'toarray'):
        # Sparse matrix
        counts = np.count_nonzero(ex_mtx.toarray(), axis=1)
    else:
        # Dense array
        counts = np.count_nonzero(ex_mtx, axis=1)

    return (
        pd.Series(counts).quantile(quantiles)
        / ex_mtx.shape[1]
    )



def _enrichment(
    shared_ro_memory_array, modules, genes, cells, auc_threshold, auc_mtx, offset, gene_overlap_threshold
):
    from ..external.ctxcore.recovery import enrichment4cells

    # The rankings dataframe is properly reconstructed (checked this).
    # Ensure genes are Python native strings for proper type matching with GeneSignature
    df_rnk = pd.DataFrame(
        data=np.frombuffer(shared_ro_memory_array, dtype=DTYPE).reshape(
            len(cells), len(genes)
        ),
        columns=pd.Index([str(g) for g in genes]),
        index=cells,
    )
    # To avoid additional memory burden de resulting AUCs are immediately stored in the output sync. array.
    result_mtx = np.frombuffer(auc_mtx.get_obj(), dtype="d")
    inc = len(cells)
    for idx, module in enumerate(modules):
        # Get enrichment results and reindex to match df_rnk cell order
        enrichment_result = enrichment4cells(df_rnk, module, auc_threshold, gene_overlap_threshold)
        # Reindex to ensure correct cell order
        enrichment_result = enrichment_result.reindex(
            [(cell, module.name) for cell in df_rnk.index]
        )
        result_mtx[
            offset + (idx * inc) : offset + ((idx + 1) * inc)
        ] = enrichment_result.values.ravel(order="C")


def aucell4r(
    df_rnk: pd.DataFrame,
    signatures,
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    num_workers: int = cpu_count(),
    gene_overlap_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.

    Arguments:
        df_rnk: The rank matrix (n_cells x n_genes).
        signatures: The gene signatures or regulons.
        auc_threshold: The fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve.
        noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
        normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
        num_workers: The number of cores to use.
        gene_overlap_threshold: Minimum fraction of genes that must be present (default: 0.80 = 80%).

    Returns:
        A dataframe with the AUCs (n_cells x n_modules).

    """
    from boltons.iterutils import chunked
    from ..external.ctxcore.recovery import enrichment4cells

    if num_workers == 1:
        # Show progress bar for pathway processing
        print(f"Computing AUC scores for {len(signatures)} pathways using single worker...")
        aucs = pd.concat(
            [
                enrichment4cells(
                    df_rnk,
                    module.noweights() if noweights else module,
                    auc_threshold=auc_threshold,
                    gene_overlap_threshold=gene_overlap_threshold,
                )
                for module in tqdm(signatures, desc="Processing pathways")
            ]
        ).unstack("Regulon")
        aucs.columns = aucs.columns.droplevel(0)
    else:
        # Multi-worker processing with progress info
        print(f"Computing AUC scores for {len(signatures)} pathways using {num_workers} workers...")
        # Prefer fork on POSIX to avoid spawn bootstrap issues in interactive environments.
        if os.name == "posix":
            from multiprocessing import get_context
            ctx = get_context("fork")
        else:
            from multiprocessing import get_context
            ctx = get_context("spawn")

        # Decompose the rankings dataframe: the index and columns are shared with the child processes via pickling.
        # Use tolist() to ensure proper type matching in child processes
        genes = df_rnk.columns.tolist()
        cells = df_rnk.index.tolist()
        # The actual rankings are shared directly. This is possible because during a fork from a parent process the child
        # process inherits the memory of the parent process. A RawArray is used instead of a synchronize Array because
        # these rankings are read-only.
        shared_ro_memory_array = ctx.RawArray(DTYPE_C, mul(*df_rnk.shape))
        array = np.frombuffer(shared_ro_memory_array, dtype=DTYPE)
        # Copy the contents of df_rank into this shared memory block using row-major ordering.
        array[:] = df_rnk.values.ravel(order="C")

        # The resulting AUCs are returned via a synchronize array.
        auc_mtx = ctx.Array("d", len(cells) * len(signatures))  # Double precision floats.

        # Convert the modules to modules with uniform weights if necessary.
        if noweights:
            signatures = list(map(lambda m: m.noweights(), signatures))

        # Do the analysis in separate child processes.
        chunk_size = ceil(float(len(signatures)) / num_workers)
        print(f"Splitting {len(signatures)} pathways into {num_workers} chunks of ~{chunk_size} pathways each...")
        
        processes = [
            ctx.Process(
                target=_enrichment,
                args=(
                    shared_ro_memory_array,
                    chunk,
                    genes,
                    cells,
                    auc_threshold,
                    auc_mtx,
                    (chunk_size * len(cells)) * idx,
                    gene_overlap_threshold,
                ),
            )
            for idx, chunk in enumerate(chunked(signatures, chunk_size))
        ]
        
        print("Starting parallel pathway processing...")
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(
                "AUCell parallel processing failed in worker process(es). "
                "Please try num_workers=1, or run from a script guarded by "
                "if __name__ == '__main__'."
            )
        print("Parallel processing completed!")

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
    
    result = aucs / aucs.max(axis=0) if normalize else aucs
    # Keep a stable output layout regardless of processing mode.
    result = result.reindex(index=df_rnk.index, columns=[module.name for module in signatures])
    print(f"AUC calculation completed! Generated scores for {result.shape[1]} pathways across {result.shape[0]} cells.")
    return result


def aucell(
    exp_mtx,
    signatures,
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    seed=None,
    num_workers: int = cpu_count(),
    index=None,
    columns=None,
    gene_overlap_threshold: float = 0.80,
) -> pd.DataFrame:
    r"""Calculate gene signature enrichment scores using AUCell algorithm.

    AUCell quantifies gene set enrichment by calculating the Area Under the Curve (AUC)
    of gene rankings for each cell, providing a robust measure of pathway activity.

    Arguments:
        exp_mtx: Expression matrix (n_cells x n_genes) - DataFrame or sparse matrix
        signatures: Gene signatures or regulons for enrichment analysis
        auc_threshold (float): Fraction of ranked genes to consider for AUC calculation (default: 0.05)
        noweights (bool): Whether to ignore gene weights in signatures (default: False)
        normalize (bool): Whether to normalize AUC values to maximum of 1.0 per signature (default: False)
        seed (int): Random seed for reproducible gene ranking (default: None)
        num_workers (int): Number of CPU cores for parallel processing (default: all cores)
        index: Custom row index for output DataFrame (default: None)
        columns: Custom column names for output DataFrame (default: None)
        gene_overlap_threshold (float): Minimum fraction of genes that must be present (default: 0.80 = 80%)

    Returns:
        pd.DataFrame: AUC enrichment scores with cells as rows and signatures as columns

    """
    # Handle different input types
    if isinstance(exp_mtx, pd.DataFrame):
        # DataFrame input - extract index, columns, and convert to sparse
        original_index = exp_mtx.index
        original_columns = exp_mtx.columns
        matrix_values = exp_mtx.values
        if hasattr(matrix_values, 'toarray'):
            matrix_sparse = matrix_values
        else:
            from scipy.sparse import csr_matrix
            matrix_sparse = csr_matrix(matrix_values)
    else:
        # Sparse matrix input - use provided or default index/columns
        matrix_sparse = exp_mtx
        original_index = index if index is not None else pd.RangeIndex(exp_mtx.shape[0])
        original_columns = columns if columns is not None else pd.RangeIndex(exp_mtx.shape[1])
    
    # Use fast_rank for efficient ranking
    if seed is None:
        seed = 42
    rank_matrix = fast_rank(matrix_sparse, seed=seed)
    
    # Convert rank matrix back to DataFrame format for aucell4r
    df_rnk = pd.DataFrame(
        data=rank_matrix.astype(DTYPE),
        columns=original_columns,
        index=original_index
    )
    
    return aucell4r(
        df_rnk,
        signatures,
        auc_threshold,
        noweights,
        normalize,
        num_workers,
        gene_overlap_threshold,
    )
