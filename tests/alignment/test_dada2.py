"""Tests for ov.alignment.dada2 — pure-Python helpers only (no pydada2 run)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_dada2_pipeline_requires_workdir():
    from omicverse.alignment import dada2_pipeline
    with pytest.raises(ValueError, match="workdir"):
        dada2_pipeline([("S1", "a.fq.gz", "b.fq.gz")])


def test_dada2_pipeline_rejects_empty_samples():
    from omicverse.alignment import dada2_pipeline
    with pytest.raises(ValueError, match="samples"):
        dada2_pipeline([], workdir="/tmp/x")


def test_seqtab_to_anndata_roundtrip():
    """Synthetic sample×sequence → AnnData with ASVn ids + taxonomy columns."""
    from omicverse.alignment.dada2 import _seqtab_to_anndata
    seqs = ["ACGTAC", "CCGTTT", "GGGAAA"]
    samples = ["S1", "S2", "S3"]
    st = pd.DataFrame(
        [[5, 3, 0], [0, 2, 7], [1, 1, 1]],
        index=samples, columns=seqs,
    )
    adata = _seqtab_to_anndata(st, sample_order=samples)
    assert adata.shape == (3, 3)
    assert list(adata.var_names) == ["ASV1", "ASV2", "ASV3"]
    assert list(adata.var["sequence"]) == seqs
    # taxonomy cols are there but empty
    for col in ("phylum", "class", "genus", "taxonomy"):
        assert col in adata.var.columns
        assert (adata.var[col] == "").all()
    import scipy.sparse as sp
    assert sp.issparse(adata.X)
    assert adata.X.dtype == np.int32


def test_backend_dada2_in_allowed_list():
    """amplicon_16s_pipeline now accepts backend='dada2' without NotImpl."""
    from omicverse.alignment import amplicon_16s_pipeline
    with pytest.raises(ValueError, match="workdir"):   # passes backend check, fails on workdir
        amplicon_16s_pipeline(
            samples=[("S1", "x.fq.gz", "y.fq.gz")],
            backend="dada2",
        )


def test_backend_emu_still_not_implemented():
    from omicverse.alignment import amplicon_16s_pipeline
    with pytest.raises(NotImplementedError, match="emu"):
        amplicon_16s_pipeline(
            samples=[("S1", "x.fq.gz", "y.fq.gz")],
            workdir="/tmp/x",
            backend="emu",
        )
