"""Unit tests for ov.micro.attach_tree."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _ete3_available() -> bool:
    try:
        import ete3  # noqa: F401
    except ImportError:
        return False
    return True


requires_ete3 = pytest.mark.skipif(
    not _ete3_available(),
    reason="ete3 not installed — install the [tests] extra "
           "(`pip install -e '.[tests]'`) to run attach_tree tests.",
)


def _mini_adata(var_ids):
    import anndata as ad
    from scipy import sparse
    X = np.array([[5, 3, 1, 0], [0, 2, 4, 7], [1, 1, 1, 1]], dtype=np.int32)
    return ad.AnnData(
        X=sparse.csr_matrix(X[:, : len(var_ids)]),
        obs=pd.DataFrame(index=[f"S{i}" for i in range(3)]),
        var=pd.DataFrame(index=list(var_ids)),
    )


@requires_ete3
def test_attach_tree_exact_tip_set():
    from omicverse.micro import attach_tree
    adata = _mini_adata(["A", "B", "C"])
    newick = "((A:0.1,B:0.2):0.05,C:0.3);"
    attach_tree(adata, newick=newick)
    assert adata.uns["tree"]
    assert adata.uns["micro"]["tree_tips"] == 3


@requires_ete3
def test_attach_tree_prunes_extra_tips():
    """Tips that aren't in var_names get dropped when prune=True."""
    from omicverse.micro import attach_tree
    adata = _mini_adata(["A", "B"])
    newick = "((A:0.1,B:0.2):0.05,(C:0.15,D:0.15):0.05);"
    attach_tree(adata, newick=newick, prune=True)
    assert adata.uns["micro"]["tree_tips"] == 2


@requires_ete3
def test_attach_tree_warns_on_missing_asv():
    from omicverse.micro import attach_tree
    adata = _mini_adata(["A", "B", "C", "D"])
    # Tree missing ASV D — should warn (but not raise) by default
    newick = "((A:0.1,B:0.2):0.05,C:0.3);"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        attach_tree(adata, newick=newick)
    msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("no matching tip" in m for m in msgs)


@requires_ete3
def test_attach_tree_strict_raises_on_missing():
    from omicverse.micro import attach_tree
    adata = _mini_adata(["A", "B", "C", "D"])
    newick = "((A:0.1,B:0.2):0.05,C:0.3);"
    with pytest.raises(ValueError, match="no matching tip"):
        attach_tree(adata, newick=newick, strict=True)


def test_attach_tree_xor_newick_tree_path():
    from omicverse.micro import attach_tree
    adata = _mini_adata(["A"])
    with pytest.raises(ValueError, match="exactly one"):
        attach_tree(adata)
    with pytest.raises(ValueError, match="exactly one"):
        attach_tree(adata, newick="(A:0.1);", tree_path="whatever.nwk")


@requires_ete3
def test_attach_tree_rejects_disjoint_tree():
    from omicverse.micro import attach_tree
    adata = _mini_adata(["X", "Y"])
    newick = "((A:0.1,B:0.2):0.05,C:0.3);"
    with pytest.raises(ValueError, match="Zero overlap"):
        attach_tree(adata, newick=newick, prune=True)
