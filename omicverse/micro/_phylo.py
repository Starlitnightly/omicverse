"""Phylogenetic-tree helpers for ``ov.micro``.

Tools to attach a newick tree to an AnnData (``uns['tree']``) and prune it
to match the ASVs actually present in ``var_names``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import warnings

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.micro._phylo") from exc

from .._registry import register_function


@register_function(
    aliases=["attach_tree", "set_tree", "phylogeny_attach"],
    category="microbiome",
    description="Attach a newick phylogenetic tree to adata.uns['tree'], pruning tips to match adata.var_names (ASVs).",
    examples=[
        "ov.micro.attach_tree(adata, newick='(...)')",
        "ov.micro.attach_tree(adata, tree_path='run/tree/tree.nwk')",
    ],
    related=["alignment.build_phylogeny", "micro.Alpha", "micro.Beta"],
)
def attach_tree(
    adata: "ad.AnnData",
    newick: Optional[str] = None,
    tree_path: Optional[Union[str, Path]] = None,
    prune: bool = True,
    store_key: str = "tree",
    strict: bool = False,
) -> "ad.AnnData":
    """Attach a phylogenetic tree to ``adata.uns[store_key]``.

    Parameters
    ----------
    adata
        The AnnData to annotate (``var_names`` = ASV / OTU ids).
    newick
        The tree as a newick string. Mutually exclusive with ``tree_path``.
    tree_path
        Path to a newick file. Mutually exclusive with ``newick``.
    prune
        When True (default), the tree is restricted to tips that appear in
        ``adata.var_names``. Tips absent from var_names are dropped.
    store_key
        Key under ``adata.uns`` where the newick string is written.
    strict
        If True, raise when any ASV in ``var_names`` has no matching tree
        tip. Default False only warns.
    """
    if (newick is None) == (tree_path is None):
        raise ValueError("Provide exactly one of `newick` or `tree_path`.")
    if tree_path is not None:
        with open(tree_path, "r", encoding="utf-8") as fh:
            newick = fh.read().strip()
    assert newick is not None

    try:
        from ete3 import Tree
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.micro.attach_tree requires ete3 (pip install ete3)."
        ) from exc

    tree = None
    for fmt in (1, 5, 0):
        try:
            tree = Tree(newick, format=fmt)
            break
        except Exception:
            continue
    if tree is None:
        raise ValueError(
            "Failed to parse newick string (tried ete3 formats 1/5/0)."
        )

    tip_names = {leaf.name for leaf in tree.get_leaves()}
    var_names = set(map(str, adata.var_names))

    if prune:
        keep = tip_names & var_names
        if not keep:
            raise ValueError(
                "Zero overlap between tree tips and adata.var_names. "
                "Is the tree built from the same ASVs as the matrix?"
            )
        tree.prune(list(keep), preserve_branch_length=True)
        tip_names = {leaf.name for leaf in tree.get_leaves()}

    missing_in_tree = var_names - tip_names
    if missing_in_tree:
        msg = (
            f"{len(missing_in_tree)} ASV(s) in adata.var_names have no "
            f"matching tip in the tree (first 5: "
            f"{sorted(missing_in_tree)[:5]}). "
            "These ASVs will be ignored by tree-based metrics."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    adata.uns[store_key] = tree.write(format=1)
    adata.uns.setdefault("micro", {})["tree_tips"] = int(len(tip_names))
    return adata
