"""End-to-end phylogeny builder: ASV FASTA → MAFFT → FastTree → newick string.

Chains :func:`omicverse.alignment.mafft` and :func:`omicverse.alignment.fasttree`
and returns both the on-disk paths and the in-memory newick text ready to
drop into ``adata.uns['tree']``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

from .._registry import register_function
from ._cli_utils import ensure_dir
# Import the functions directly — `from . import mafft` would resolve to the
# re-exported function (not the submodule) after __init__ runs, because
# alignment/__init__.py does `from .mafft import mafft`.
from .mafft import mafft as _mafft_fn
from .fasttree import fasttree as _fasttree_fn


_SIZE_ANNOT = re.compile(r";size=\d+")


def _strip_size_annotations(newick: str) -> str:
    """Remove ``;size=NNN`` tokens that vsearch leaves on ASV ids.

    These annotations survive MAFFT and FastTree into the newick tip labels;
    because the semicolon is the newick statement terminator, downstream
    parsers (ete3, unifrac) choke on them. ``size=`` only encodes
    dereplication multiplicity — downstream diversity metrics use the count
    matrix for abundance, so it's safe to drop here.
    """
    return _SIZE_ANNOT.sub("", newick)


@register_function(
    aliases=["build_phylogeny", "build_tree", "asv_tree", "phylogeny"],
    category="alignment",
    description="Build a phylogenetic tree from ASV centroids: MAFFT align → FastTree infer. Returns both the tree path and the newick string.",
    examples=[
        "tree = ov.alignment.build_phylogeny(asvs_fasta, workdir='/run/phylogeny')",
    ],
    related=["alignment.mafft", "alignment.fasttree"],
)
def build_phylogeny(
    asvs_fasta: str,
    workdir: Optional[str] = None,
    *,
    mafft_mode: str = "auto",
    fasttree_model: str = "gtr",
    gamma: bool = True,
    mafft_threads: int = 4,
    fasttree_threads: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Build a phylogenetic tree end-to-end.

    Returns ``{"aligned": ..., "tree": ..., "newick": "<tree-text>", ...}``.
    The newick string has had vsearch's ``;size=N`` annotations stripped so
    that ete3 / unifrac can parse tip labels directly.
    """
    if not workdir:
        raise ValueError(
            "`workdir` is required. omicverse never writes phylogeny "
            "intermediates to an implicit location. Example: "
            "workdir='/scratch/<user>/phylogeny_run1'."
        )
    workdir = str(Path(workdir).expanduser().resolve())
    ensure_dir(workdir)
    aligned_dir = str(Path(workdir) / "aligned")
    tree_dir = str(Path(workdir) / "tree")

    aln = _mafft_fn(
        asvs_fasta,
        output_dir=aligned_dir,
        mode=mafft_mode,
        threads=mafft_threads,
        overwrite=overwrite,
    )
    tr = _fasttree_fn(
        aln["aligned"],
        output_dir=tree_dir,
        model=fasttree_model,
        gamma=gamma,
        nt=True,
        threads=fasttree_threads,
        overwrite=overwrite,
    )
    with open(tr["tree"], "r", encoding="utf-8") as fh:
        newick = fh.read().strip()
    newick = _strip_size_annotations(newick)
    # Overwrite the on-disk tree so ete3 / unifrac can read it directly.
    with open(tr["tree"], "w", encoding="utf-8") as fh:
        fh.write(newick + "\n")
    return {
        "aligned": aln["aligned"],
        "tree": tr["tree"],
        "newick": newick,
        "aligned_log": aln["log"],
        "tree_log": tr["log"],
    }
