"""FastTree wrapper for approximately-maximum-likelihood phylogenetic inference.

Wraps the real ``FastTree`` / ``FastTreeMP`` CLI
(http://www.microbesonline.org/fasttree/). Install via
``conda install -c bioconda fasttree``.

Input : one aligned FASTA (from :func:`omicverse.alignment.mafft`).
Output: one newick tree.

No ``$HOME`` writes — the output path is always resolved from ``output_dir``.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from .._registry import register_function
from ._cli_utils import build_env, ensure_dir, resolve_executable


@register_function(
    aliases=["fasttree", "phylogenetic_tree", "approx_ml_tree"],
    category="alignment",
    description="Infer an approximately-ML phylogenetic tree from an aligned nucleotide FASTA with FastTree.",
    examples=[
        "ov.alignment.fasttree('/run/aligned/aligned.fasta', output_dir='/run/tree')",
    ],
    related=["alignment.mafft", "alignment.build_phylogeny"],
)
def fasttree(
    aligned_fasta: str,
    output_dir: str,
    output_name: str = "tree.nwk",
    model: str = "gtr",
    gamma: bool = True,
    nt: bool = True,
    threads: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
    fasttree_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Run FastTree to infer a phylogenetic tree."""
    if nt and model not in ("gtr", "jc"):
        raise ValueError(f"Unknown nt model {model!r}; use 'gtr' or 'jc'.")

    out_root = ensure_dir(output_dir)
    tree = Path(out_root) / output_name
    log = Path(out_root) / "fasttree.log"

    if not overwrite and tree.exists() and tree.stat().st_size > 0:
        return {"input": str(aligned_fasta), "tree": str(tree), "log": str(log)}

    bin_name = "FastTreeMP" if (threads is not None and threads > 1) else "FastTree"
    exe = resolve_executable(bin_name, fasttree_path, auto_install=auto_install)
    env = build_env(extra_paths=[str(Path(exe).parent)])
    if threads is not None and threads > 1:
        env["OMP_NUM_THREADS"] = str(threads)

    cmd = [exe]
    if nt:
        cmd.append("-nt")
        if model == "gtr":
            cmd.append("-gtr")
    if gamma:
        cmd.append("-gamma")
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    cmd.append(str(aligned_fasta))

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(tree, "w") as out_fh, open(log, "w") as log_fh:
        proc = subprocess.run(
            cmd, stdout=out_fh, stderr=log_fh, env=env, text=True,
        )
    if (proc.returncode != 0
            or not tree.exists()
            or tree.stat().st_size == 0):
        raise RuntimeError(
            f"FastTree failed (exit {proc.returncode}) — see {log}"
        )
    return {"input": str(aligned_fasta), "tree": str(tree), "log": str(log)}
