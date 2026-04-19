"""MAFFT wrapper for multiple sequence alignment.

Wraps the real ``mafft`` CLI (https://mafft.cbrc.jp/alignment/software/).
Install via ``conda install -c bioconda mafft``.

Input:  one FASTA with the sequences to align (typically ASV centroids
         produced by :func:`omicverse.alignment.vsearch.unoise3` /
         ``uchime3_denovo``).
Output: one aligned FASTA (same ids, gaps inserted).

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
    aliases=["mafft", "msa", "multiple_sequence_alignment"],
    category="alignment",
    description="Multiple sequence alignment of ASV centroids with MAFFT (typically used as input to FastTree / IQ-TREE for a phylogenetic tree).",
    examples=[
        "ov.alignment.mafft('/run/asv/asvs.fasta', output_dir='/run/aligned')",
    ],
    related=["alignment.fasttree", "alignment.build_phylogeny"],
)
def mafft(
    input_fasta: str,
    output_dir: str,
    output_name: str = "aligned.fasta",
    mode: str = "auto",
    threads: int = 4,
    extra_args: Optional[Sequence[str]] = None,
    mafft_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Run MAFFT multiple sequence alignment.

    Parameters
    ----------
    input_fasta
        Path to unaligned FASTA (ASV centroids, 16S sequences, …).
    output_dir
        Directory for the aligned FASTA + log. Required (no ``$HOME``).
    output_name
        Filename for the aligned FASTA inside ``output_dir``.
    mode
        MAFFT strategy flag: ``'auto'`` (default; ``--auto`` picks FFT-NS-1 /
        L-INS-i depending on size), ``'fftns'``, ``'linsi'``, ``'einsi'``,
        ``'ginsi'``, or ``'retree1'`` (fastest).
    threads
        Passed as ``--thread``. MAFFT supports ``--thread -1`` for auto-detect.
    extra_args
        Appended verbatim to the mafft command line (coerced to ``str``).
    mafft_path
        Explicit path to the ``mafft`` binary.
    auto_install
        Try ``conda install -c bioconda mafft`` if absent.
    overwrite
        Re-run even if the aligned file already exists.

    Returns
    -------
    ``{"input": …, "aligned": …, "log": …}`` — absolute paths.
    """
    out_root = ensure_dir(output_dir)
    aligned = Path(out_root) / output_name
    log = Path(out_root) / "mafft.log"

    if not overwrite and aligned.exists() and aligned.stat().st_size > 0:
        return {"input": str(input_fasta), "aligned": str(aligned), "log": str(log)}

    mafft_bin = resolve_executable("mafft", mafft_path, auto_install=auto_install)
    env = build_env(extra_paths=[str(Path(mafft_bin).parent)])

    mode_flags = {
        "auto":    ["--auto"],
        "fftns":   ["--retree", "2"],
        "linsi":   ["--localpair", "--maxiterate", "1000"],
        "einsi":   ["--genafpair", "--maxiterate", "1000"],
        "ginsi":   ["--globalpair", "--maxiterate", "1000"],
        "retree1": ["--retree", "1"],
    }
    if mode not in mode_flags:
        raise ValueError(
            f"Unknown MAFFT mode {mode!r}; use one of {sorted(mode_flags)}"
        )
    cmd = [mafft_bin, "--thread", str(threads), *mode_flags[mode]]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    cmd.append(str(input_fasta))

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(aligned, "w") as out_fh, open(log, "w") as log_fh:
        proc = subprocess.run(
            cmd, stdout=out_fh, stderr=log_fh, env=env, text=True,
        )
    if (proc.returncode != 0
            or not aligned.exists()
            or aligned.stat().st_size == 0):
        raise RuntimeError(
            f"mafft failed (exit {proc.returncode}) — see {log}"
        )
    return {"input": str(input_fasta), "aligned": str(aligned), "log": str(log)}
