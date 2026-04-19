"""Unit tests for ov.alignment phylogeny helpers.

Only the pure-Python glue is covered here — MAFFT / FastTree themselves
are external CLIs. The size-stripping regex and the end-to-end import
structure are the main things to regression-test.
"""
from __future__ import annotations

import pytest


def test_strip_size_annotations_basic():
    from omicverse.alignment.phylogeny import _strip_size_annotations
    newick = "(A;size=1:0.1,B;size=42:0.2):0.0;"
    out = _strip_size_annotations(newick)
    assert out == "(A:0.1,B:0.2):0.0;"


def test_strip_size_annotations_no_size():
    from omicverse.alignment.phylogeny import _strip_size_annotations
    newick = "(A:0.1,B:0.2):0.0;"
    assert _strip_size_annotations(newick) == newick


def test_build_phylogeny_requires_workdir():
    from omicverse.alignment import build_phylogeny
    with pytest.raises(ValueError, match="workdir"):
        build_phylogeny("/tmp/nonexistent.fa")


def test_mafft_mode_validation(tmp_path):
    """Bad MAFFT mode should raise a clear error (before touching the CLI)."""
    from omicverse.alignment import mafft
    fa = tmp_path / "x.fa"
    fa.write_text(">A\nACGT\n>B\nACCT\n")
    with pytest.raises(ValueError, match="MAFFT mode"):
        mafft(str(fa), output_dir=str(tmp_path / "out"), mode="bogus")


def test_fasttree_model_validation(tmp_path):
    from omicverse.alignment import fasttree
    fa = tmp_path / "x.fa"
    fa.write_text(">A\nACGT\n>B\nACCT\n")
    with pytest.raises(ValueError, match="nt model"):
        fasttree(str(fa), output_dir=str(tmp_path / "out"), model="silly")
