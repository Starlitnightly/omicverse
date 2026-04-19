"""Unit tests for the pure-Python helpers in ov.alignment.amplicon_16s.

These do NOT invoke vsearch/cutadapt subprocesses — they only exercise the
parsers, the sample-discovery regex logic, and the AnnData assembly.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _discover_samples
# ---------------------------------------------------------------------------


def _mk_fastq(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.write_text("@r1\nNNN\n+\n!!!\n")
    return p


def test_discover_samples_standard_illumina_naming(tmp_path):
    from omicverse.alignment.amplicon_16s import _discover_samples
    _mk_fastq(tmp_path, "S1_S10_L001_R1_001.fastq")
    _mk_fastq(tmp_path, "S1_S10_L001_R2_001.fastq")
    _mk_fastq(tmp_path, "S2_S20_L001_R1_001.fastq")
    _mk_fastq(tmp_path, "S2_S20_L001_R2_001.fastq")
    out = _discover_samples(str(tmp_path))
    names = sorted(s[0] for s in out)
    assert names == ["S1", "S2"]
    for _, f1, f2 in out:
        assert f1.endswith("_R1_001.fastq")
        assert f2 and f2.endswith("_R2_001.fastq")


def test_discover_samples_rejects_underscore_two_false_positive(tmp_path):
    """`sample_group2_R1_001.fastq` contains `_2` but is NOT an R2 file."""
    from omicverse.alignment.amplicon_16s import _discover_samples
    # This is a real sample name with "_2" in it
    _mk_fastq(tmp_path, "sample_group2_R1_001.fastq")
    _mk_fastq(tmp_path, "sample_group2_R2_001.fastq")
    out = _discover_samples(str(tmp_path))
    assert len(out) == 1
    assert out[0][0] == "sample_group2"


def test_discover_samples_empty_dir(tmp_path):
    from omicverse.alignment.amplicon_16s import _discover_samples
    with pytest.raises(ValueError, match="No R1/R2 FASTQ pairs"):
        _discover_samples(str(tmp_path))


# ---------------------------------------------------------------------------
# _parse_sintax_tsv
# ---------------------------------------------------------------------------


def test_parse_sintax_tsv_full_row(tmp_path):
    from omicverse.alignment.amplicon_16s import _parse_sintax_tsv
    tsv = tmp_path / "sintax.tsv"
    tsv.write_text(
        # ASV1: full 7-rank call
        "ASV1;size=10\td:Bacteria(1.00),p:Firmicutes(0.99),"
        "c:Clostridia(0.95),o:Clostridiales(0.92),"
        "f:Lachnospiraceae(0.88),g:Blautia(0.70),s:Blautia_sp(0.40)\t"
        "+\t"
        "d:Bacteria,p:Firmicutes,c:Clostridia,o:Clostridiales,"
        "f:Lachnospiraceae,g:Blautia\n"
    )
    df = _parse_sintax_tsv(str(tsv))
    assert list(df.index) == ["ASV1"]
    row = df.loc["ASV1"]
    assert row["phylum"] == "Firmicutes"
    assert row["genus"] == "Blautia"
    assert row["species"] == ""                    # not passed by cutoff
    assert row["taxonomy"].startswith("domain=Bacteria;phylum=Firmicutes")
    # confidence = min bootstrap (conservative)
    assert row["sintax_confidence"] == pytest.approx(0.40, abs=1e-6)


def test_parse_sintax_tsv_empty_file(tmp_path):
    from omicverse.alignment.amplicon_16s import _parse_sintax_tsv
    tsv = tmp_path / "sintax.tsv"
    tsv.write_text("")
    df = _parse_sintax_tsv(str(tsv))
    assert df.empty
    for col in ("domain", "phylum", "genus", "taxonomy", "sintax_confidence"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# _parse_asv_fasta + _load_otutab + build_amplicon_anndata
# ---------------------------------------------------------------------------


def test_parse_asv_fasta_strips_size_annotation(tmp_path):
    from omicverse.alignment.amplicon_16s import _parse_asv_fasta
    fa = tmp_path / "asv.fa"
    fa.write_text(">ASV1;size=10\nACGTACGT\n>ASV2;size=4\nACCCGGGT\n")
    seqs = _parse_asv_fasta(str(fa))
    assert list(seqs.index) == ["ASV1", "ASV2"]
    assert seqs["ASV1"] == "ACGTACGT"


def test_load_otutab_strips_hash_header(tmp_path):
    from omicverse.alignment.amplicon_16s import _load_otutab
    tsv = tmp_path / "otutab.tsv"
    tsv.write_text("#OTU ID\tS1\tS2\nASV1\t5\t1\nASV2\t0\t3\n")
    df = _load_otutab(str(tsv))
    assert df.index.name == "asv"
    assert list(df.columns) == ["S1", "S2"]
    assert df.loc["ASV1", "S1"] == 5
    assert df.loc["ASV2", "S2"] == 3


def test_build_amplicon_anndata_shape_and_var(tmp_path):
    from omicverse.alignment import build_amplicon_anndata

    otu = tmp_path / "otu.tsv"
    otu.write_text("#OTU ID\tS1\tS2\tS3\nASV1\t5\t1\t0\nASV2\t0\t3\t4\n")
    fa = tmp_path / "asv.fa"
    fa.write_text(">ASV1;size=6\nACG\n>ASV2;size=7\nTTT\n")
    sx = tmp_path / "sintax.tsv"
    sx.write_text(
        "ASV1;size=6\td:Bacteria(1.0),p:Firmicutes(0.99)\t+\t"
        "d:Bacteria,p:Firmicutes\n"
        "ASV2;size=7\td:Bacteria(1.0),p:Bacteroidetes(0.95)\t+\t"
        "d:Bacteria,p:Bacteroidetes\n"
    )
    meta = pd.DataFrame({"group": ["A", "B", "A"]}, index=["S1", "S2", "S3"])

    adata = build_amplicon_anndata(
        otutab_tsv=str(otu),
        asv_fasta=str(fa),
        sintax_tsv=str(sx),
        sample_metadata=meta,
        sample_order=["S1", "S2", "S3"],
    )
    assert adata.shape == (3, 2)
    # sparse int32 counts
    import scipy.sparse as sp
    assert sp.issparse(adata.X)
    assert adata.X.dtype == np.int32
    # counts preserved
    dense = adata.X.toarray()
    assert dense[0, 0] == 5
    assert dense[1, 1] == 3
    # taxonomy populated
    assert adata.var.loc["ASV1", "phylum"] == "Firmicutes"
    assert adata.var.loc["ASV2", "phylum"] == "Bacteroidetes"
    # obs metadata merged
    assert list(adata.obs["group"]) == ["A", "B", "A"]


def test_build_amplicon_anndata_no_taxonomy(tmp_path):
    from omicverse.alignment import build_amplicon_anndata
    otu = tmp_path / "otu.tsv"
    otu.write_text("#OTU ID\tS1\tS2\nASV1\t5\t1\n")
    fa = tmp_path / "asv.fa"
    fa.write_text(">ASV1;size=6\nACG\n")
    adata = build_amplicon_anndata(
        otutab_tsv=str(otu),
        asv_fasta=str(fa),
        sintax_tsv=None,
    )
    assert adata.shape == (2, 1)
    assert adata.var.loc["ASV1", "phylum"] == ""
    assert pd.isna(adata.var.loc["ASV1", "sintax_confidence"])
