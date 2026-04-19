"""End-to-end 16S amplicon pipeline orchestrator.

Chains the real tool wrappers in this package:
  1. :func:`omicverse.alignment.cutadapt`         (optional — primer trimming)
  2. :func:`omicverse.alignment.vsearch.merge_pairs`
  3. :func:`omicverse.alignment.vsearch.filter_quality`
  4. :func:`omicverse.alignment.vsearch.dereplicate`
  5. :func:`omicverse.alignment.vsearch.unoise3`
  6. :func:`omicverse.alignment.vsearch.uchime3_denovo`        (optional, on by default)
  7. :func:`omicverse.alignment.vsearch.sintax`                (if ``db_fasta`` set)
  8. :func:`omicverse.alignment.vsearch.usearch_global`        (build count matrix)

Output: an :class:`anndata.AnnData` with
  - ``X``               — samples × ASVs count matrix (sparse CSR, int32)
  - ``obs``             — one row per sample (``sample`` column)
  - ``var``             — one row per ASV; columns: ``sequence``, taxonomy
                          per rank (``domain`` / ``phylum`` / ``class`` /
                          ``order`` / ``family`` / ``genus`` / ``species``),
                          ``taxonomy`` (``;``-joined), ``sintax_confidence``
                          (per rank, mean bootstrap)
  - ``uns['pipeline']``  — run parameters + per-step output paths

No implicit writes to ``$HOME``. All paths live under the caller-supplied
``workdir`` and ``db_fasta`` / ``db_dir``.
"""
from __future__ import annotations

import gzip
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover - anndata is a hard omicverse dep
    raise ImportError(
        "anndata is required for ov.alignment.amplicon_16s_pipeline"
    ) from exc

from .._registry import register_function
from ._cli_utils import ensure_dir
from . import cutadapt as _cutadapt_mod
from . import vsearch as _vsearch_mod


_SampleTuple = Tuple[str, str, Optional[str]]


_FASTQ_R1_PATTERNS = [
    re.compile(r"^(?P<sample>.+?)_S\d+_L\d+_R1_001\.fastq(\.gz)?$"),
    re.compile(r"^(?P<sample>.+?)_R1_001\.fastq(\.gz)?$"),
    re.compile(r"^(?P<sample>.+?)_R1\.fastq(\.gz)?$"),
    re.compile(r"^(?P<sample>.+?)_1\.fastq(\.gz)?$"),
    re.compile(r"^(?P<sample>.+?)\.R1\.fastq(\.gz)?$"),
]


def _discover_samples(fastq_dir: str) -> List[_SampleTuple]:
    """Auto-pair R1/R2 FASTQs under ``fastq_dir`` by common Illumina naming."""
    d = Path(fastq_dir)
    if not d.exists():
        raise FileNotFoundError(f"fastq_dir not found: {fastq_dir}")

    pairs: Dict[str, Dict[str, str]] = {}
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        matched = False
        for pat in _FASTQ_R1_PATTERNS:
            m = pat.match(name)
            if m:
                sample = m.group("sample")
                pairs.setdefault(sample, {})["fq1"] = str(f)
                matched = True
                break
        if matched:
            continue
        # R2 counterpart
        r2_name = None
        for tag in ("_R2_001", "_R2", "_2", ".R2"):
            if tag in name:
                r2_name = name
                break
        if not r2_name:
            continue
        # Derive sample name from R2 by mirroring patterns
        for pat_src, pat_tgt in [
            ("_R2_001", "_R1_001"),
            ("_R2", "_R1"),
            ("_2.", "_1."),
            (".R2.", ".R1."),
        ]:
            if pat_src in r2_name:
                r1_name = r2_name.replace(pat_src, pat_tgt)
                fq1_path = d / r1_name
                if fq1_path.exists():
                    base = r1_name
                    for p in _FASTQ_R1_PATTERNS:
                        m = p.match(base)
                        if m:
                            sample = m.group("sample")
                            pairs.setdefault(sample, {})["fq2"] = str(f)
                            break
                    break

    out: List[_SampleTuple] = []
    for sample in sorted(pairs):
        info = pairs[sample]
        fq1 = info.get("fq1")
        if not fq1:
            continue
        out.append((sample, fq1, info.get("fq2")))
    if not out:
        raise ValueError(
            f"No R1/R2 FASTQ pairs discovered under {fastq_dir}. "
            "Provide `samples=[(name, fq1, fq2), ...]` explicitly."
        )
    return out


def _parse_sintax_tsv(path: str) -> pd.DataFrame:
    """Parse vsearch ``--sintax --tabbedout`` output.

    Columns produced:
      * ``asv``
      * raw rank columns ``domain`` / ``phylum`` / ``class`` / ``order`` /
        ``family`` / ``genus`` / ``species`` (the filtered, cutoff-applied
        taxonomy from the 4th column of the vsearch tabbedout file)
      * ``taxonomy`` (``;``-joined from those rank columns)
      * ``sintax_confidence`` (min bootstrap across reported ranks, pre-cutoff)
    """
    rank_codes = ["d", "p", "c", "o", "f", "g", "s"]
    rank_names = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    rank_prefix_map = dict(zip(rank_codes, rank_names))

    records = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            # vsearch sintax tabbed format:
            # col0 = query, col1 = raw (with confidences), col2 = strand,
            # col3 = cutoff-filtered taxonomy (no confidences)
            query = parts[0].split(";")[0]  # strip size= annotation
            raw = parts[1] if len(parts) > 1 else ""
            filtered = parts[3] if len(parts) > 3 else ""
            rec = {
                "asv": query,
                "sintax_raw": raw,
            }
            for rn in rank_names:
                rec[rn] = ""
            if filtered:
                for token in filtered.split(","):
                    token = token.strip()
                    if len(token) < 2 or token[1] != ":":
                        continue
                    code = token[0]
                    rn = rank_prefix_map.get(code)
                    if rn:
                        rec[rn] = token[2:]
            confs: List[float] = []
            if raw:
                for token in raw.split(","):
                    token = token.strip()
                    m = re.match(r"([a-z]):([^()]+)\(([\d.]+)\)", token)
                    if m:
                        confs.append(float(m.group(3)))
            rec["sintax_confidence"] = min(confs) if confs else np.nan
            tax_parts = [f"{rank_prefix_map[c]}={rec[rank_prefix_map[c]]}"
                         for c in rank_codes if rec[rank_prefix_map[c]]]
            rec["taxonomy"] = ";".join(tax_parts)
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        df = pd.DataFrame(columns=["asv", "sintax_raw"] + rank_names + ["sintax_confidence", "taxonomy"])
    return df.set_index("asv")


def _parse_asv_fasta(path: str) -> pd.Series:
    """Read ASV FASTA → Series ``{asv_id: sequence}`` (strip size= from header)."""
    ids: List[str] = []
    seqs: List[str] = []
    current = None
    buf: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                if current is not None:
                    ids.append(current)
                    seqs.append("".join(buf))
                current = line[1:].strip().split(";")[0]
                buf = []
            else:
                buf.append(line.strip())
    if current is not None:
        ids.append(current)
        seqs.append("".join(buf))
    return pd.Series(seqs, index=ids, name="sequence")


def _load_otutab(path: str) -> pd.DataFrame:
    """Read vsearch ``otutab.tsv`` (first col = ASV id, rest = samples)."""
    df = pd.read_csv(path, sep="\t", comment=None)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "asv"}).set_index("asv")
    df.columns = [str(c) for c in df.columns]
    return df


@register_function(
    aliases=[
        "build_amplicon_anndata", "amplicon_to_anndata",
        "vsearch_to_anndata", "asv_anndata",
    ],
    category="alignment",
    description="Assemble a samples × ASVs AnnData from vsearch stepwise outputs "
                "(otutab TSV + ASV FASTA + optional SINTAX TSV). Same schema as the "
                "one-shot amplicon_16s_pipeline.",
    examples=[
        "adata = ov.alignment.build_amplicon_anndata("
        "otutab_tsv='otutab.tsv', asv_fasta='asvs.fasta', "
        "sintax_tsv='sintax.tsv', sample_metadata=meta)"
    ],
    related=[
        "alignment.amplicon_16s_pipeline", "alignment.vsearch",
    ],
)
def build_amplicon_anndata(
    otutab_tsv: str,
    asv_fasta: str,
    sintax_tsv: Optional[str] = None,
    sample_metadata: Optional[pd.DataFrame] = None,
    sample_order: Optional[Sequence[str]] = None,
):
    """Compose :class:`anndata.AnnData` from vsearch stepwise outputs.

    Parameters
    ----------
    otutab_tsv
        Path to the ``--otutabout`` TSV from :func:`vsearch.usearch_global`.
        First column = ASV id, remaining columns = sample ids.
    asv_fasta
        Path to the ASV centroid FASTA (output of :func:`vsearch.unoise3` or
        the non-chimera FASTA from :func:`vsearch.uchime3_denovo`).
    sintax_tsv
        Optional path to the vsearch ``--sintax --tabbedout`` TSV. When
        supplied, 7-rank taxonomy columns (``domain`` / ``phylum`` / ``class``
        / ``order`` / ``family`` / ``genus`` / ``species``), the ``;``-joined
        ``taxonomy`` string and ``sintax_confidence`` are written into ``var``.
    sample_metadata
        Optional DataFrame indexed by sample id; merged into ``obs``.
    sample_order
        Optional list of sample ids to enforce row order in ``obs`` / ``X``.
        Defaults to the column order of the otutab TSV.

    Returns
    -------
    anndata.AnnData
        ``X`` is a ``scipy.sparse.csr_matrix`` of int32 counts (samples ×
        ASVs). Same schema as :func:`amplicon_16s_pipeline`.
    """
    from scipy import sparse

    counts = _load_otutab(otutab_tsv)                         # ASVs x samples
    seqs = _parse_asv_fasta(asv_fasta)
    tax_df = _parse_sintax_tsv(sintax_tsv) if sintax_tsv else None

    all_asvs = sorted(set(counts.index) | set(seqs.index))
    counts = counts.reindex(all_asvs).fillna(0).astype(np.int32)

    if sample_order is None:
        sample_order = list(counts.columns)
    else:
        sample_order = list(sample_order)
        missing = [s for s in sample_order if s not in counts.columns]
        if missing:
            raise ValueError(f"sample_order has ids missing from otutab: {missing}")

    X = counts[sample_order].T.values

    var = pd.DataFrame(index=pd.Index(all_asvs, name="asv"))
    var["sequence"] = seqs.reindex(all_asvs).fillna("")
    rank_cols = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    if tax_df is not None:
        tax_df = tax_df.reindex(all_asvs)
        for col in rank_cols + ["taxonomy", "sintax_raw"]:
            var[col] = tax_df[col] if col in tax_df.columns else ""
        var["sintax_confidence"] = (
            tax_df["sintax_confidence"] if "sintax_confidence" in tax_df.columns
            else np.nan
        )
    else:
        for col in rank_cols + ["taxonomy"]:
            var[col] = ""
        var["sintax_confidence"] = np.nan
        var["sintax_raw"] = ""

    obs = pd.DataFrame(index=pd.Index(sample_order, name="sample"))
    obs["sample"] = sample_order
    if sample_metadata is not None:
        meta = sample_metadata.copy()
        meta.index = meta.index.astype(str)
        obs = obs.join(meta, how="left")

    return ad.AnnData(
        X=sparse.csr_matrix(X, dtype=np.int32),
        obs=obs,
        var=var,
    )


@register_function(
    aliases=[
        "amplicon_16s_pipeline", "16s_pipeline", "pipeline_16s",
        "amplicon_pipeline", "asv_pipeline",
    ],
    category="alignment",
    description="Run the end-to-end 16S amplicon pipeline "
                "(cutadapt + vsearch merge/filter/derep/unoise3/uchime3/sintax/otutab) and return AnnData.",
    examples=[
        "adata = ov.alignment.amplicon_16s_pipeline("
        "fastq_dir='raw/', workdir='run1', db_fasta='/scratch/.../rdp_16s_v18.fa.gz')"
    ],
    related=[
        "alignment.cutadapt", "alignment.vsearch",
        "alignment.fetch_silva", "alignment.fetch_rdp",
    ],
)
def amplicon_16s_pipeline(
    fastq_dir: Optional[str] = None,
    samples: Optional[Sequence[_SampleTuple]] = None,
    workdir: str = "amplicon_16s_run",
    db_fasta: Optional[str] = None,
    *,
    primer_fwd: Optional[str] = None,
    primer_rev: Optional[str] = None,
    backend: str = "vsearch",
    threads: int = 4,
    jobs: Optional[int] = None,
    # vsearch knobs
    merge_max_diffs: int = 10,
    merge_min_overlap: int = 16,
    filter_max_ee: float = 1.0,
    filter_min_len: int = 0,
    filter_max_len: int = 0,
    derep_min_uniq: int = 2,
    unoise_alpha: float = 2.0,
    unoise_minsize: int = 2,
    chimera_removal: bool = True,
    otutab_identity: float = 0.97,
    sintax_cutoff: float = 0.8,
    sintax_strand: str = "both",
    # metadata
    sample_metadata: Optional[pd.DataFrame] = None,
    overwrite: bool = False,
):
    """Run the full 16S amplicon pipeline.

    Parameters
    ----------
    fastq_dir
        Directory containing paired Illumina FASTQs. Samples are
        auto-discovered by R1/R2 naming. Mutually exclusive with ``samples``.
    samples
        Explicit list of ``(sample, fq1, fq2)`` tuples (fq2 may be None for
        single-end). Mutually exclusive with ``fastq_dir``.
    workdir
        Root directory for all intermediate files. No ``$HOME`` fallback.
    db_fasta
        Path to a SINTAX-formatted 16S reference FASTA (``.fa`` or ``.fa.gz``).
        If ``None``, taxonomy assignment is skipped.
    primer_fwd, primer_rev
        PCR primer sequences. When both are provided, :func:`cutadapt` runs
        first; otherwise primer trimming is skipped (e.g. the mothur MiSeq
        SOP test dataset ships with primers already removed).
    backend
        Currently only ``'vsearch'`` is implemented. The ``'dada2'`` /
        ``'emu'`` / ``'qiime2'`` backends raise ``NotImplementedError`` —
        stubs exist to keep the API stable.
    threads, jobs
        CPU parallelism.
    overwrite
        If True, re-run each step regardless of existing outputs.

    Returns
    -------
    anndata.AnnData
        Samples × ASVs matrix with taxonomy / sequence / confidence in
        ``var``. Sample metadata (if provided) is merged into ``obs``.
    """
    if backend != "vsearch":
        raise NotImplementedError(
            f"backend={backend!r} is not implemented yet. "
            "Use backend='vsearch' (the UNOISE3-based default)."
        )
    if not fastq_dir and not samples:
        raise ValueError("Provide either `fastq_dir` or `samples`.")
    if fastq_dir and samples:
        raise ValueError("Specify only one of `fastq_dir` or `samples`.")

    workdir = str(Path(workdir).expanduser().resolve())
    ensure_dir(workdir)

    if fastq_dir:
        sample_list = _discover_samples(fastq_dir)
    else:
        sample_list = [tuple(s) for s in samples]  # type: ignore[arg-type]

    paths: Dict[str, str] = {"workdir": workdir}

    # --- Step 1: optional primer trim ---------------------------------
    current_samples: List[_SampleTuple] = list(sample_list)
    if primer_fwd:
        trim_dir = str(Path(workdir) / "cutadapt")
        trim_results = _cutadapt_mod.cutadapt(
            samples=current_samples,
            primer_fwd=primer_fwd,
            primer_rev=primer_rev,
            output_dir=trim_dir,
            threads=threads,
            jobs=jobs,
            overwrite=overwrite,
        )
        if isinstance(trim_results, dict):
            trim_results = [trim_results]
        new_samples: List[_SampleTuple] = []
        for r in trim_results:
            t2 = r.get("trim2") or None
            new_samples.append((r["sample"], r["trim1"], t2 if t2 else None))
        current_samples = new_samples
        paths["cutadapt"] = trim_dir

    # --- Step 2: merge pairs -----------------------------------------
    merge_dir = str(Path(workdir) / "merged")
    merge_results = _vsearch_mod.merge_pairs(
        current_samples,
        output_dir=merge_dir,
        max_diffs=merge_max_diffs,
        min_overlap=merge_min_overlap,
        threads=threads,
        jobs=jobs,
        overwrite=overwrite,
    )
    paths["merged"] = merge_dir

    # --- Step 3: quality filter --------------------------------------
    filt_dir = str(Path(workdir) / "filtered")
    filt_results = _vsearch_mod.filter_quality(
        merge_results,
        output_dir=filt_dir,
        max_ee=filter_max_ee,
        min_len=filter_min_len,
        max_len=filter_max_len,
        threads=threads,
        jobs=jobs,
        overwrite=overwrite,
    )
    paths["filtered"] = filt_dir

    # --- Step 4: dereplicate (concat + unique) -----------------------
    derep_dir = str(Path(workdir) / "derep")
    derep = _vsearch_mod.dereplicate(
        filt_results,
        output_dir=derep_dir,
        min_uniq=derep_min_uniq,
        threads=threads,
        overwrite=overwrite,
    )
    paths.update({
        "derep_combined": derep["combined"],
        "derep_uniques": derep["uniques"],
    })

    # --- Step 5: UNOISE3 denoise -------------------------------------
    asv_dir = str(Path(workdir) / "asv")
    unoise = _vsearch_mod.unoise3(
        derep["uniques"],
        output_dir=asv_dir,
        alpha=unoise_alpha,
        minsize=unoise_minsize,
        threads=threads,
        overwrite=overwrite,
    )
    asv_fasta = unoise["asv"]
    paths["asv_pre_chimera"] = asv_fasta

    # --- Step 6: chimera removal -------------------------------------
    if chimera_removal:
        nochim = _vsearch_mod.uchime3_denovo(
            asv_fasta,
            output_dir=asv_dir,
            overwrite=overwrite,
        )
        asv_fasta = nochim["asv"]
        paths["chimeras"] = nochim["chimeras"]
    paths["asv"] = asv_fasta

    # --- Step 7: taxonomy --------------------------------------------
    tax_df: Optional[pd.DataFrame] = None
    if db_fasta:
        tax_dir = str(Path(workdir) / "taxonomy")
        sintax_out = _vsearch_mod.sintax(
            asv_fasta,
            db_fasta=db_fasta,
            output_dir=tax_dir,
            cutoff=sintax_cutoff,
            strand=sintax_strand,
            threads=threads,
            overwrite=overwrite,
        )
        tax_df = _parse_sintax_tsv(sintax_out["tsv"])
        paths["taxonomy_tsv"] = sintax_out["tsv"]

    # --- Step 8: count matrix (sample × ASV) -------------------------
    otutab_dir = str(Path(workdir) / "otutab")
    otutab = _vsearch_mod.usearch_global(
        derep["combined"],
        asv_fasta,
        output_dir=otutab_dir,
        identity=otutab_identity,
        threads=threads,
        overwrite=overwrite,
    )
    paths["otutab_tsv"] = otutab["otutab"]

    # --- Assemble AnnData --------------------------------------------
    # honour the original sample order (not whatever order otutab emits)
    ordered = [s for (s, _f1, _f2) in sample_list]
    adata = build_amplicon_anndata(
        otutab_tsv=otutab["otutab"],
        asv_fasta=asv_fasta,
        sintax_tsv=paths.get("taxonomy_tsv"),
        sample_metadata=sample_metadata,
        sample_order=ordered,
    )
    adata.uns["pipeline"] = {
        "tool": "vsearch UNOISE3 + SINTAX",
        "backend": backend,
        "paths": paths,
        "params": {
            "primer_fwd": primer_fwd,
            "primer_rev": primer_rev,
            "merge_max_diffs": merge_max_diffs,
            "merge_min_overlap": merge_min_overlap,
            "filter_max_ee": filter_max_ee,
            "derep_min_uniq": derep_min_uniq,
            "unoise_alpha": unoise_alpha,
            "unoise_minsize": unoise_minsize,
            "chimera_removal": chimera_removal,
            "otutab_identity": otutab_identity,
            "sintax_cutoff": sintax_cutoff,
            "sintax_strand": sintax_strand,
            "db_fasta": db_fasta,
            "threads": threads,
        },
    }
    return adata
