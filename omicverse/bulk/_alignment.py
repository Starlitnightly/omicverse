# omicverse/bulk/alignment.py
# Contributor: Zhi Luo

"""
Alignment pipeline for bulk RNA-seq in OmicVerse.

This class is a thin, composable wrapper around existing step modules in your repo:
- sra_prefetch / sra_fasterq   : SRA download & conversion
- qc_fastp / qc_tools          : FASTQ QC & trimming
- count_step / count_tools     : gene-level counting via featureCounts
- (optional) your aligner step : HISAT2/STAR wrapper if available
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence, Tuple, Optional, Dict, Any

# Import your existing step modules (assumes they are importable in the package env)
# If your modules live at project root, adjust relative imports accordingly.
try:
    from . import geo_meta_fetcher as _geo
except Exception:
    import geo_meta_fetcher as _geo

try:
    from . import entrez_direct as _ed
except Exception:
    import entrez_direct as _ed

try:
    from . import sra_prefetch as _sra_prefetch
except Exception:
    import sra_prefetch as _sra_prefetch
    
try:
    from . import sra_fasterq as _sra_fasterq
except Exception:
    import sra_fasterq as _sra_fasterq

try:
    from . import qc_fastp as _qc_fastp
except Exception:
    import qc_fastp as _qc_fastp

try:
    from . import star_step as _star_step
except Exception:
    import star_step as _star_step

try:
    from . import count_step as _count_step
except Exception:
    import count_step as _count_step

try: 
    from . import tools_check as _tools_check
except Exception:
    import tools_check as _tools_check

@dataclass
class AlignmentConfig:
    # IO roots
    work_root: Path = Path("work")
    meta_root: Path = Path("work/meta")
    prefetch_root: Path = Path("work/prefetch")
    fasterq_root: Path = Path("work/fasterq")
    qc_root: Path = Path("work/fastp")
    align_root: Path = Path("work/align")     # placeholder if you later add HISAT2/STAR
    counts_root: Path = Path("work/counts")

    star_index_root: Path = field(default_factory=lambda: Path("index"))        # 索引根目录（和 star_tools 设定一致）
    star_align_root: Path  = field(default_factory=lambda: Path("work/star"))   # STAR 输出根目录
    

    # Resources
    threads: int = 16
    memory: str = "8G"
    genome: Literal["human", "mouse", "custom"] = "human"
    gtf: Optional[Path] = None                 # if provided, overrides genome GTF discovery

    # Behavior
    gzip_fastq: bool = True
    fastp_enabled: bool = True
    simple_counts: bool = True                 # only gene_id,count

    # Metadata
    by: Literal["auto","srr","accession"] = "auto"


class Alignment:
    """
    A cohesive, user-facing API for the bulk RNA-seq alignment pipeline.
    Each step delegates to your existing step modules, preserving their behavior,
    while unifying inputs/outputs and logging.
    """

    def __init__(self, config: AlignmentConfig | None = None):
        self.cfg = config or AlignmentConfig()
        # normalize paths
        for p in [
            self.cfg.work_root, self.cfg.meta_root,self.cfg.prefetch_root, self.cfg.fasterq_root,
            self.cfg.qc_root, self.cfg.align_root, self.cfg.counts_root,self.cfg.star_index_root,self.cfg.star_align_root
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)

     # ---------- Fetch Metadata ----------
    def fetch_metadata(
        self,
        accession: str,
        meta_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        organism_filter: Optional[str] = None,   # 例如 "Homo sapiens"
        layout_filter: Optional[str] = None,     # "PAIRED" / "SINGLE"
    ):
        """
        给一个 GEO accession（GSE/GSM），抓取 SOFT→保存 meta JSON，
        再走 EDirect 生成 RunInfo CSV，并返回 SRR 列表与路径。
        """
        _tools_check.check()
        # 1) 目录设置
        meta_root = Path(meta_dir) if meta_dir else (Path(self.cfg.work_root) / "meta")
        #sra_meta_root = Path(out_dir) if out_dir else (Path(self.cfg.work_root) / "meta")
        meta_root.mkdir(parents=True, exist_ok=True)
        #sra_meta_root.mkdir(parents=True, exist_ok=True)
    
        # 2) 生成/更新 JSON metadata（注意是 out_dir 参数）
        _geo.geo_accession_to_meta_json(accession, out_dir=str(meta_root))
    
        # 3) 生成 RunInfo CSV（注意是 accession + meta_dir/out_dir）
        info = _ed.gse_meta_to_runinfo_csv(
            accession=accession,
            meta_dir=str(meta_root),
            out_dir=str(meta_root),
            organism_filter=organism_filter,
            layout_filter=layout_filter,
        )
        runinfo_csv = Path(info["csv"]) if info.get("csv") else None
    
        # 4) 提取 SRR 列表
        srrs: list[str] = []
        if runinfo_csv and runinfo_csv.exists():
            import csv
            with runinfo_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ("Run", "acc", "Run Accession", "RunAccession"):
                        if key in row and row[key]:
                            srrs.append(row[key].strip())
                            break
            # 去重保序
            seen = set(); srrs = [x for x in srrs if not (x in seen or seen.add(x))]
    
        # 5) 可选：结构化字段
        try:
            meta_struct = _geo.parse_geo_soft_to_struct(_geo.fetch_geo_text(accession))
        except Exception:
            meta_struct = None
    
        return {
            "meta_json": meta_root / f"{accession}_meta.json",
            "runinfo_csv": runinfo_csv,
            "srr_list": srrs,
            "edirect_info": info,   # 包含 term_used/rows 等
            "meta_struct": meta_struct,
        }

    # ---------- SRA: prefetch ----------
    def prefetch(self, srr_list: Sequence[str]) -> Sequence[Path]:
        """
        Prefetch SRAs to .sra files.
        Returns: list of .sra paths.
        """
        # Delegate to your prefetch implementation. We assume it exposes a function like:
        # _sra_prefetch.prefetch_batch(srr_list, out_root=..., threads=...)
        if not hasattr(_sra_prefetch, "prefetch_batch"):
            raise RuntimeError("sra_prefetch.prefetch_batch(...) not found. Please expose it.")
        return _sra_prefetch.prefetch_batch(
            srr_list=srr_list,
            out_root=str(self.cfg.prefetch_root),
            threads=self.cfg.threads
        )

    # ---------- SRA: fasterq-dump ----------
    def fasterq(self, srr_list: Sequence[str]) -> Sequence[Tuple[str, Path, Path]]:
        """
        Convert SRA to paired FASTQ(.gz).
        Returns: list of tuples (srr, fq1_path, fq2_path).
        """
        if not hasattr(_sra_fasterq, "fasterq_batch"):
            raise RuntimeError("sra_fasterq.fasterq_batch(...) not found. Please expose it.")
        return _sra_fasterq.fasterq_batch(
            srr_list=srr_list,
            out_root=str(self.cfg.fasterq_root),
            threads=self.cfg.threads,
            mem = self.cfg.memory,
            gzip_output=self.cfg.gzip_fastq
        )

    # ---------- QC: fastp ----------
    def fastp(self, fq_pairs: Sequence[Tuple[str, Path, Path]]) -> Sequence[Tuple[str, Path, Path]]:
        """
        Run fastp on paired FASTQ files.
        Returns: list of tuples (srr, fq1_qc, fq2_qc).
        """
        if not self.cfg.fastp_enabled:
            # pass-through without QC
            return [(srr, fq1, fq2) for srr, fq1, fq2 in fq_pairs]

        if not hasattr(_qc_fastp, "fastp_batch"):
            raise RuntimeError("qc_fastp.fastp_batch(...) not found. Please expose it.")
        return _qc_fastp.fastp_batch(
            pairs=fq_pairs,
            out_root=str(self.cfg.qc_root),
            threads=self.cfg.threads
        )

    # ---------- Alignment (placeholder) ----------
    def star_align(
        self,
        clean_fastqs: Sequence[Tuple[str, Path, Path]],
        *,
        gencode_release: str = "v44",
        sjdb_overhang: Optional[int] = 149,
        accession_for_species: Optional[str] = None,   # 所有样本同一 GSE 时可统一传；否则保持 None
        max_workers: Optional[int] = None,             # 同时跑多少个样本；None=串行，日志更清晰
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        批量跑 STAR（调用 batch 版 star_step），返回：
          [(srr, bam_path, index_dir|None), ...]
        - 幂等：若 <SRR>/Aligned.sortedByCoord.out.bam 已存在且>1MB，则 [SKIP]
        - index_dir 若在 star_tools 返回中可解析则给出，否则为 None（与你后续 GTF 推断逻辑一致）
        """
        if not hasattr(_star_step, "make_star_step"):
            raise RuntimeError("star_step.make_star_step(...) not found")

        # 构造一步“可批量”的 step（与原有工厂接口完全一致）
        step = _star_step.make_star_step(
            index_root=str(self.cfg.star_index_root),
            out_root=str(self.cfg.star_align_root),
            threads=int(self.cfg.threads),
            gencode_release=gencode_release,
            sjdb_overhang=sjdb_overhang,
            accession_for_species=accession_for_species,
            max_workers=max_workers,   # None=串行；也可外部传 2/4 并发
        )

        # 规范输入为 [(srr, str(fq1), str(fq2)), ...]
        pairs: List[Tuple[str, str, str]] = [
            (srr, str(Path(fq1)), str(Path(fq2))) for srr, fq1, fq2 in clean_fastqs
        ]

        # 直接调用批量 command，得到 [(srr, bam, index_dir|None), ...]
        products = step["command"](pairs, logger=None)
        # 与随后 pipeline 的“三元组规范化”完全兼容
        return products

    # ---------- Counting via featureCounts ----------
    def featurecounts(
        self,
        bam_triples: Sequence[Tuple[str, str | Path, Optional[str]]],   # [(srr, bam, index_dir|None)]
        *,
        gtf: Optional[str | Path] = None,         # 显式 GTF（优先级最高）
        simple: Optional[bool] = None,            # None→cfg.featurecounts_simple
        by: Optional[str] = None,                 # None→cfg.featurecounts_by
        threads: Optional[int] = None,            # None→cfg.threads
        max_workers: Optional[int] = None,        # 预留（count_tools 可并行时透传）
    ) -> Dict[str, object]:
        """
        批量调用 featureCounts。返回：
          { "tables": [(srr, table_path), ...], "matrix": <path|None>, "failed": [] }
        幂等：<counts_root>/<SRR>/<SRR>.counts.txt 存在且>0则跳过计算。
        """
        if not hasattr(_count_step, "make_featurecounts_step"):
            raise RuntimeError("count_step.make_featurecounts_step(...) not found")

        out_root = Path(self.cfg.counts_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # ---------- 内置 GTF 自动推断 ----------
        def _infer_gtf_from_bams(triples: Sequence[Tuple[str, str | Path, Optional[str]]]) -> Optional[str]:
            # 1) 优先：从每个样本携带的 index_dir 推断
            for _srr, _bam, idx_dir in triples:
                if not idx_dir:
                    continue
                idx = Path(idx_dir)
                # (a) 本目录 / 父目录搜 *.gtf
                for base in {idx, idx.parent}:
                    for p in base.glob("*.gtf"):
                        return str(p.resolve())
                # (b) _cache 下搜 *.gtf
                for base in {idx.parent, idx.parent.parent}:
                    cache = base / "_cache"
                    if cache.exists():
                        hits = list(cache.rglob("*.gtf"))
                        if hits:
                            return str(hits[0].resolve())
                # (c) 再向上一级补充一轮
                for p in idx.parent.parent.glob("*.gtf"):
                    return str(p.resolve())

            # 2) 其次：从配置的 star_index_root 下兜底搜索
            idx_root = Path(getattr(self.cfg, "star_index_root", "index"))
            if idx_root.exists():
                hits = list(idx_root.rglob("*.gtf"))
                if hits:
                    return str(hits[0].resolve())

            # 3) 最后：环境变量 FC_GTF_HINT
            env_hint = os.environ.get("FC_GTF_HINT")
            if env_hint and Path(env_hint).exists():
                return str(Path(env_hint).resolve())

            return None

        # 若未显式给 gtf，则自动推断
        if gtf is None:
            inferred = _infer_gtf_from_bams(bam_triples)
            if inferred:
                print(f"[INFO] featureCounts: inferred GTF -> {inferred}")
                gtf = inferred
            else:
                raise RuntimeError(
                    "[featureCounts] 无法自动找到 GTF，请显式传入 gtf= 或设置环境变量 FC_GTF_HINT。"
                )

        # ---------- 构建 step 工厂并幂等检查 ----------
        step = _count_step.make_featurecounts_step(
            out_root=str(out_root),
            simple=(self.cfg.featurecounts_simple if simple is None else bool(simple)),
            gtf=None,  # 运行时 gtf 通过 command(...) 传入，优先级最高
            by=(by or self.cfg.featurecounts_by),
            threads=int(threads or self.cfg.threads),
            gtf_path=str(gtf),  # 作为工厂的后备（内部优先用 command 的 gtf）
        )

        def _table_path_for(srr: str) -> Path:
            # 若你的 count_tools 产物实际是 .csv，这里改成 .csv 并同步改 outputs 模板
            return out_root / srr / f"{srr}.counts.txt"

        # 幂等：全部已有则跳过
        outs_by_srr: List[Tuple[str, Path]] = [(str(srr), _table_path_for(str(srr))) for srr, _bam, _ in bam_triples]
        if all(step["validation"]([str(p)]) for _, p in outs_by_srr):
            print("[SKIP] featureCounts for all")
            tables = [(srr, str(p)) for srr, p in outs_by_srr]
            return {"tables": tables, "matrix": None, "failed": []}

        # 组装 (srr, bam) 列表并运行
        bam_pairs = [(str(srr), str(bam)) for (srr, bam, _idx) in bam_triples]
        ret = step["command"](
            bam_pairs,
            logger=None,
            gtf=str(gtf),  # 显式传入，优先级最高
        )

        tables = [(srr, str(_table_path_for(str(srr)))) for srr, _ in bam_pairs]
        matrix_path = ret.get("matrix") if isinstance(ret, dict) else None
        return {"tables": tables, "matrix": matrix_path, "failed": []}

    # ---------- End-to-end convenience ----------
    def run(self, srr_list: Sequence[str], *, with_align: bool = False, align_index: Optional[Path] = None) -> Dict[str, Any]:
        """
        Convenience runner: prefetch -> fasterq -> (fastp) -> [align] -> featureCounts

        Returns a dict of paths keyed by step.
        """
        sras = self.prefetch(srr_list)
        fastqs = self.fasterq(srr_list)
        fastqs_qc = self.fastp(fastqs)

        result: Dict[str, Any] = {
            "prefetch": sras,
            "fastq": fastqs,
            "fastq_qc": fastqs_qc,
        }

        if with_align:
            bams = self.align(fastqs_qc, index_root=align_index)
        else:
            # If you have a separate step that takes fastq->counts directly, wire it here.
            # Otherwise you need an alignment step. We raise for clarity.
            raise NotImplementedError("No aligner wired. Set with_align=True and implement .align().")

        counts = self.featurecounts(bams, gtf=self.cfg.gtf)
        result["bam"] = bams
        result["counts"] = counts
        return result
