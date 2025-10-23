# pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Union
import pandas as pd
from .sra_prefetch import make_prefetch_step, run_prefetch_with_progress, _make_local_logger
from .sra_fasterq import make_fasterq_step
from .qc_fastp import make_fastp_step
from .star_step import make_star_step
from .count_step import make_featurecounts_step
import .tools_check
import os

# 1) 声明式组合步骤（可根据项目需求自由调整参数）
tools_check.check()
STEPS = [
    make_prefetch_step(),
    make_fasterq_step(outdir_pattern="work/fasterq", threads_per_job=12, compress_after=True),
    make_fastp_step(out_root="work/fastp", threads_per_job=12),
    make_star_step(index_root="index", out_root="work/star", threads=12, gencode_release="v44", sjdb_overhang=149),  
    make_featurecounts_step(out_root="work/counts", simple=True, gtf=None, by="auto", threads=8)
]

def _render_paths(patterns, **kwargs):
    return [p.format(**kwargs) for p in patterns]
def _safe_call_validation(validation_fn, outs, logger):
    try:
        return bool(validation_fn(outs))
    except FileNotFoundError:
        # ✅ 即使验证函数内部有人误写了 getsize 也不至于炸
        logger.warning(f"[VALIDATION] missing file(s): {outs}")
        return False
    except Exception as e:
        logger.warning(f"[VALIDATION] raised {type(e).__name__}: {e} | outs={outs}")
        return False
def _normalize_srr_list(
    srrs_or_csv: Union[str, Path, Iterable[str], pd.DataFrame]
) -> List[str]:
    """
    接受：SRR 列表 / CSV 路径 / DataFrame
    - CSV 会优先尝试 'Run' 列（不区分大小写），其次 'run_accession'
    - 自动去重但保持原始顺序
    """
    if isinstance(srrs_or_csv, (str, Path)):
        csv_path = Path(srrs_or_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    elif isinstance(srrs_or_csv, pd.DataFrame):
        df = srrs_or_csv
    else:
        # 可迭代对象（list/tuple/set/generator）
        seq = list(srrs_or_csv)
        # 去重保序
        return list(dict.fromkeys(str(x).strip() for x in seq if str(x).strip()))

    # 针对 DataFrame：找列
    cols_lower = {c.lower(): c for c in df.columns}
    col = None
    for cand in ("run", "run_accession"):
        if cand in cols_lower:
            col = cols_lower[cand]
            break
    if col is None:
        raise ValueError(
            f"CSV must contain a 'Run' (or 'run_accession') column. Got columns: {list(df.columns)}"
        )

    values = [str(x).strip() for x in df[col].tolist()]
    # 去重保序
    return list(dict.fromkeys(v for v in values if v))
    
def run_pipeline(srr_list_or_csv):
    # NEW: 兼容 CSV / DataFrame / 列表
    srr_list = _normalize_srr_list(srr_list_or_csv)
    if not srr_list:
        raise ValueError("未解析到任何 SRR 编号。")
    logger = _make_local_logger("PIPE")
    # 数据在步骤之间的“携带形式”
    fastq_paths = []       # [(srr, fq1, fq2)]
    clean_fastqs = []      # [(srr, clean1, clean2)]
    bam_paths = []         # [(srr, bam)]
    count_tables = []      # 可选

    # Step 0: prefetch（逐个 SRR）
    #logger.info(f"[RUN] prefetch {srr_id} -> {output_path}")
    prefetch_step = STEPS[0]

    for srr in srr_list:
        outs = _render_paths(prefetch_step["outputs"], SRR=srr)
        # 使用安全验证
        if _safe_call_validation(prefetch_step["validation"], outs, logger):
            logger.info(f"[SKIP] prefetch {srr}")
        else:
            ok = prefetch_step["command"](srr, logger=logger)
            if not ok:
                logger.error(f"[FAIL] prefetch {srr}")
                return False

    # Step 1: fasterq（批处理）
    fasterq_step = STEPS[1]
    outs_by_srr = []
    for srr in srr_list:
        outs = _render_paths(fasterq_step["outputs"], SRR=srr)
        outs_by_srr.append((srr, outs))
    # 如果全部存在就跳过
    if all(fasterq_step["validation"](outs) for _, outs in outs_by_srr):
        logger.info("[SKIP] fasterq for all")
        fastq_paths = [(srr, outs[0], outs[1]) for srr, outs in outs_by_srr]
    else:
        ret = fasterq_step["command"]([s for s,_ in outs_by_srr], logger=logger)
        # 规范化到 [(srr, fq1, fq2)]：你可以从 ret["success"] 解构；这里简单按模板返回
        fastq_paths = [(srr, outs[0], outs[1]) for srr, outs in outs_by_srr]

    # Step 2: fastp（批处理）
    fastp_step = STEPS[2]
    outs_by_srr = []
    for srr, fq1, fq2 in fastq_paths:
        outs = _render_paths(fastp_step["outputs"], SRR=srr)
        outs_by_srr.append((srr, fq1, fq2, outs))
    if all(fastp_step["validation"](o) for *_, o in outs_by_srr):
        logger.info("[SKIP] fastp for all")
        clean_fastqs = [(srr, o[0], o[1]) for srr, _, _, o in outs_by_srr]
    else:
        ret = fastp_step["command"]([(srr, fq1, fq2) for srr, fq1, fq2, _ in outs_by_srr], logger=logger)
        clean_fastqs = [(srr, o[0], o[1]) for srr, _, _, o in outs_by_srr]

    # Step 3: STAR（逐样本/可并行，这里串行示例，便于日志清晰）
    star_step = STEPS[3]
    outs_by_srr = []
    for srr, c1, c2 in clean_fastqs:
        outs = _render_paths(star_step["outputs"], SRR=srr)
        outs_by_srr.append((srr, c1, c2, outs))
    
    bam_paths = []  # 形如 [(srr, bam, idx_dir)]
    for srr, c1, c2, outs in outs_by_srr:
        if star_step["validation"](outs):
            logger.info(f"[SKIP] STAR {srr}")
            # 如果 validation 为真，说明 outs[0] 的 bam 已存在，但此时我们拿不到 index_dir。
            # 最稳妥：把 index_dir 设为 None，后面统一再推断一次（见 Step 4 前的逻辑）。
            bam_paths.append((srr, outs[0], None))  # NEW
        else:
            prods = star_step["command"]([(srr, c1, c2)], logger=logger)
            # 要求 star_step["command"] 返回 [(srr, bam_path, index_dir)] 结构  ← NEW（见说明）
            bam_paths.extend(prods)

    # 统一规范为三元组 (srr, bam, index_dir|None)  —— 关键修补
    _norm_bams = []
    for rec in bam_paths:
        if isinstance(rec, (list, tuple)):
            if len(rec) == 3:
                _norm_bams.append((rec[0], rec[1], rec[2]))
            elif len(rec) == 2:
                srr, bam = rec
                _norm_bams.append((srr, bam, None))
            else:
                raise ValueError(f"Unexpected bam_paths record (len={len(rec)}): {rec}")
        else:
            raise ValueError(f"Unexpected bam_paths record type: {type(rec)} -> {rec}")
    bam_paths = _norm_bams  # 现在结构统一了
    
    # --- NEW: 在进入 featureCounts 前，统一用任意一个非 None 的 index_dir 推断 GTF ---
    def _find_gtf_from_index(index_dir: str | os.PathLike) -> str:
        """
        给 STAR 的 index_dir（通常是 .../index/<species>/<build>/STAR），
        尝试在该结构的“缓存区/父目录”中找解压过的 .gtf。
        规则：
          1) 同目录或上级目录的 *.gtf
          2) 上上级目录下的 _cache/**.gtf
        """
        from pathlib import Path
        idx = Path(index_dir)
        if not idx:
            return None
    
        # 1) 本目录或父目录搜 *.gtf
        for base in {idx, idx.parent}:
            for p in base.glob("*.gtf"):
                return str(p.resolve())
    
        # 2) 两级父目录下的 _cache 搜索（兼容 ensure_star_index 的下载布局）
        for base in {idx.parent, idx.parent.parent}:
            cache = base / "_cache"
            if cache.exists():
                hits = list(cache.rglob("*.gtf"))
                if hits:
                    return str(hits[0].resolve())
    
        # 兜底：再向上一层搜一圈 *.gtf
        for p in idx.parent.parent.glob("*.gtf"):
            return str(p.resolve())
    
        return None
    
    # 选一个 index_dir 推断 GTF
    inferred_gtf = None
    for _, __, idx_dir in bam_paths:
        if idx_dir:
            inferred_gtf = _find_gtf_from_index(idx_dir)
            if inferred_gtf:
                break
    
    if not inferred_gtf:
        # 如果 STAR 是 “全部都走了 [SKIP]”，上面 idx_dir 可能全是 None。
        # 这时可以从约定位置再试一次（你可以按自己的布局改这段兜底逻辑）。
        # 例如：index/human/GRCh38/STAR 的上上级 _cache 里通常有 gtf
        candidate = Path("index")
        if candidate.exists():
            for p in candidate.rglob("*.gtf"):
                inferred_gtf = str(p.resolve())
                break
    
    if not inferred_gtf:
        logger.error("[featureCounts] 无法自动找到 GTF，请确认 ensure_star_index 已下载并解压注释文件。")
        # 在这里 return 或抛异常都可以；为了和你现有逻辑匹配，这里抛异常：
        raise RuntimeError("GTF not found. Please check STAR index/annotation preparation.")
    
    # Step 4: featureCounts（批处理）
    fc_step = STEPS[4]
    outs_by_srr = []
    for srr, bam, _idx in bam_paths:  # 注意现在的 bam_paths 有 3 元组
        outs = _render_paths(fc_step["outputs"], SRR=srr)
        outs_by_srr.append((srr, bam, outs))
    
    if all(fc_step["validation"](o) for _, _, o in outs_by_srr):
        logger.info("[SKIP] featureCounts for all")
        count_tables = [o[0] for _, _, o in outs_by_srr]
    else:
        # 显式把 gtf 传给 featureCounts 的命令函数  ← NEW
        ret = fc_step["command"]([(srr, bam) for srr, bam, _ in outs_by_srr],
                                 logger=logger,
                                 gtf=inferred_gtf)
        count_tables = [o[0] for _, _, o in outs_by_srr]
'''
How to use
import data_prepare_pipline
srrs = ["SRR12544419", "SRR12544565", "SRR12544566"]
data_prepare_pipline.run_pipeline(srrs)
'''
