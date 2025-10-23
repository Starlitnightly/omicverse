# star_step.py  —— 批量 STAR 步骤（与现有 star_tools 保持兼容）
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
import os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from .star_tools import star_align_auto     # 复用你的实现
from .tools_check import which_or_find, merged_env


def _bam_ok(p: Path) -> bool:
    """BAM>1MB 即认为有效；若有 .bai 更好，但不强制。"""
    try:
        return p.exists() and p.stat().st_size > 1_000_000
    except Exception:
        return False


def _normalize_bam(bam_path: Path, sample_dir: Path, srr: str) -> Path:
    """
    归一化 BAM 命名为 <SRR>/Aligned.sortedByCoord.out.bam
    - 若 star_align_auto 产出例如: run.Aligned.sortedByCoord.out.bam，则移动/重命名到规范名称
    - 同步处理 .bai（若存在）
    归一化：
      1) 确保标准名 <SRR>/Aligned.sortedByCoord.out.bam 存在（必要时移动原产物）
      2) 暴露 SRR 命名句柄 <SRR>/<srr>.sorted.bam （优先软链接；失败则复制）
      3) 返回 SRR 命名句柄（供下游 featureCounts 使用，列名唯一）
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    target = sample_dir / "Aligned.sortedByCoord.out.bam"
    target_bai = target.with_suffix(".bam.bai")

    # 如果不是标准名，先把 bam_path 挪到标准名
    if bam_path.resolve() != target.resolve():
        # 先把旧索引名定位出来（若存在）
        src_bai = bam_path.with_suffix(".bam.bai")
        # 移动 BAM
        bam_path.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        if not target.exists():
            shutil.move(str(bam_path), str(target))
        # 移动 BAI
        if src_bai.exists():
            if target_bai.exists():
                target_bai.unlink()
            shutil.move(str(src_bai), str(target_bai))

    # 确保有索引（如果还没有）
    if not target_bai.exists():
        samtools = which_or_find("samtools")
        subprocess.run([samtools, "index", str(target)], check=True)

    # 创建/刷新 SRR 命名的软链接
    srr_bam = sample_dir / f"{srr}.sorted.bam"
    srr_bai = sample_dir / f"{srr}.sorted.bam.bai"
    try:
        if srr_bam.is_symlink() or srr_bam.exists():
            srr_bam.unlink()
        if srr_bai.is_symlink() or srr_bai.exists():
            srr_bai.unlink()
        # 相对链接更稳（目录内）
        srr_bam.symlink_to(target.name)
        srr_bai.symlink_to(target_bai.name)
    except OSError:
        # 某些文件系统/权限不支持 symlink，则复制一份
        shutil.copy2(str(target), str(srr_bam))
        shutil.copy2(str(target_bai), str(srr_bai))

    return srr_bam


def _try_parse_index_dir(ret) -> Optional[Path]:
    """
    star_align_auto 的返回可能是:
      - str(bam)
      - (bam,)
      - (bam, index_dir)
      - [bam, index_dir, ...]
    这里尽量解析 index_dir；实在确定不了就返回 None。
    """
    if isinstance(ret, (list, tuple)):
        if len(ret) >= 2:
            cand = Path(ret[1])
            if cand.exists() and cand.is_dir():
                return cand
    return None


def _align_one(
    srr: str,
    fq1: str | Path,
    fq2: str | Path,
    index_root: str,
    out_root: str,
    threads: int,
    gencode_release: str,
    sjdb_overhang: Optional[int],
    accession_for_species: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    """
    单样本对齐；返回 (srr, bam_path, index_dir|None)
    """
    sample_dir = Path(out_root) / srr
    sample_dir.mkdir(parents=True, exist_ok=True)
    # 规范化预期产物路径（用于幂等跳过 & 下游验证）
    expected_bam = sample_dir / "Aligned.sortedByCoord.out.bam"

    # 幂等：已有就跳过
    if _bam_ok(expected_bam):
        print(f"[SKIP] STAR {srr}: {expected_bam}")
        # 补齐 SRR 命名句柄（若不存在）
        srr_bam = _normalize_bam(expected_bam, sample_dir, srr)
        return srr, str(srr_bam), None

    # 仍兼容你原来的 out_prefix="run."
    out_prefix = sample_dir / "run."
    acc = accession_for_species or srr

    # 调你的现有封装
    ret = star_align_auto(
        accession=acc,
        fq1=str(fq1),
        fq2=str(fq2),
        index_root=index_root,
        out_prefix=str(out_prefix),
        threads=threads,
        gencode_release=gencode_release,
        sjdb_overhang=sjdb_overhang,
        sample=srr,
    )

    # 解析 bam_path
    if isinstance(ret, (list, tuple)):
        bam_path = Path(ret[0])
    else:
        bam_path = Path(ret)

    if not bam_path.exists():
        raise FileNotFoundError(f"[STAR] BAM not found for {srr}: {bam_path}")

    # 归一化命名为 <SRR>/Aligned.sortedByCoord.out.bam
    bam_path = _normalize_bam(bam_path, sample_dir, srr)   # ← 传 srr
    idx_dir = _try_parse_index_dir(ret)
    return srr, str(bam_path), (str(idx_dir) if idx_dir else None)


def make_star_step(
    index_root: str = "index",
    out_root: str = "work/star",
    threads: int = 12,
    gencode_release: str = "v44",
    sjdb_overhang: int | None = 149,
    accession_for_species: str | None = None,  # 若每个样本同一 GSE，可统一传
    max_workers: int | None = None,            # ← 新增：批量并发数（None=串行）
):
    """
    输入：[(srr, fq1_clean, fq2_clean), ...]
    输出：work/star/{SRR}/Aligned.sortedByCoord.out.bam（每样本独立子目录）
    验证：BAM 存在且 > 1MB
    """
    def _cmd(clean_fastqs: Sequence[Tuple[str, str | Path, str | Path]], logger=None) -> List[Tuple[str, str, Optional[str]]]:
        os.makedirs(out_root, exist_ok=True)

        # 若未指定并发，默认串行（保持日志顺序清晰）
        if not max_workers or max_workers <= 1:
            products = []
            for srr, fq1, fq2 in clean_fastqs:
                rec = _align_one(
                    srr=srr, fq1=fq1, fq2=fq2,
                    index_root=index_root, out_root=out_root,
                    threads=threads, gencode_release=gencode_release,
                    sjdb_overhang=sjdb_overhang,
                    accession_for_species=accession_for_species,
                )
                products.append(rec)  # (srr, bam, index_dir|None)
            return products

        # 并发模式（每样本一个任务）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        products: List[Tuple[str, str, Optional[str]]] = []
        errors: List[Tuple[str, str]] = []

        def _worker(item):
            srr, fq1, fq2 = item
            return _align_one(
                srr=srr, fq1=fq1, fq2=fq2,
                index_root=index_root, out_root=out_root,
                threads=threads, gencode_release=gencode_release,
                sjdb_overhang=sjdb_overhang,
                accession_for_species=accession_for_species,
            )

        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            futs = {ex.submit(_worker, it): it[0] for it in clean_fastqs}
            for fut in as_completed(futs):
                srr = futs[fut]
                try:
                    products.append(fut.result())
                except Exception as e:
                    msg = str(e)
                    errors.append((srr, msg))
                    if logger:
                        logger.error(f"[STAR] {srr} failed: {msg}")

        if errors:
            # 你也可以选择继续返回成功的部分
            err_msg = "; ".join([f"{s}:{m}" for s, m in errors])
            raise RuntimeError(f"STAR failed for {len(errors)} samples: {err_msg}")

        # 保持与输入顺序一致
        order = {s: i for i, (s, _, _) in enumerate(clean_fastqs)}
        products.sort(key=lambda x: order[x[0]])
        return products

    return {
        "name": "star",
        "command": _cmd,  # 接收[(srr, fq1_clean, fq2_clean), ...]，返回[(srr, bam, index_dir|None), ...]
        "outputs": [f"{out_root}" + "/{SRR}/Aligned.sortedByCoord.out.bam"],  # 与你的验证一致
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 1_000_000 for f in fs),
        "takes": "CLEAN_FASTQ_PATHS",
        "yields": "BAM_PATHS"  # 实际返回三元组，后续仍按你已有规范化逻辑处理
    }