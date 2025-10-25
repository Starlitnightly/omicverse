# qc_fastp.py fastp 步骤工厂
from __future__ import annotations

try:
    from .qc_tools import fastp_clean_parallel
except ImportError:
    from qc_tools import fastp_clean_parallel

import os, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Tuple, List, Dict

try:
    from .tools_check import which_or_find, merged_env
except ImportError:
    from tools_check import which_or_find, merged_env


def _run(cmd: list[str], env: dict | None = None):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ext(p: Path) -> str:
    # 保持与输入一致：输入是 .gz 则输出也 .gz
    return ".fastq.gz" if p.suffix == ".gz" or p.name.endswith(".fastq.gz") else ".fastq"

def _fastp_one(
    srr: str,
    fq1: Path,
    fq2: Path,
    out_root: Path,
    threads: int = 8,
) -> Tuple[str, Path, Path, Path, Path]:
    """
    跑一个样本的 fastp（paired-end）。
    返回：(srr, clean1, clean2, json, html)
    """
    env = merged_env()
    fastp_bin = which_or_find("fastp")  # 解析可执行路径
    out_dir = _ensure_dir(out_root / srr)

    # 输出文件名（与输入后缀保持一致）
    oext = _ext(fq1)
    clean1 = out_dir / f"{srr}_clean_1{oext.replace('.fastq', '')}"
    clean2 = out_dir / f"{srr}_clean_2{oext.replace('.fastq', '')}"
    json = out_dir / f"{srr}.fastp.json"
    html = out_dir / f"{srr}.fastp.html"

    # 已存在就跳过（可按需改成强制覆盖）
    if clean1.exists() and clean2.exists() and json.exists() and html.exists():
        return srr, clean1, clean2, json, html

    cmd = [
        fastp_bin,
        "-i", str(fq1),
        "-I", str(fq2),
        "-o", str(clean1),
        "-O", str(clean2),
        "-w", str(threads),
        "-j", str(json),
        "-h", str(html),
        "--detect_adapter_for_pe",
        "--thread", str(threads),   # 一些版本也接受 --thread
        "--overrepresentation_analysis",
    ]

    # 如果输出是 .gz，fastp 会自动压缩（由后缀决定），不必额外 gzip
    # 可选：加更严格过滤参数（示例）——按需启用：
    # cmd += ["-q", "20", "-u", "30", "-l", "30"]  # 质量阈值/不合格比例/最短长度

    _run(cmd, env=env)

    # 简单校验
    if not (clean1.exists() and clean1.stat().st_size > 0 and clean2.exists() and clean2.stat().st_size > 0):
        raise RuntimeError(f"fastp outputs missing/empty for {srr} in {out_dir}")

    return srr, clean1, clean2, json, html


def fastp_batch(
    pairs: Sequence[Tuple[str, Path, Path]],  # [(srr, fq1, fq2), ...]
    out_root: str | Path,
    threads: int = 12,          # 每个样本 fastp 工作线程
    max_workers: int | None = None,  # 同时处理的样本数；None=和 threads 一样或自动
) -> List[Tuple[str, Path, Path, Path, Path]]:
    """
    并发跑 fastp。
    返回：[(srr, clean1, clean2, json, html), ...]（按输入顺序）
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        # 默认同时处理的样本数 = min(线程数, 核心数/2)（你也可以换成固定值）
        import os, math
        max_workers = max(1, min(threads, (os.cpu_count() or 8) // 2))

    results: Dict[str, Tuple[str, Path, Path, Path, Path]] = {}
    errors: List[Tuple[str, str]] = []

    def _worker(item: Tuple[str, Path, Path]):
        srr, fq1, fq2 = item
        return _fastp_one(srr, Path(fq1), Path(fq2), out_root=out_root, threads=threads)

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futs = {ex.submit(_worker, it): it[0] for it in pairs}
        for fut in as_completed(futs):
            srr = futs[fut]
            try:
                ret = fut.result()
                results[srr] = ret
            except Exception as e:
                errors.append((srr, str(e)))

    if errors:
        msg = "; ".join([f"{s}:{m}" for s, m in errors])
        raise RuntimeError(f"fastp_batch failed for {len(errors)} samples: {msg}")

    # 按输入顺序组织
    order = {s: i for i, (s, _, _) in enumerate(pairs)}
    out = [results[s] for s, _, _ in sorted(pairs, key=lambda x: order[x[0]])]
    return out

def make_fastp_step(
    out_root: str = "work/fastp",
    threads_per_job: int = 12,
    max_workers: int | None = None,
    backend: str = "process",
):
    """
    输入：FASTQ 列表（[(srr, fq1, fq2), ...]）
    输出：work/fastp/{SRR}_1.clean.fq.gz, {SRR}_2.clean.fq.gz
    验证：清洗后的 fq 均存在且 size > 0
    """
    def _cmd(fastq_triplets: Sequence[tuple[str, str, str]], logger=None):
        # 三元组模式直接传给 fastp_clean_parallel
        # outdir 在三元组模式下不会被使用，但函数签名需要，给个合理值即可
        os.makedirs(out_root, exist_ok=True)

        # 取第一个样本的 fq1 所在目录作为占位 outdir；若列表为空或路径异常，则用 "."
        if fastq_triplets:
            first_fq1_dir = os.path.dirname(fastq_triplets[0][1]) or "."
        else:
            first_fq1_dir = "."

        # 关键点：让 fastp_clean 的输出落在 {work_dir}/fastp/ 之下，
        # 而 out_root = "{work_dir}/fastp"。因此取 work_dir = dirname(out_root)
        work_dir = os.path.dirname(out_root) or "."

        return fastp_clean_parallel(
            samples=list(fastq_triplets),           # 三元组模式 [(srr, fq1, fq2), ...]
            outdir=first_fq1_dir,                   # 三元组模式会忽略此参数
            work_dir=work_dir,                      # 确保输出到 {work_dir}/fastp = out_root
            fastp_threads=threads_per_job,
            max_workers=max_workers,
            retries=2,
            backend=backend
        )

    return {
        "name": "fastp",
        "command": _cmd,  # 接收 [(srr, fq1, fq2), ...]
        "outputs": [f"{out_root}" + "/{SRR}_1.clean.fq.gz",
                    f"{out_root}" + "/{SRR}_2.clean.fq.gz"],
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fs),
        "takes": "FASTQ_PATHS",
        "yields": "CLEAN_FASTQ_PATHS"
    }