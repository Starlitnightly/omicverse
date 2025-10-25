# QCtools Fastp

import os, sys, shutil, subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from math import floor

# --- 你已有的单样本清洗函数，保持不变（示意） ---
def fastp_clean(fq1: str, fq2: Optional[str], sample: str, work_dir: str, threads: int = 4) -> Tuple[str, str]:
    """
    读取 fq1,fq2，输出到 {work_dir}/fastp/{sample}_1.clean.fq.gz / _2.clean.fq.gz
    若已存在则跳过，返回输出路径。
    """
    outdir = Path(work_dir) / "fastp"
    outdir.mkdir(parents=True, exist_ok=True)
    out1 = outdir / f"{sample}_1.clean.fq.gz"
    out2 = outdir / f"{sample}_2.clean.fq.gz"

    if out1.exists() and out2.exists():
        return str(out1), str(out2)

    cmd = [
        "fastp",
        "-i", fq1,
        "-I", fq2 if fq2 else fq1.replace("_1.fastq", "_2.fastq"),
        "-o", str(out1),
        "-O", str(out2),
        "-w", str(threads),
        "-j", str(outdir / f"{sample}.json"),
        "-h", str(outdir / f"{sample}.html"),
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return str(out1), str(out2)

# ---------- 新增：顶层子任务执行器（可被进程池 pickling） ----------
def _fastp_run_one_triplet(srr: str, fq1: str, fq2: Optional[str], work_dir: str, fastp_threads: int, retries: int):
    """
    三元组模式：直接用传入的 fq1/fq2。
    返回：(sample, out1, out2, status)  status in {"OK","SKIP"}
    """
    outdir = Path(work_dir) / "fastp"
    out1 = outdir / f"{srr}_1.clean.fq.gz"
    out2 = outdir / f"{srr}_2.clean.fq.gz"
    outdir.mkdir(parents=True, exist_ok=True)

    # 已存在直接跳过
    if out1.exists() and out2.exists():
        return (srr, str(out1), str(out2), "SKIP")

    last_err = None
    for _ in range(max(1, retries)):
        try:
            c1, c2 = fastp_clean(fq1, fq2, srr, work_dir, threads=fastp_threads)
            return (srr, c1, c2, "OK")
        except Exception as e:
            last_err = e
    raise last_err

def _fastp_run_one_from_outdir(srr: str, outdir: str, work_dir: str, fastp_threads: int, retries: int):
    """
    SRR 列表模式：从 outdir/{SRR}_1.fastq / _2.fastq 推断输入。
    """
    fq1 = str(Path(outdir) / f"{srr}_1.fastq")
    fq2 = str(Path(outdir) / f"{srr}_2.fastq")
    return _fastp_run_one_triplet(srr, fq1, fq2, work_dir, fastp_threads, retries)

# ---------- 这里是你的并行 API：最小改动以兼容两种输入 ----------
def fastp_clean_parallel(
    samples: List[Union[str, Tuple[str, str, Optional[str]]]],
    outdir: str,
    work_dir: str,
    fastp_threads: int = 4,
    max_workers: Optional[int] = None,
    retries: int = 2,
    backend: str = "process",
):
    """
    并行批量运行 fastp_clean。
    - 输入可为：
        1) SRR 列表：["SRRxxxx", ...]  将从 outdir/{SRR}_1.fastq/_2.fastq 读取
        2) 三元组列表：[ (srr, fq1, fq2), ... ]  直接使用指定的 FASTQ 路径
    - 输出：{"success": [(srr, out1, out2, status), ...], "failed": [(srr, err), ...]}
    """
    total_cores = os.cpu_count() or 8
    if max_workers is None:
        max_workers = max(1, floor(total_cores / max(1, fastp_threads)))

    print(f"[INFO] CPU cores={total_cores}, per fastp threads={fastp_threads}, max parallel={max_workers}")
    print(f"[INFO] fastp path: {shutil.which('fastp')}")
    print(f"[INFO] Python exec: {sys.executable}")

    # 判定输入模式
    def is_triplet(x):
        return isinstance(x, (tuple, list)) and (2 <= len(x) <= 3)

    use_triplets = len(samples) > 0 and is_triplet(samples[0])

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    results, errors = [], []

    with Executor(max_workers=max_workers) as ex:
        if use_triplets:
            # items: List[(srr, fq1, fq2)]
            futs = {
                ex.submit(_fastp_run_one_triplet, srr, fq1, (fq2 if len(item) > 2 else None),
                          work_dir, fastp_threads, retries): srr
                for item in samples
                for srr, fq1, fq2 in [item if len(item) == 3 else (item[0], item[1], None)]
            }
        else:
            # items: List[str]  (SRR)
            futs = {
                ex.submit(_fastp_run_one_from_outdir, srr, outdir, work_dir, fastp_threads, retries): srr
                for srr in samples
            }

        for fut in as_completed(futs):
            s = futs[fut]
            try:
                sample, out1, out2, status = fut.result()
                results.append((sample, out1, out2, status))
                print(f"[{status}] {sample} -> {out1}, {out2}")
            except Exception as e:
                errors.append((s, str(e)))
                print(f"[ERR] {s}: {e}")

    ok_n = sum(1 for r in results if r[3] == "OK")
    skip_n = sum(1 for r in results if r[3] == "SKIP")
    print(f"[SUMMARY] Completed={ok_n}, Skipped={skip_n}, Failed={len(errors)}")
    return {"success": results, "failed": errors}