# sra_fasterq.py
from __future__ import annotations
import os, sys, shutil, time, subprocess,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Sequence, Tuple, List, Dict

try:
    from .tools_check import which_or_find, merged_env
except ImportError:
    from tools_check import which_or_find, merged_env
    
def _fqdump_one(srr: str, outdir: str, fqdump_bin: str, threads: int, mem_gb: int, do_gzip: bool):
    os.makedirs(outdir, exist_ok=True)
    env = merged_env()
    # fasterq-dump 输出 .fastq（未压缩）
    cmd = [
        fqdump_bin, srr, "-p", "-O", outdir,
        "-e", str(threads), "--mem", f"{mem_gb}G", "--split-files"
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # 产物探测
    fq1 = os.path.join(outdir, f"{srr}_1.fastq")
    fq2 = os.path.join(outdir, f"{srr}_2.fastq")
    if not (os.path.exists(fq1) and os.path.exists(fq2)):
        raise FileNotFoundError(f"fasterq outputs missing for {srr} in {outdir}")

    # 可选：gzip 压缩并做 md5 校验
    if do_gzip:
        import hashlib, gzip, shutil as _sh
        for p in (fq1, fq2):
            gz = p + ".gz"
            if not os.path.exists(gz):
                with open(p, "rb") as f_in, gzip.open(gz, "wb") as f_out:
                    _sh.copyfileobj(f_in, f_out)
                # 简单完整性：再打开一次读头若干字节
                with gzip.open(gz, "rb") as f_chk:
                    _ = f_chk.read(128)
                os.remove(p)
        fq1 += ".gz"; fq2 += ".gz"

    return srr, fq1, fq2

def _have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def _candidate_fastq_paths(out_root: Path, srr: str, gzip_output: bool):
    suffix = ".fastq.gz" if gzip_output else ".fastq"
    fq1 = out_root / f"{srr}_1{suffix}"
    fq2 = out_root / f"{srr}_2{suffix}"
    return fq1, fq2

def _run(cmd: list[str]):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True)


def _work_one_fasterq(
    srr: str,
    out_root: Path,
    threads_per_job: int,
    mem_gb: int,
    tmp_root: Path,
    gzip_output: bool,
    retries: int = 3,
) -> tuple[str, Path, Path]:
    
    # 1) 子目录：每个 SRR 一个输出目录（更稳）
    out_dir = out_root / srr
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / srr
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Prefer local .sra if present
    local_sra = Path("work/prefetch") / srr / f"{srr}.sra"
    input_token = str(local_sra) if local_sra.exists() else srr
    cand_dirs = (out_dir, out_root)
    stems = {srr}
    try:
        stems.add(Path(input_token).stem)  # 当 input_token 是 .sra 路径时，可能是 "SRR" 或 "SRR.sra"
    except Exception:
        pass

    def _find_pair(ext_gz: bool) -> tuple[Path, Path] | None:
        suffix = ".fastq.gz" if ext_gz else ".fastq"
        for base in cand_dirs:
            for st in stems:
                # 同时兼容 "SRR_1" 与 "SRR.sra_1"
                for name1 in (f"{st}_1{suffix}", f"{st}.sra_1{suffix}"):
                    name2 = name1.replace("_1", "_2", 1)
                    a, b = base / name1, base / name2
                    if _have(a) and _have(b):
                        return a, b
        return None

    # ---------------- 早退 1：如果要求 gzip，且 .gz 成对已存在 -> 直接返回 ----------------
    if gzip_output:
        hit_gz = _find_pair(ext_gz=True)
        if hit_gz:
            fq_a, fq_b = hit_gz
            print(f"[SKIP] {srr}: paired .fastq.gz 已存在，跳过 fasterq-dump。")
            return srr, fq_a, fq_b

    # ---------------- 早退 2：若 .fastq 成对已存在 ----------------
    hit_plain = _find_pair(ext_gz=False)
    if hit_plain:
        fq_a, fq_b = hit_plain
        if gzip_output:
            # 只做 gzip（不再调用 fasterq-dump）
            print(f"[INFO] {srr}: 检测到未压缩 .fastq，start gzip step.")
            for src in (fq_a, fq_b):
                if src.exists() and src.suffix == ".fastq":
                    _run(["gzip", "-f", str(src)])
            fq_a = fq_a.with_suffix(".fastq.gz")
            fq_b = fq_b.with_suffix(".fastq.gz")
            if not (_have(fq_a) and _have(fq_b)):
                raise RuntimeError(f"gzip step did not produce gz files for {srr}")
            print(f"[SKIP] {srr}: 已完成 gzip 压缩，跳过 fasterq-dump。")
        else:
            print(f"[SKIP] {srr}: paired .fastq exists，jump fasterq-dump。")
        return srr, fq_a, fq_b


    # Build fasterq-dump command

    base_cmd = [
        shutil.which("fasterq-dump") or "fasterq-dump",
        input_token,
        "-p",
        "-O", str(out_dir),
        "-e", str(threads_per_job),
        "--mem", f"{mem_gb}G",
        "--split-files",
        "-t", str(tmp_dir),
        "-f",
    ]
    print(">>>>>> ", " ".join(base_cmd), flush=True)

    # Retry loop (most of your errors are rc=3 timeouts from S3)
    sleep = 8
    for attempt in range(1, retries + 1):
        try:
            _run(base_cmd)
            break
        except subprocess.CalledProcessError as e:
            # If we used network accession and have a prefetched .sra, retry on the local file immediately
            if input_token == srr and local_sra.exists():
                input_token = str(local_sra)
                base_cmd[1] = input_token
                print(f"[WARN] Switching to local SRA for retry: {local_sra}", flush=True)

            if attempt == retries:
                raise
            print(f"[WARN] fasterq-dump failed (try {attempt}/{retries}), sleep {sleep}s and retry...", flush=True)
            time.sleep(sleep)
            sleep *= 2

    # Validate raw fastq results exist
    # 产物定位：兼容输入为 .sra 时的 "SRR.sra_1.fastq" 命名 
    # 候选目录：SRR 子目录 + 根目录
    cand_dirs = (out_dir, out_root)

    # 生成候选前缀：通常是 srr；若 input_token 是 ".sra" 路径，则补充它的 stem 作为候选
    stems = {srr}
    try:
        stems.add(Path(input_token).stem)  # 可能是 "SRR123", 也可能是 "SRR123.sra"
    except Exception:
        pass

    def _find_pair(ext_gz: bool) -> tuple[Path, Path] | None:
        """在 cand_dirs × stems 中寻找成对的 _1/_2.fastq(.gz)"""
        suffix = ".fastq.gz" if ext_gz else ".fastq"
        for base in cand_dirs:
            for st in stems:
                # 兼容 "SRR_1" 与 "SRR.sra_1"
                for name1 in (f"{st}_1{suffix}", f"{st}.sra_1{suffix}"):
                    name2 = name1.replace("_1", "_2", 1)
                    a, b = base / name1, base / name2
                    if _have(a) and _have(b):
                        return a, b
        return None

    hit = _find_pair(ext_gz=True) or _find_pair(ext_gz=False)

    if not hit:
        # 再检查是否出现单端（不符合预期）
        for base in cand_dirs:
            for st in stems:
                for single in (base / f"{st}.sra.fastq", base / f"{st}.fastq"):
                    if _have(single):
                        raise RuntimeError(
                            f"{srr} produced single-end file: {single} (no _1/_2). "
                            f"Check library layout or remove --split-files."
                        )
        searched = ", ".join(str(d) for d in cand_dirs)
        raise RuntimeError(f"fasterq outputs missing for {srr} (searched: {searched})")

    fq_a, fq_b = hit
    # 若需要 gzip 且目前是 .fastq，就地压缩
    if gzip_output and fq_a.suffix == ".fastq":
        for src in (fq_a, fq_b):
            if src.exists() and src.suffix == ".fastq":
                _run(["gzip", "-f", str(src)])
        fq_a = fq_a.with_suffix(".fastq.gz")
        fq_b = fq_b.with_suffix(".fastq.gz")
        if not (_have(fq_a) and _have(fq_b)):
            raise RuntimeError(f"gzip step did not produce gz files for {srr}")

    return srr, fq_a, fq_b

def fasterq_batch(
    srr_list: Sequence[str],
    out_root: str | Path,
    threads: int = 8,          # ← “同时处理多少个 SRR”（映射到 max_workers）
    gzip_output: bool = True,  # 是否对 fastq 输出 gzip 压缩
    mem: str = "8G",
    tmp_root: str | Path = "work/tmp",
    backend: str = "process",
    threads_per_job: int = 24,       # 每个 SRR 内部的 fasterq-dump 线程数
) ->list[tuple[str, Path, Path]]:
    """
    适配器：把类传来的参数映射到 fasterq_dump_parallel，并把 dict 结果转成 list[tuple]
    """
    # 解析 mem (支持 "4G"/"8G" 这种形式；不匹配时默认 8G)
    try:
        mem_gb = int(str(mem).upper().rstrip("G"))
    except Exception:
        mem_gb = 8

    res = fasterq_dump_parallel(
        srr_list=list(srr_list),
        out_root=str(out_root),
        threads_per_job=threads_per_job,
        mem_gb=mem_gb,
        max_workers=int(threads),      # ← 并发度
        gzip_output=bool(gzip_output),
        backend=backend,
        tmp_root=str(tmp_root),
    )

    by_srr = res.get("by_srr", {})
    errs = res.get("failed", [])

    if errs:
        # 你也可以选择仅告警不抛错
        msgs = "; ".join([f"{s}: {m}" for s, m in errs])
        raise RuntimeError(f"fasterq_batch failed for {len(errs)} samples: {msgs}")

    # 转换为 [(srr, Path, Path)]，并按输入顺序排序
    out_pairs = []
    for srr in srr_list:
        if srr in by_srr:
            fq1, fq2 = by_srr[srr]
            out_pairs.append((srr, Path(fq1), Path(fq2)))

    
    return out_pairs

def fasterq_dump_parallel(
    srr_list: list[str],
    out_root: str = "work/fasterq",
    threads_per_job: int = 24,
    mem_gb: int = 8,
    max_workers: int | None = None,
    gzip_output: bool = True,
    backend: str = "process",
    tmp_root: str = "work/tmp"
)-> Dict[str, object]:
    
    total_cores = os.cpu_count() or 8
    out_root = Path(out_root)
    tmp_root = Path(tmp_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 解析二进制绝对路径（关键！）
    fqdump_bin = which_or_find("fasterq-dump")
    print(f"[INFO] fasterq-dump: {fqdump_bin}")

    if max_workers is None:
        import math, os as _os
        max_workers = max(1, math.floor((_os.cpu_count() or 8) / max(1, threads_per_job)))

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    by_srr: Dict[str, Tuple[str, str]] = {}
    errs: List[Tuple[str, str]] = []
    
    Path(tmp_root).mkdir(parents=True, exist_ok=True)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    with Executor(max_workers=max_workers) as ex:
        #futs = {
        #    ex.submit(_fqdump_one, srr, out_root, fqdump_bin, threads_per_job, mem_gb, gzip_output): srr
        #    for srr in srr_list
        futs = {
            ex.submit(
                _work_one_fasterq,
                srr, out_root, threads_per_job, mem_gb, tmp_root, gzip_output
            ): srr for srr in srr_list
        }
        for fut in as_completed(futs):
            srr = futs[fut]
            try:
                srr_id, fq1p, fq2p = fut.result()  # older return
            except TypeError:
                # new return (srr, fq1, fq2)
                srr_id, fq1p, fq2p = fut.result()
            except Exception as e:
                errs.append((srr, str(e)))
                continue
            by_srr[srr_id] = (str(fq1p), str(fq2p))

    if errs:
        print("[SUMMARY] fasterq errors:", errs)
    return {"by_srr": by_srr, "failed": errs}

def _parse_mem_gb(mem_per_job: str | int) -> int:
    """
    兼容传入 '4G' / '8g' / 4 / '4' 等形式，统一返回整数 GB。
    """
    if isinstance(mem_per_job, int):
        return mem_per_job
    s = str(mem_per_job).strip().lower()
    if s.endswith("gb"):
        s = s[:-2]
    elif s.endswith("g"):
        s = s[:-1]
    return int(float(s))


'''
def fasterq_batch(
    srr_list: Sequence[str],
    out_root: str | Path,
    threads: int = 24,
    gzip_output: bool = True,
    mem: str = "8G",
    tmp_root: str = "work/tmp",
    retries: int = 2,
) -> list[Tuple[str, Path, Path]]:
    """
    批量执行 fasterq-dump。
    参数：
        srr_list : SRR 编号列表
        out_root : 输出目录（FASTQ 文件存放位置）
        threads  : 每个样本使用的线程数
        gzip_output : 是否对输出进行 gzip 压缩
        mem : fasterq-dump 内存参数（默认 4G）
        tmp_root : 临时目录
        retries : 单样本失败重试次数
    返回：
        [(srr, fq1_path, fq2_path), ...]
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for srr in srr_list:
        srr, fq1, fq2, status = fasterq_dump_parallel(
            srr=srr,
            outdir=str(out_root),
            threads=threads,
            mem=mem,
            retries=retries,
            tmp_root=str(tmp_root),
        )

        # 如果需要压缩输出（可选）
        if gzip_output:
            import gzip, shutil
            for fq in [fq1, fq2]:
                fq_path = Path(fq)
                if not fq_path.with_suffix(fq_path.suffix + ".gz").exists():
                    with open(fq_path, "rb") as f_in, gzip.open(str(fq_path) + ".gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    fq_path.unlink()
            fq1 = str(Path(fq1).with_suffix(".fastq.gz"))
            fq2 = str(Path(fq2).with_suffix(".fastq.gz"))

        results.append((srr, Path(fq1), Path(fq2)))

    return results
    '''

def make_fasterq_step(
    outdir_pattern: str = "work/fasterq",   # 产出目录（与之前一致）
    threads_per_job: int = 24,
    mem_per_job: str = "4G",               # 兼容旧写法，内部转成 int GB
    max_workers: int | None = None,
    retries: int = 2,                      # 新实现未用到，先保留以免破坏接口
    tmp_root: str = "work/tmp",            # 同上
    backend: str = "process",
    compress_after: bool = True,           # 映射到 gzip_output
    compress_threads: int = 8,             # 新实现未用到，先保留
):
    """
    输入：srr_list（由 pipeline 传入）
    输出：work/fasterq/{SRR}_1.fastq.gz, {SRR}_2.fastq.gz
    验证：两端均存在且 size > 0
    """
    def _cmd(srr_list: List[str], logger=None) -> List[Tuple[str, str, str]]:
        os.makedirs(outdir_pattern, exist_ok=True)

        # 映射参数到新的 fasterq_dump_parallel
        mem_gb = _parse_mem_gb(mem_per_job)

        ret = fasterq_dump_parallel(
            srr_list=srr_list,
            out_root=outdir_pattern,         # 对应 outdir_pattern
            threads_per_job=threads_per_job,
            mem_gb=mem_gb,                   # 统一成 int GB
            max_workers=max_workers,
            gzip_output=compress_after,      # compress_after -> gzip_output
            backend=backend,
        )
        # 规范化输出为 [(srr, fq1, fq2), ...]，供 pipeline 使用
        by_srr = ret.get("by_srr", {})
        products = [(srr, paths[0], paths[1]) for srr, paths in by_srr.items()]
        # 若需要，你也可以在此打印 ret["failed"] 以便日志观察
        if logger and ret.get("failed"):
            for srr, err in ret["failed"]:
                logger.error(f"[fasterq] {srr} failed: {err}")
        return products

    return {
        "name": "fasterq",
        "command": _cmd,  # 接收 srr_list
        "outputs": [f"{outdir_pattern}" + "/{SRR}_1.fastq.gz",
                    f"{outdir_pattern}" + "/{SRR}_2.fastq.gz"],
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fs),
        "takes": "SRR_LIST",
        "yields": "FASTQ_PATHS"
    }