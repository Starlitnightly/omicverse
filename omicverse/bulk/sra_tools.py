import os,sys,time, shutil, hashlib, subprocess, requests,argparse
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import Tuple, Optional, List, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from math import floor

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from filelock import FileLock
import logging


# ================= Safer single-sample worker =================
def _sra_tools_version() -> str | None:
    p = shutil.which("fasterq-dump")
    if not p:
        return None
    try:
        out = subprocess.check_output(["fasterq-dump", "--version"], text=True, stderr=subprocess.STDOUT)
        # expected like: fasterq-dump : 3.0.8
        for line in out.splitlines():
            if "fasterq-dump" in line and ":" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return None

def _exists_fastqs(outdir: str, srr: str) -> bool:
    return (Path(outdir)/f"{srr}_1.fastq").exists() and (Path(outdir)/f"{srr}_2.fastq").exists()

def _run(cmd: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(">> " + " ".join(cmd) + "\n")
        subprocess.run(cmd, check=True, stdout=fh, stderr=fh)

def _try_direct_fasterq(srr: str, outdir: str, threads: int, mem: str, parallel: bool, tmpdir: str | None, log: Path):
    cmd = ["fasterq-dump", srr, "-O", outdir, "-e", str(threads), "--mem", str(mem), "--split-files", "--force", "--progress"]
    if parallel:
        cmd.insert(2, "-p")
    if tmpdir:
        cmd += ["--temp", tmpdir]
    _run(cmd, log)

def _prefetch_and_convert(srr: str, outdir: str, threads: int, mem: str, parallel: bool, tmpdir: str | None, log: Path):
    # prefetch to cache, then validate, then dump from local
    _run(["prefetch", srr, "--progress"], log)
    # optional validate (helpful for corrupt/incomplete)
    try:
        _run(["vdb-validate", srr], log)
    except Exception:
        # don't hard-fail: some records may warn; we still attempt dump
        pass
    # Now convert
    _try_direct_fasterq(srr, outdir, threads, mem, parallel, tmpdir, log)

def _fasterq_run_one_resilient(
    srr: str,
    outdir: str,
    threads: int = 24,
    mem: str = "4G",
    retries: int = 2,
    tmp_root: str = "work/tmp"  # place for --temp
):
    """
    Resilient fasterq-dump:
    1) skip if outputs exist
    2) try direct download
    3) on error: prefetch -> (validate) -> local fasterq-dump
    4) on repeated error: lower threads, disable -p, fresh tmp
    """
    os.makedirs(outdir, exist_ok=True)
    fq1 = os.path.join(outdir, f"{srr}_1.fastq")
    fq2 = os.path.join(outdir, f"{srr}_2.fastq")
    log = Path(outdir) / f"{srr}.fasterq.log"

    if _exists_fastqs(outdir, srr):
        return (srr, fq1, fq2, "SKIP")

    ver = _sra_tools_version()
    if ver is None:
        raise FileNotFoundError("fasterq-dump not found in PATH")
    try:
        major = int(ver.split(".")[0])
        if major < 2:
            print(f"[WARN] SRA-tools version looks very old: {ver}", file=sys.stderr)
    except Exception:
        pass

    last_err = None
    # 1st pass: direct (parallel on), then fallback (prefetch), then degraded (no -p, fewer threads)
    plans = [
        ("direct_parallel", dict(parallel=True,  t=threads, mem=mem)),
        ("prefetch_parallel", dict(parallel=True,  t=threads, mem=mem)),
        ("prefetch_serial",   dict(parallel=False, t=max(1, threads//2), mem=mem)),
    ]

    for name, cfg in plans:
        for attempt in range(1, retries+1):
            try:
                # isolate temp per attempt to avoid stale temp issues
                tmpdir = os.path.join(tmp_root, f"{srr}.{name}.try{attempt}")
                Path(tmpdir).mkdir(parents=True, exist_ok=True)

                if "prefetch" in name:
                    _prefetch_and_convert(srr, outdir, cfg["t"], cfg["mem"], cfg["parallel"], tmpdir, log)
                else:
                    _try_direct_fasterq(srr, outdir, cfg["t"], cfg["mem"], cfg["parallel"], tmpdir, log)

                if not _exists_fastqs(outdir, srr):
                    raise RuntimeError("Outputs missing after fasterq-dump.")
                return (srr, fq1, fq2, "OK")
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)  # brief backoff
        # next plan

    raise RuntimeError(f"[{srr}] fasterq-dump ultimately failed: {last_err}")

# =================Utilities: gzip with integrity check  ====================
def _gzip_available() -> tuple[str, bool]:
    """
    Return (exe_path, is_pigz) for pigz or gzip; prefer pigz.
    """
    pigz = shutil.which("pigz")
    if pigz:
        return pigz, True
    gz = shutil.which("gzip")
    if gz:
        return gz, False
    raise FileNotFoundError("Neither pigz nor gzip is available in PATH.")

def _gzip_test_ok(path_gz: Path) -> bool:
    """
    Test gzip integrity via 'gzip -t' or 'pigz -t'. Return True if OK.
    """
    exe, is_pigz = _gzip_available()
    cmd = [exe, "-t", str(path_gz)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _gzip_compress(path_fastq: str, threads: int = 4) -> str:
    """
    Compress a .fastq to .fastq.gz using pigz (if available) or gzip.
    Replaces the input file (no second copy). Verifies with gzip -t.
    Returns the .gz path (string). Raises on failure.
    """
    exe, is_pigz = _gzip_available()
    p = Path(path_fastq)
    if not p.exists():
        raise FileNotFoundError(f"File to compress not found: {p}")
    # If already gz, just test and return
    if p.suffix == ".gz":
        if not _gzip_test_ok(p):
            raise RuntimeError(f"Gzip integrity test failed: {p}")
        return str(p)

    # compress in place
    cmd = [exe]
    if is_pigz:
        cmd += ["-p", str(max(1, threads))]
    cmd += ["-f", str(p)]
    subprocess.run(cmd, check=True)

    gz = Path(str(p) + ".gz")
    if not gz.exists():
        raise RuntimeError(f"Gzip did not produce: {gz}")

    if not _gzip_test_ok(gz):
        # attempt to remove bad output to avoid confusion
        try:
            gz.unlink(missing_ok=True)
        finally:
            raise RuntimeError(f"Gzip integrity test failed: {gz}")

    return str(gz)

# ================= 获取 SRA 文件元信息（大小 + MD5） =================
def get_sra_metadata(srr_id):
    ena_url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={srr_id}&result=read_run&fields=run_accession,fastq_bytes,fastq_md5"
    try:
        response = requests.get(ena_url)
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("返回数据格式异常")
        data_line = lines[1]
        cols = data_line.split('\t')
        size_str, md5_str = cols[1], cols[2]
        sizes = [int(x) for x in size_str.split(';') if x.isdigit()]
        md5s = [x.strip() for x in md5_str.split(';') if x.strip()]
        return {'size': sum(sizes), 'md5s': md5s}
    except Exception as e:
        print(f"获取SRA元信息失败: {str(e)}")
        return {'size': None, 'md5s': []}

def calculate_md5(filepath, chunk_size=8192):
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"计算MD5失败: {e}")
        return None

def human_readable_speed(speed_bps):
    units = ['B/s', 'KB/s', 'MB/s', 'GB/s']
    unit_index = 0
    while speed_bps >= 1024 and unit_index < 3:
        speed_bps /= 1024
        unit_index += 1
    return f"{speed_bps:.1f} {units[unit_index]}"

def estimate_remaining_time(current, total, speed):
    if total is None or speed == 0:
        return "--:--"
    remaining = (total - current) / speed
    return time.strftime("%H:%M:%S", time.gmtime(remaining))

def get_downloaded_file_size(output_root: Path, srr_id: str):
    tmp_file = output_root / srr_id / f"{srr_id}.sra.tmp"
    return tmp_file.stat().st_size if tmp_file.exists() else 0

def record_to_log(srr_id: str, status: str):
    log_file = "success.log" if status == "success" else "fail.log"
    with open(log_file, "a") as f:
        f.write(srr_id + "\n")

def setup_logger(srr_id):
    logger = logging.getLogger(srr_id)
    logger.setLevel(logging.DEBUG)
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{srr_id}.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(f"[%(asctime)s] [{srr_id}] %(levelname)s: %(message)s"))

    if not logger.handlers:
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(f"[%(asctime)s] [{srr_id}] %(levelname)s: %(message)s"))
        logger.addHandler(stream_handler)

    return logger

# ====================Prefetch Setting==============================
def resolve_sra_path(srr_id: str) -> Path | None:
    """
    用 srapath 解析 SRR 对应的真实 .sra 路径（可能在 ~/.ncbi/public/sra/）。
    返回 Path 或 None。
    """
    try:
        out = subprocess.check_output(["srapath", srr_id], text=True).strip().splitlines()
    except Exception:
        return None
    for line in out:
        p = Path(line.strip())
        if p.suffix.lower() == ".sra" and p.exists():
            return p
    for line in out:
        p = Path(line.strip()) / f"{srr_id}.sra"
        if p.exists():
            return p
    return None

def ensure_symlink(target: Path, real: Path) -> None:
    """
    在 target 位置建立到 real 的软链（若不支持软链则复制），确保 target 所在目录存在。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # 已经是同一个目标则直接返回
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == real.resolve():
                return
        except Exception:
            pass
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    try:
        target.symlink_to(real)
    except OSError:
        shutil.copy2(real, target)

def check_step_completed(step: dict, srr_id: str, logger=None) -> tuple[bool, list[str]]:
    """
    通用的“此步骤是否完成”检查：
      - 基于 step["outputs"] 渲染出期望产物路径
      - 调用 step["validation"](outputs) 判断
      - 针对 prefetch 步骤做稳健化：如果期望 .sra 不在指定 cache，
        则用 srapath 找真实文件，存在则补建软链/复制到期望位置，再判定完成
    返回：(done: bool, outputs: List[str])
    """
    outs = _render_paths(step["outputs"], SRR=srr_id)

    # 先直接跑 validation（你之前已有 _safe_call_validation 也可以复用）
    try:
        done = step["validation"](outs)
    except Exception as e:
        if logger:
            logger.warning(f"[WARN] validation raised: {e}")
        done = all(Path(p).exists() for p in outs)

    # 额外稳健化：prefetch 步骤（通常只有 1 个 .sra 期望路径）
    # 约定：prefetch 的 outputs 模板里包含 “.sra”
    is_prefetch = any(str(p).endswith(".sra") for p in outs)
    if not done and is_prefetch:
        # outs[0] 就是期望的 sra_cache/SRR/SRR.sra
        expected = Path(outs[0])
        real = resolve_sra_path(srr_id)
        if real and real.exists():
            # 补建软链/复制
            ensure_symlink(expected, real)
            # 重新判定
            try:
                done = step["validation"]([str(expected)])
            except Exception:
                done = expected.exists() and expected.stat().st_size > 1_000_000
            if done and logger:
                logger.info(f"[FIX] {srr_id}: found real .sra at {real}, linked to {expected}")

    return done, outs

def run_prefetch_with_progress(srr_id, logger, retries=3, output_root: str | Path = "sra_cache"):
    """
    稳健的 prefetch：
      1) 目标路径：sra_cache/<SRR>/<SRR>.sra
      2) 已存在则先 vdb-validate，通过即跳过
      3) 使用 --verify yes --resume yes 下载
      4) 下载结束统一 vdb-validate 作为最终校验
    说明：ENA 的 fastq_md5 是 FASTQ 的 MD5，不适用于 .sra；因此不再用其做 .sra 校验
    """
    # 根目录与目标路径
    output_root = Path(output_root)
    srr_dir = output_root / srr_id
    output_path = srr_dir / f"{srr_id}.sra"
    srr_dir.mkdir(exist_ok=True, parents=True)
    # （可选）仅作显示用途的预估总大小；拿不到也不影响流程
    try:
        meta = get_sra_metadata(srr_id)
        total_size = meta.get("size") if isinstance(meta, dict) else None
        if not isinstance(total_size, int) or total_size <= 0:
            total_size = None
    except Exception:
        total_size = None

    # 如果文件已存在，先做官方校验；通过就跳过
    if output_path.exists():
        try:
            subprocess.run(["vdb-validate", str(output_path)],
                           check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            logger.info(f"{srr_id}.sra 已存在且通过 vdb-validate，跳过下载")
            record_to_log(srr_id, "success")
            return True
        except subprocess.CalledProcessError:
            logger.info(f"{srr_id}.sra 存在但未通过 vdb-validate，将尝试重新下载")

    # 进度条（拿不到 total_size 就显示为“未知总量”）
    downloaded_size = get_downloaded_file_size(output_root, srr_id)
    bar_total = total_size if isinstance(total_size, int) and total_size > 0 else None
    pbar = tqdm(
        desc=f"\U0001F4E5 Prefetch {srr_id}",
        total=total_size,
        initial=downloaded_size if bar_total else 0,
        unit='B', unit_scale=True, unit_divisor=1024,
        dynamic_ncols=True, leave=False, mininterval=0.25,
    )

    last_err = None
    for attempt in range(retries):
        try:
            # 清理锁文件（偶发残留）
            ext_lock = output_path.with_suffix('.sra.lock')
            if ext_lock.exists():
                ext_lock.unlink()
                logger.warning(f"清除外部锁文件: {ext_lock}")
            inner_lock = srr_dir / f"{srr_id}.sra.lock"
            if inner_lock.exists():
                inner_lock.unlink()
                logger.warning(f"清除内部锁文件: {inner_lock}")
            
            start_time = time.time()
            # 启动 prefetch
            proc = subprocess.Popen(
                [
                    "prefetch", srr_id,
                    "--output-directory", str(output_root),
                    #"--verify", "yes",     # 官方完整性校验
                    #"--resume", "yes",     # 断点续传
                    #"--progress"           # 控制台进度（也会写到 stdout）
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )

            # 轮询更新进度
            
            while True:
                # 读 stdout（可选：记录到 logger）
                last_size = downloaded_size
                prev_size = last_size
                current_size = get_downloaded_file_size(output_root, srr_id)
                
                if proc.stdout:
                    line = proc.stdout.readline()
                    if line:
                        logger.debug(line.strip())

                # 更新进度：读取 .tmp 文件大小（prefetch 的写入临时）
                cur_size = current_size

                if cur_size > last_size:
                    delta = cur_size - last_size
                    pbar.update(delta)  # 没 total 时不累加 total，只显示 desc
                    # 可选显示速度/ETA（total 不确定时仅显示速度）
                    speed = delta / 0.25
                    
                    eta_str = estimate_remaining_time(cur_size, bar_total, speed) if bar_total else ""
                    pbar.set_postfix_str(
                        f"{human_readable_speed(speed)}" + (f" | 剩余: {eta_str}" if eta_str else "")
                    )
                    last_size = cur_size

                if proc.poll() is not None:
                    break

                time.sleep(0.25)

            # 读取剩余输出，检查退出码
            stdout_rest, _ = proc.communicate()
            if proc.returncode != 0:
                logger.error(f"prefetch 失败 (退出码 {proc.returncode})\n{stdout_rest}")
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

            # 至此 prefetch 正常退出，关闭进度条
            pbar.close()

            # 确认目标文件存在
            #if not output_path.exists():
             #   logger.error(f"{srr_id}.sra 未找到：{output_path}")
             #   last_err = FileNotFoundError(str(output_path))
              #  continue

            # 结束后统一做 vdb-validate（**你要的校验就加在这里**）
            val = subprocess.run(["vdb-validate", str(output_path)], capture_output=True, text=True)
            if val.returncode == 0:
                logger.info(f"{srr_id}.sra 下载成功并通过 vdb-validate")
                record_to_log(srr_id, "success")
                return True
            else:
                logger.warning(
                    f"{srr_id}.sra 下载完成但 vdb-validate 失败；输出：\n{val.stdout}\n{val.stderr}"
                )
                last_err = RuntimeError("vdb-validate failed")
                # 失败则进入下一轮重试（如果还有）

        except subprocess.CalledProcessError as e:
            last_err = e
            logger.error(f"Prefetch失败 (尝试 {attempt}/{retries}): {e}")
            time.sleep(5)

    # 全部重试失败
    pbar.close()
    logger.error(f"{srr_id} 最终失败：{last_err}")
    record_to_log(srr_id, "fail")
    return False

COMMAND_STEPS = [
    {
        'name': 'prefetch',
        'command': run_prefetch_with_progress,
        'outputs': ['{SRR}/{SRR}.sra'],
        'validation': lambda fs: all(os.path.getsize(f) > 1_000_000 for f in fs)
    }
]

def process_srr(srr_id, max_retry=3):
    logger = setup_logger(srr_id)
    if os.path.exists("success.log") and srr_id in Path("success.log").read_text().splitlines():
        logger.info(f"{srr_id} 已在 success.log 中记录，跳过")
        return True

    logger.info(f"开始处理 {srr_id}")
    start_time = time.time()
    try:
        for step in COMMAND_STEPS:
            step_name = step['name']
            lock_file = f"{srr_id}_{step_name}.lock"
            with FileLock(lock_file + ".lock"):
                if not step['command'](srr_id, logger):
                    raise RuntimeError("Prefetch步骤失败")
                if not check_step_completed(step, srr_id, logger):
                    raise RuntimeError(f"最终验证失败: {step_name}")

        duration = time.time() - start_time
        logger.info(f"{srr_id} 下载完成，总耗时: {duration:.1f} 秒")
        record_to_log(srr_id, "success")
        return True
    except Exception as e:
        logger.exception(f"{srr_id} 处理失败: {e}")
        record_to_log(srr_id, "fail")
        return False


# ================= 获取 SRA  =================
def raw_data_downloader(sample_csv="sample_list.csv", start=0, end=100, max_workers=10, resume=True):
    try:
        df = pd.read_csv(sample_csv)
        srr_list = df["Run"].tolist()
        if resume and os.path.exists("success.log"):
            completed = set(Path("success.log").read_text().splitlines())
            srr_list = [s for s in srr_list if s not in completed]
        if end:
            srr_list = srr_list[start:end]
        else:
            srr_list = srr_list[start:]
        print(f"待处理样本数: {len(srr_list)}")
    except Exception as e:
        print(f"读取样本列表失败: {str(e)}")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_srr, srr): srr for srr in srr_list}
        with tqdm(total=len(srr_list), desc="处理进度") as pbar:
            for future in concurrent.futures.as_completed(futures):
                srr_id = futures[future]
                try:
                    success = future.result()
                    status = "成功" if success else "失败"
                    print(f"{srr_id}: {status}")
                except Exception as e:
                    print(f"{srr_id}: 异常终止 - {str(e)}")
                finally:
                    pbar.update(1)



# ================= Bash Run  =================

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ================= 数据前处理 fasterq-dump  ================= 
#
def pyfasterq_dump(srr, outdir):
    """
    Convert SRA accession to FASTQ using fasterq-dump with parallel threads and memory limit.
    Produces paired-end FASTQ files in the specified output directory.
    """
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "fasterq-dump", srr,        # SRA accession ID
        "-p",                       # enable parallel processing
        "-O", outdir,               # output directory
        "-e", "24",                 # use 24 threads
        "--mem", "4G",              # allocate 4 GB memory
        "--split-files"             # separate paired-end reads
    ]

    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Construct expected output filenames
    fq1 = os.path.join(outdir, f"{srr}_1.fastq")
    fq2 = os.path.join(outdir, f"{srr}_2.fastq")
    return [fq1, fq2]


# ======== 顶层 worker（可被pickle）========

'''
def _fasterq_run_one(srr: str,
                     outdir: str,
                     threads: int = 24,
                     mem: str = "4G",
                     retries: int = 2):
    """
    单个 SRR 的 fasterq-dump 任务。
    若已存在 {srr}_1.fastq / {srr}_2.fastq 则跳过。
    """
    import os, time, subprocess, shutil

    os.makedirs(outdir, exist_ok=True)
    fq1 = os.path.join(outdir, f"{srr}_1.fastq")
    fq2 = os.path.join(outdir, f"{srr}_2.fastq")

    # 已有产物：跳过
    if os.path.exists(fq1) and os.path.exists(fq2):
        return (srr, fq1, fq2, "SKIP")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            cmd = [
                "fasterq-dump", srr,
                "-p",
                "-O", outdir,
                "-e", str(threads),
                "--mem", str(mem),
                "--split-files"
            ]
            print(">>", " ".join(cmd))
            # 确认二进制可见
            if not shutil.which("fasterq-dump"):
                raise FileNotFoundError("fasterq-dump not found in PATH")

            subprocess.run(cmd, check=True)

            # 基本校验
            if not (os.path.exists(fq1) and os.path.exists(fq2)):
                raise RuntimeError(f"Missing outputs after fasterq-dump: {fq1}, {fq2}")
            return (srr, fq1, fq2, "OK")
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * attempt)
    # 重试用尽
    raise RuntimeError(f"[{srr}] fasterq-dump failed after {retries} attempts: {last_err}")
'''
def _fasterq_run_one_resilient(
    srr: str,
    outdir: str,
    threads: int = 24,
    mem: str = "4G",
    retries: int = 2,
    tmp_root: str = "work/tmp",
    compress_after: bool = True,          # NEW: auto-compress after dump
    compress_threads: int = 8             # NEW: pigz threads if available
):
    """
    Resilient fasterq-dump:
      1) skip if outputs already exist (.fastq or .fastq.gz)
      2) try direct stream
      3) fallback: prefetch -> (validate) -> local fasterq-dump
      4) degrade: fewer threads, no -p
      5) optional: gzip compress outputs + integrity check + remove .fastq
    Returns (srr, fq1_out, fq2_out, status) where fq*_out may be .fastq.gz
    """
    os.makedirs(outdir, exist_ok=True)
    fq1 = os.path.join(outdir, f"{srr}_1.fastq")
    fq2 = os.path.join(outdir, f"{srr}_2.fastq")
    fq1_gz = fq1 + ".gz"
    fq2_gz = fq2 + ".gz"
    log = Path(outdir) / f"{srr}.fasterq.log"

    def _outputs_exist() -> bool:
        # consider either uncompressed or compressed pairs as "existing"
        return ((Path(fq1).exists() and Path(fq2).exists()) or
                (Path(fq1_gz).exists() and Path(fq2_gz).exists()))

    # If already produced (gz or fq), skip
    if _outputs_exist():
        # normalize return to gz paths if present, else fq paths
        out1 = fq1_gz if Path(fq1_gz).exists() else fq1
        out2 = fq2_gz if Path(fq2_gz).exists() else fq2
        return (srr, out1, out2, "SKIP")

    ver = _sra_tools_version()
    if ver is None:
        raise FileNotFoundError("fasterq-dump not found in PATH")

    last_err = None
    plans = [
        ("direct_parallel",   dict(parallel=True,  t=threads,                mem=mem)),
        ("prefetch_parallel", dict(parallel=True,  t=threads,                mem=mem)),
        ("prefetch_serial",   dict(parallel=False, t=max(1, threads // 2),   mem=mem)),
    ]

    for name, cfg in plans:
        for attempt in range(1, retries + 1):
            try:
                tmpdir = os.path.join(tmp_root, f"{srr}.{name}.try{attempt}")
                Path(tmpdir).mkdir(parents=True, exist_ok=True)

                if "prefetch" in name:
                    _prefetch_and_convert(srr, outdir, cfg["t"], cfg["mem"], cfg["parallel"], tmpdir, log)
                else:
                    _try_direct_fasterq(srr, outdir, cfg["t"], cfg["mem"], cfg["parallel"], tmpdir, log)

                # basic existence check (uncompressed)
                if not (Path(fq1).exists() and Path(fq2).exists()):
                    # some single-end runs produce only _1; consider extending if needed
                    raise RuntimeError("Outputs missing after fasterq-dump.")

                # Post-step compression
                out1, out2 = fq1, fq2
                if compress_after:
                    out1 = _gzip_compress(fq1, threads=compress_threads)
                    out2 = _gzip_compress(fq2, threads=compress_threads)
                    # after _gzip_compress, .fastq is removed and .gz integrity tested

                return (srr, out1, out2, "OK")

            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)

    raise RuntimeError(f"[{srr}] fasterq-dump ultimately failed: {last_err}")

# ======== 并行调度器（对外 API）========
def pyfasterq_dump_parallel(
    srr_list: list[str],
    outdir: str,
    threads_per_job: int = 24,     # 传给 fasterq-dump 的 -e
    mem_per_job: str = "4G",       # 传给 fasterq-dump 的 --mem
    max_workers: int | None = None,
    retries: int = 2,
    tmp_root: str = "work/tmp",
    compress_after: bool = True,      
    compress_threads: int = 8,         
    backend: str = "process",      # "process" 或 "thread"
):
    """
    批量并行执行 fasterq-dump（与单个 pyfasterq_dump 逻辑一致）。
    - 自动跳过已存在 {SRR}_1.fastq / {SRR}_2.fastq
    - 支持失败重试
    - max_workers 缺省为 (CPU核数 // threads_per_job)

    返回:
      {"success": [(srr, fq1, fq2, status), ...], "failed": [(srr, errmsg), ...]}
    """
    

    total_cores = os.cpu_count() or 8
    if max_workers is None:
        max_workers = max(1, floor(total_cores / max(1, threads_per_job)))

    print(f"[INFO] CPU cores={total_cores}, per-job threads={threads_per_job}, max parallel={max_workers}")
    print(f"[INFO] fasterq-dump path: {shutil.which('fasterq-dump')}")
    print(f"[INFO] Python exec: {sys.executable}")

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    results, errors = [], []
    with Executor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_fasterq_run_one_resilient,
                      srr, outdir,
                      threads_per_job, mem_per_job,
                      retries, tmp_root,
                      compress_after, compress_threads): srr
            for srr in srr_list
        }
        for fut in as_completed(futs):
            srr = futs[fut]
            try:
                res = fut.result()  # (srr, fq1_out, fq2_out, status) | .gz if compressed
                results.append(res)
                print(f"[{res[3]}] {srr} -> {res[1]}, {res[2]}")
            except Exception as e:
                errors.append((srr, str(e)))
                print(f"[ERR] {srr}: {e}")

    print(f"[SUMMARY] OK={len([r for r in results if r[3]=='OK'])}, "
          f"SKIP={len([r for r in results if r[3]=='SKIP'])}, "
          f"Failed={len(errors)}")
    return {"success": results, "failed": errors}

'''
how to use

srrs = ["SRR12544419", "SRR12544565", "SRR12544566"]

ret = pyfasterq_dump_parallel(
    srr_list=srrs,
    outdir="work",
    threads_per_job=24,     # passed to fasterq-dump -e
    mem_per_job="4G",
    # max_workers=8,        # optional; defaults to cores // threads_per_job
    retries=2,
    tmp_root="work/tmp",
    backend="process",
    compress_after=True,     # turn on gzip
    compress_threads=8       # pigz -p 8 (if pigz available)
)

# outputs for each SRR are .fastq.gz paths (if compressed)

'''

# ===============数据质控===================

def require_tool(name, hint=""):
    import shutil
    path = shutil.which(name)
    if path:
        print(f"[OK] {name} found at {path}")
        return path
    else:
        raise FileNotFoundError(f"{name} not found. {hint}")


def fastp_clean(fq1, fq2, sample, out_dir, threads = "12" ):
    # 单一数据的处理 对于多数据并行使用 fastp_clean_parallel(),
    outdir = os.path.join(out_dir, "fastp"); os.makedirs(outdir, exist_ok=True)
    #require_tool("fastp", "Try: conda install -c bioconda fastp")
    if fq2:  # paired
        out1 = os.path.join(outdir, f"{sample}_1.clean.fq.gz")
        out2 = os.path.join(outdir, f"{sample}_2.clean.fq.gz")
        run(["fastp","-i",fq1,"-I",fq2,
             "-o",out1,"-O",out2,
             "-w",threads,
             "-j",os.path.join(outdir,f"{sample}.json"),
             "-h",os.path.join(outdir,f"{sample}.html")])
        # verify outputs
        if not (os.path.exists(out1) and os.path.exists(out2)):
            raise RuntimeError(f"fastp finished but outputs missing: {out1}, {out2}")
        return out1, out2
    else:
        out1 = os.path.join(outdir, f"{sample}.clean.fq.gz")
        run(["fastp","-i",fq1,
             "-o",out1,
             "-w",threads,
             "-j",os.path.join(outdir,f"{sample}.json"),
             "-h",os.path.join(outdir,f"{sample}.html")])
        if not os.path.exists(out1):
            raise RuntimeError(f"fastp finished but output missing: {out1}")
        return out1, ""

from concurrent.futures import ProcessPoolExecutor, as_completed
from math import floor

# ---- 顶层可pickle的worker ----
def _fastp_run_one(sample: str, outdir: str, work_dir: str, fastp_threads: int, retries: int = 2):
    """
    单样本清洗的子进程任务。需要在模块顶层定义，便于 ProcessPoolExecutor pickle。
    依赖同模块中的 fastp_clean(fq1, fq2, sample, work_dir, threads=...).
    """
    import os, time

    fq1 = os.path.join(outdir, f"{sample}_1.fastq")
    fq2 = os.path.join(outdir, f"{sample}_2.fastq")
    if not os.path.exists(fq1):
        raise FileNotFoundError(f"[{sample}] FASTQ not found: {fq1}")
    if fq2 and not os.path.exists(fq2):
        raise FileNotFoundError(f"[{sample}] FASTQ not found: {fq2}")

    # 已完成则跳过（与 fastp_clean 输出命名保持一致）
    out1 = os.path.join(work_dir, f"{sample}_1.clean.fq.gz")
    out2 = os.path.join(work_dir, f"{sample}_2.clean.fq.gz")
    if os.path.exists(out1) and os.path.exists(out2):
        return sample, out1, out2, "SKIP"

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # 直接调用你已有的 fastp_clean（确保它是模块级函数）
            fq1_clean, fq2_clean = fastp_clean(fq1, fq2, sample, work_dir, threads=fastp_threads)
            if not os.path.exists(fq1_clean) or not os.path.exists(fq2_clean):
                raise RuntimeError(f"[{sample}] Missing output after fastp_clean.")
            return sample, fq1_clean, fq2_clean, "OK"
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * attempt)
    # 重试用尽
    raise RuntimeError(f"[{sample}] fastp_clean failed after {retries} attempts: {last_err}")

# ---- 并行调度器（对外API）----
def fastp_clean_parallel(
    samples: list[str],
    outdir: str,
    work_dir: str,
    fastp_threads: int = 4,
    max_workers: int | None = None,
    retries: int = 2,
    backend: str = "process",   # 可选 "process" 或 "thread"
):
    """
    并行批量运行 fastp_clean（对单个样本仍可用原 fastp_clean）。

    参数
    ----
    samples: SRR 列表
    outdir: 原始 FASTQ 所在目录
    work_dir: 清洗输出目录（fastp_clean里会用到）
    fastp_threads: 传给 fastp 的 -w
    max_workers: 并发度（默认=CPU核数/fastp_threads）
    retries: 每个样本失败重试次数
    backend: "process"（默认）或 "thread"；若仍遇到pickling问题可临时用 "thread"
    """
    import os, sys, shutil
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from math import floor

    total_cores = os.cpu_count() or 8
    if max_workers is None:
        max_workers = max(1, floor(total_cores / fastp_threads))

    print(f"[INFO] CPU cores={total_cores}, per fastp threads={fastp_threads}, max parallel={max_workers}")
    print(f"[INFO] fastp path: {shutil.which('fastp')}")
    print(f"[INFO] Python exec: {sys.executable}")

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    results, errors = [], []
    with Executor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_fastp_run_one, s, outdir, work_dir, fastp_threads, retries): s
            for s in samples
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

    print(f"[SUMMARY] Completed={len([r for r in results if r[3]=='OK'])}, "
          f"Skipped={len([r for r in results if r[3]=='SKIP'])}, "
          f"Failed={len(errors)}")
    return {"success": results, "failed": errors}
