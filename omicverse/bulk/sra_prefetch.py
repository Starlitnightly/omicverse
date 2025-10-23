# sra_prefetch.py Prefetch步骤工厂
from __future__ import annotations
import os, time, subprocess, logging
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence
try:
    from tqdm.auto import tqdm 
except Exception:
    from tqdm import tqdm
    
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from .sra_tools import (get_sra_metadata, human_readable_speed, estimate_remaining_time, record_to_log, get_downloaded_file_size, run_prefetch_with_progress)
except ImportError:
    from sra_tools import (get_sra_metadata, human_readable_speed, estimate_remaining_time, record_to_log, get_downloaded_file_size, run_prefetch_with_progress)
# ---------- 工具函数（按你的实现/命名保留） ----------
def find_sra_file(srr_id: str, output_root: Path) -> Path | None:
    """
    在一系列候选位置查找 {srr_id}.sra：
      1) output_root/SRR/SRR.sra
      2) output_root/SRR.sra
      3) srapath 返回的缓存路径
      4) 常见本地缓存目录 (~/.ncbi/public/sra, ~/ncbi/public/sra)
    找到就返回 Path，否则 None。
    """
    candidates = [
        output_root / srr_id / f"{srr_id}.sra",
        output_root / f"{srr_id}.sra",
    ]
    for p in candidates:
        if p.exists():
            return p

    # 3) srapath
    try:
        cp = subprocess.run(["srapath", srr_id],
                            capture_output=True, text=True, check=True)
        for line in cp.stdout.splitlines():
            p = Path(line.strip())
            if p.exists():
                return p
    except Exception:
        pass

    # 4) 常见缓存根
    home = Path.home()
    for root in [home/".ncbi/public/sra", home/"ncbi/public/sra"]:
        p = root / f"{srr_id}.sra"
        if p.exists():
            return p
    return None
def ensure_link_at(output_path: Path, real_file: Path, logger=None, prefer="symlink") -> Path:
    """
    确保在 output_path 有一个可用的文件指向 real_file：
      - 优先创建符号链接；不支持时退回硬链接/复制
    返回 output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        # 已有就直接用
        return output_path

    try:
        if prefer == "symlink":
            output_path.symlink_to(real_file)
        elif prefer == "hardlink":
            os.link(real_file, output_path)
        else:
            shutil.copy2(real_file, output_path)
        if logger:
            logger.info(f"[LINK] {output_path} -> {real_file}")
    except Exception as e:
        # 兼容不支持软链接的环境：退回拷贝
        if logger:
            logger.warning(f"[LINK] failed ({e}), fallback to copy")
        shutil.copy2(real_file, output_path)
    return output_path
    
def _make_local_logger(name: str, level=logging.INFO) -> logging.Logger:
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True) 
    logger = logging.getLogger(f"sraprefetch.{name}")
    if not logger.handlers:
        logger.setLevel(level)
        fh = logging.FileHandler(log_dir/f"{name}.log", mode="a")
        sh = logging.StreamHandler()
        fmt = logging.Formatter(f"[%(asctime)s] [{name}] %(levelname)s: %(message)s")
        for h in (fh, sh):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger

   ################################################
  #                                              #
 #      兼容Alignment Class与PipLine工厂         #
#                                             #
##############################################

# ---------- For Alignment Class ----------
def _monitor_file_size(path: Path, pbar: tqdm, stop_event: threading.Event, interval: float = 1.0):
    """外部监控文件大小并刷新 tqdm"""
    last = 0
    while not stop_event.is_set():
        try:
            size = path.stat().st_size if path.exists() else 0
            inc = max(0, size - last)
            if inc:
                pbar.update(inc)
                last = size
        except Exception:
            pass
        time.sleep(interval)
    # 收尾再刷一次
    try:
        size = path.stat().st_size if path.exists() else last
        if size > last:
            pbar.update(size - last)
    except Exception:
        pass
    pbar.close()
    
def prefetch_batch(
    srr_list: Sequence[str], 
    out_root: str | Path, 
    threads: int = 4, # 同时下载的 SRR 数
    retries: int = 3, 
    link_mode: str = "symlink",
    cache_root: str | Path = "sra_cache",
):
    """
    Adapter to integrate with Alignment.prefetch().
    For each SRR:
      1) call run_prefetch_with_progress(srr, logger, retries, output_root="sra_cache")
      2) locate the real .sra in the cache
      3) place a link/copy at <out_root>/<SRR>/<SRR>.sra
      并发下载 SRA：对 srr_list 中的样本并行执行 prefetch。
      threads 控制“同时下载任务数”
      下载后在 cache_root 中找到 .sra，再链接/复制到 out_root/<SRR>/<SRR>.sra
    返回：按输入顺序对齐的目标 .sra 路径列表
    """
    out_root = Path(out_root)
    cache_root = out_root 
    out_root.mkdir(parents=True, exist_ok=True)
    
    logger = _make_local_logger("prefetch")
    # 单个任务
    def _worker(pos: int, srr: str) -> tuple[str, Path]:
        
        # 监控的目标文件（与你原函数保持一致的落盘位置）
        srr_dir = cache_root / srr
        srr_dir.mkdir(parents=True, exist_ok=True)
        sra_file = srr_dir / f"{srr}.sra"

        '''# 尝试使用多行进度条位置；不支持时自动降级
        pbar_kwargs = dict(
            desc=f"📥 {srr}",
            total=None,  # 若能拿到预估大小可替换为具体值
            unit="B", unit_scale=True, unit_divisor=1024,
            dynamic_ncols=True, leave=False, mininterval=0.5
        )
        try:
            pbar = tqdm(position=pos, **pbar_kwargs)
        except TypeError:
            pbar = tqdm(**pbar_kwargs)
            
        stop_evt = threading.Event()
        mon = threading.Thread(target=_monitor_file_size, args=(sra_file, pbar, stop_evt), daemon=True)
        mon.start()
        '''
        # 下载
        ok = run_prefetch_with_progress(
            srr_id=srr, logger=logger, retries=retries, output_root=str(cache_root)
        )
        # 停止监控并收尾
        #stop_evt.set()
        #mon.join()
        
        if not ok:
            raise RuntimeError(f"Prefetch failed for {srr}")

        real = find_sra_file(srr_id=srr, output_root=cache_root)
        if not real or not Path(real).exists():
            raise RuntimeError(f"Prefetch done but cannot locate .sra for {srr}")

        dest = out_root / srr / f"{srr}.sra"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ensure_link_at(Path(real), dest, prefer=link_mode, logger=logger)
        return srr, dest

    results_map: dict[str, Path] = {}
    failures: list[tuple[str, Exception]] = []

    # Multiple Download 
    with ThreadPoolExecutor(max_workers=int(threads)) as pool:
        fut_map = {pool.submit(_worker, pos,srr): srr for pos, srr in enumerate(srr_list)}
        for fut in as_completed(fut_map):
            srr = fut_map[fut]
            try:
                srr_id, dest_path = fut.result()
                results_map[srr_id] = dest_path
                logger.info(f"[prefetch] OK: {srr_id} -> {dest_path}")
            except Exception as e:
                failures.append((srr, e))
                logger.error(f"[prefetch] FAIL: {srr}: {e}")

    if failures:
        failed = ", ".join([f"{srr}({repr(err)})" for srr, err in failures])
        raise RuntimeError(f"Prefetch failed for {len(failures)} samples: {failed}")
    # 保持与输入顺序一致
    return [results_map[s] for s in srr_list if s in results_map]

# ---------- “步骤工厂”：需要时在编排模块里调用 ----------
def make_prefetch_step(output_pattern: str = "sra_cache/{SRR}/{SRR}.sra", validation=None):
    def _safe_valid(fs: list[str]) -> bool:
        # fs 只会有一个：期望路径，但我们用 find_sra_file 来兜底
        try:
            f = Path(fs[0])
            if f.exists() and f.stat().st_size > 1_000_000:
                return True
            # 用定位函数检查真实缓存位置
            real = find_sra_file(f.stem, f.parent.parent)  # stem=SRRxxxxxx, parent.parent=output_root
            return bool(real and real.exists() and real.stat().st_size > 1_000_000)
        except Exception:
            return False

    return {
        "name": "prefetch",
        "command": run_prefetch_with_progress,
        "outputs": [output_pattern],
        "validation": validation or _safe_valid,
    }