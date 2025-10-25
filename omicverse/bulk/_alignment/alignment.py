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

import os
import json
import re
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence, Tuple, Optional, Dict, Any, List, Union

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

try:
    from .iseq_handler import ISeqHandler
except Exception:
    from iseq_handler import ISeqHandler

# 设置日志
logger = logging.getLogger(__name__)


class DownloadProgressBar:
    """实时下载进度条显示类"""

    def __init__(self, total_size=None, desc="Downloading"):
        self.total_size = total_size
        self.desc = desc
        self.current_size = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.last_size = 0
        self.active = True
        self.lock = threading.Lock()

    def update(self, current_size):
        """更新进度"""
        with self.lock:
            self.current_size = current_size
            current_time = time.time()

            # 每0.5秒更新一次显示，避免过于频繁
            if current_time - self.last_update >= 0.5:
                self._display_progress(current_time)
                self.last_update = current_time
                self.last_size = current_size

    def _display_progress(self, current_time):
        """显示进度条"""
        if not self.active:
            return

        elapsed = current_time - self.start_time
        if elapsed == 0:
            return

        # 计算速度 (bytes/second)
        speed = (self.current_size - self.last_size) / (current_time - self.last_update) if current_time > self.last_update else 0

        # 格式化文件大小
        current_str = self._format_size(self.current_size)
        speed_str = self._format_size(speed) + "/s"

        if self.total_size and self.total_size > 0:
            # 如果有总大小，显示百分比进度条
            percentage = (self.current_size / self.total_size) * 100
            bar_length = 40
            filled_length = int(bar_length * self.current_size // self.total_size)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            total_str = self._format_size(self.total_size)
            eta = (self.total_size - self.current_size) / speed if speed > 0 else 0
            eta_str = self._format_time(eta)

            print(f'\r{self.desc}: {percentage:.1f}%|{bar}| {current_str}/{total_str} [{speed_str}, ETA: {eta_str}]', end='', flush=True)
        else:
            # 如果没有总大小，只显示当前大小和速度
            print(f'\r{self.desc}: {current_str} [{speed_str}]', end='', flush=True)

    def _format_size(self, size):
        """格式化文件大小"""
        if size == 0:
            return "0B"

        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size_float = float(size)

        while size_float >= 1024 and unit_index < len(units) - 1:
            size_float /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(size_float)}{units[unit_index]}"
        else:
            return f"{size_float:.1f}{units[unit_index]}"

    def _format_time(self, seconds):
        """格式化时间"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"

    def finish(self):
        """完成进度条"""
        with self.lock:
            self.active = False
            # 显示最终状态
            total_time = time.time() - self.start_time
            total_str = self._format_size(self.current_size)
            avg_speed = self.current_size / total_time if total_time > 0 else 0
            avg_speed_str = self._format_size(avg_speed) + "/s"

            if self.total_size and self.current_size >= self.total_size:
                print(f'\r{self.desc}: 100%|{"█" * 40}| {total_str}/{total_str} [{avg_speed_str}, Total: {self._format_time(total_time)}]')
            else:
                print(f'\r{self.desc}: {total_str} [{avg_speed_str}, Total: {self._format_time(total_time)}]')
            print()  # 换行

    def stop(self):
        """停止进度条"""
        self.active = False


@dataclass
class AlignmentConfig:
    # IO roots
    work_root: Path = Path("work")
    meta_root: Path = field(init=False)
    prefetch_root: Path = field(init=False)
    fasterq_root: Path = field(init=False)
    qc_root: Path = field(init=False)
    align_root: Path = field(init=False)     # placeholder if you later add HISAT2/STAR
    counts_root: Path = field(init=False)

    star_index_root: Path = field(init=False)        # 索引根目录（和 star_tools 设定一致）
    star_align_root: Path  = field(init=False)   # STAR 输出根目录

    # Basic prefetch configuration (no mirror switching)
    prefetch_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,                    # Enable prefetch
        "location": "ncbi",                 # Default location: ncbi
        "transport": "fasp",                # Default transport: fasp
        "retries": 3                        # Number of retry attempts
    })

    def __post_init__(self):
        
        if not isinstance(self.work_root, Path):
            self.work_root = Path(self.work_root)
        self.meta_root = self.work_root / "meta"
        self.prefetch_root = self.work_root / "prefetch"
        self.fasterq_root = self.work_root / "fasterq"
        self.qc_root = self.work_root / "fastp"
        self.align_root = self.work_root / "align" # other align method预留位
        self.counts_root = self.work_root / "counts"
        self.star_align_root = self.work_root / "star"
        self.star_index_root = self.work_root / "index"

        # Automatically enable iseq if download_method is set to "iseq"
        if self.download_method == "iseq":
            self.iseq_enabled = True
    

    # Resources
    threads: int = 16
    memory: str = "8G"
    genome: Literal["human", "mouse", "custom"] = "human"
    gtf: Optional[Path] = None                 # if provided, overrides genome GTF discovery

    # Behavior
    gzip_fastq: bool = True
    fastp_enabled: bool = True
    simple_counts: bool = True                 # only gene_id,count
    download_method: str = "prefetch"          # "prefetch" or "iseq"

    # Metadata
    by: Literal["auto","srr","accession","sample"] = "auto"

    # iSeq settings
    iseq_enabled: bool = False
    iseq_sample_prefix: str = "Sample"
    iseq_sample_pattern: str = "auto"

    # iseq download options
    iseq_gzip: bool = True                    # Download FASTQ files in gzip format
    iseq_aspera: bool = False                 # Use Aspera for download
    iseq_database: str = "sra"                # Database: ena, sra
    iseq_protocol: str = "ftp"                # Protocol: ftp, https
    iseq_parallel: int = 4                    # Number of parallel downloads
    iseq_threads: int = 8                     # Threads for conversion/processing


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

        # 初始化iSeq处理器
        self.iseq_handler = ISeqHandler(
            sample_id_pattern=self.cfg.iseq_sample_pattern,
            paired_end=True,  # 默认双端
            validate_files=True
        )

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
        srrs: List[str] = []
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
            seen = set()
            srrs = [x for x in srrs if not (x in seen or seen.add(x))]
    
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

    def _parse_progress_line(self, line: str, progress_bar: DownloadProgressBar):
        """解析iseq输出行的进度信息"""
        try:
            line_lower = line.lower()

            # 匹配常见的进度格式
            # 格式1: "45.2MB/125.8MB (36%)" 或 "45.2MB / 125.8MB (36%)"
            progress_pattern = r'(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*\((\d+)%\)'
            match = re.search(progress_pattern, line, re.IGNORECASE)

            if match:
                current_size_str, current_unit, total_size_str, total_unit, percentage = match.groups()

                # 转换为字节
                current_bytes = self._parse_size_to_bytes(float(current_size_str), current_unit)
                total_bytes = self._parse_size_to_bytes(float(total_size_str), total_unit)

                # 更新进度条
                progress_bar.total_size = total_bytes
                progress_bar.update(current_bytes)
                return

            # 格式2: "Downloading: 45.2MB of 125.8MB"
            download_pattern = r'(\d+(?:\.\d+)?)\s*([KMGT]?B)\s+of\s+(\d+(?:\.\d+)?)\s*([KMGT]?B)'
            match = re.search(download_pattern, line, re.IGNORECASE)

            if match:
                current_size_str, current_unit, total_size_str, total_unit = match.groups()

                current_bytes = self._parse_size_to_bytes(float(current_size_str), current_unit)
                total_bytes = self._parse_size_to_bytes(float(total_size_str), total_unit)

                progress_bar.total_size = total_bytes
                progress_bar.update(current_bytes)
                return

            # 格式3: "45.2MB downloaded" 或 "Downloaded: 45.2MB"
            downloaded_pattern = r'(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*(?:downloaded|download)'
            match = re.search(downloaded_pattern, line, re.IGNORECASE)

            if match:
                size_str, unit = match.groups()
                current_bytes = self._parse_size_to_bytes(float(size_str), unit)
                progress_bar.update(current_bytes)
                return

            # 格式4: 百分比格式 "36%" 或 "Progress: 36%"
            percent_pattern = r'(\d+)%'
            match = re.search(percent_pattern, line)

            if match and progress_bar.total_size:
                percentage = int(match.group(1))
                current_bytes = int(progress_bar.total_size * percentage / 100)
                progress_bar.update(current_bytes)
                return

        except Exception as e:
            # 解析失败时不影响主程序
            logger.debug(f"Failed to parse progress line '{line}': {e}")

    def _monitor_file_sizes(self, out_dir: Path, srr_list: Sequence[str], progress_bar: DownloadProgressBar):
        """监控下载目录中的文件大小变化"""
        try:
            logger.debug("Starting file size monitoring...")
            start_time = time.time()
            monitored_files = {}
            last_total_size = 0

            while progress_bar.active:
                current_time = time.time()

                # 每2秒检查一次文件大小
                if current_time - start_time >= 2:
                    total_size = 0
                    found_files = 0

                    # 检查每个SRR对应的文件
                    for srr in srr_list:
                        # 查找可能的文件模式
                        patterns = [
                            f"{srr}*.fastq*",      # FASTQ文件（可能压缩）
                            f"{srr}*.fq*",         # 短扩展名
                            f"{srr}*.sra*",        # SRA格式
                            f"{srr}*.sralite"      # SRA lite格式
                        ]

                        srr_size = 0
                        for pattern in patterns:
                            files = list(out_dir.glob(pattern))
                            for file_path in files:
                                if file_path.is_file():
                                    try:
                                        file_size = file_path.stat().st_size
                                        srr_size += file_size

                                        # 记录文件大小变化
                                        file_key = str(file_path)
                                        if file_key not in monitored_files:
                                            monitored_files[file_key] = file_size
                                            logger.debug(f"Found new file: {file_path.name} ({self._format_size(file_size)})")
                                        elif monitored_files[file_key] != file_size:
                                            # 文件大小有变化，说明正在下载
                                            monitored_files[file_key] = file_size

                                    except Exception as e:
                                        logger.debug(f"Error getting file size for {file_path}: {e}")

                        if srr_size > 0:
                            found_files += 1
                            total_size += srr_size

                    # 如果有文件大小变化，更新进度条
                    if total_size > last_total_size:
                        progress_bar.update(total_size)
                        last_total_size = total_size
                        logger.debug(f"Total download progress: {self._format_size(total_size)} ({found_files}/{len(srr_list)} files)")

                    start_time = current_time

                time.sleep(0.5)  # 减少CPU使用率

        except Exception as e:
            logger.debug(f"File monitoring error: {e}")
        finally:
            logger.debug("File size monitoring stopped")

    def _parse_size_to_bytes(self, size: float, unit: str) -> int:
        """将大小字符串转换为字节数"""
        unit = unit.upper()
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4
        }
        return int(size * multipliers.get(unit, 1))

    # ---------- SRA: iseq download ----------
    def iseq_download(self, srr_list: Sequence[str], **iseq_options) -> Sequence[Path]:
        """
        Download SRA data using iseq tool.
        Returns: list of downloaded file paths.

        Args:
            srr_list: List of SRR accessions
            **iseq_options: Additional iseq options like gzip=True, aspera=True, etc.
        """
        # Check if iseq is available
        from . import tools_check as _tools_check

        # First check and auto-install axel if needed (Jupyter Lab specific)
        logger.info("Checking axel availability for iseq...")
        axel_available, axel_path = _tools_check.check_axel(auto_install=True)
        if not axel_available:
            logger.warning(f"axel is not available, which may cause issues with iseq: {axel_path}")
            # Continue anyway, as iseq might work without axel for some databases
        else:
            logger.info(f"axel is available: {axel_path}")

        # Check iseq availability
        iseq_available, iseq_path = _tools_check.check_iseq()
        if not iseq_available:
            raise RuntimeError(f"iseq not found: {iseq_path}. Please install with: conda install -c bioconda iseq -y")

        # Create output directory
        out_dir = Path(self.cfg.prefetch_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build iseq command
        # For multiple accessions, create a file with one accession per line
        if len(srr_list) == 1:
            # Single accession - pass directly
            cmd = [iseq_path, "-i", srr_list[0]]
        else:
            # Multiple accessions - create input file
            input_file = out_dir / "iseq_input.txt"
            input_file.write_text("\n".join(srr_list))
            cmd = [iseq_path, "-i", str(input_file)]

        # Add default options from configuration
        cmd.extend(["-o", str(out_dir)])
        cmd.extend(["-t", str(self.cfg.iseq_threads)])

        # Add configuration-based options
        if self.cfg.iseq_gzip:
            cmd.append("-g")

        # Aspera only supported for GSA/ENA databases, not SRA
        if self.cfg.iseq_aspera and self.cfg.iseq_database in ["ena", "gsa"]:
            cmd.append("-a")

        if self.cfg.iseq_parallel > 1:
            cmd.extend(["-p", str(self.cfg.iseq_parallel)])
        if self.cfg.iseq_database:
            cmd.extend(["-d", self.cfg.iseq_database])

        # Protocol parameter only for ENA database
        if self.cfg.iseq_protocol and self.cfg.iseq_database == "ena":
            cmd.extend(["-r", self.cfg.iseq_protocol])

        # Add user-specified options (override config)
        if iseq_options.get('gzip') is not None:
            if iseq_options['gzip']:
                cmd.append("-g")
            elif "-g" in cmd:
                cmd.remove("-g")

        # Handle Aspera - only for ENA/GSA databases
        if iseq_options.get('aspera') is not None:
            if iseq_options['aspera']:
                # 检查当前数据库设置
                current_db = iseq_options.get('database', self.cfg.iseq_database)
                if current_db in ["ena", "gsa"]:
                    cmd.append("-a")
                else:
                    logger.warning(f"Aspera不支持{current_db}数据库，跳过-a参数")
            elif "-a" in cmd:
                cmd.remove("-a")

        if iseq_options.get('parallel'):
            cmd.extend(["-p", str(iseq_options['parallel'])])

        if iseq_options.get('database'):
            new_db = iseq_options['database']
            cmd.extend(["-d", new_db])
            # 如果切换到非ENA数据库，需要移除协议参数
            if new_db != "ena" and "-r" in cmd:
                cmd.remove("-r")
                cmd.remove(cmd[cmd.index("-r") + 1])  # 移除协议值

        # 协议参数仅适用于ENA数据库
        if iseq_options.get('protocol'):
            current_db = iseq_options.get('database', self.cfg.iseq_database)
            if current_db == "ena":
                cmd.extend(["-r", iseq_options['protocol']])
            else:
                logger.warning(f"协议参数仅适用于ENA数据库，当前数据库为{current_db}，跳过-r参数")

        #if iseq_options.get('quiet', True):
        #    cmd.append("-Q")

        logger.info(f"Running iseq command: {' '.join(cmd)}")

        # Run iseq with real-time progress monitoring
        try:
            # 使用merged_env确保子进程有正确的环境变量
            from . import tools_check as _tools_check

            logger.info("Starting iseq download with real-time progress monitoring...")

            # 创建进度条
            progress_bar = DownloadProgressBar(desc=f"📥 Downloading {len(srr_list)} SRA file(s)")

            # 启动文件大小监控线程
            file_monitor_thread = None
            if len(srr_list) <= 5:  # 只对少量文件启用文件监控
                file_monitor_thread = threading.Thread(
                    target=self._monitor_file_sizes,
                    args=(out_dir, srr_list, progress_bar)
                )
                file_monitor_thread.daemon = True
                file_monitor_thread.start()

            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=_tools_check.merged_env()
            )

            # 实时监控输出
            def monitor_output(pipe, pipe_name):
                """监控输出流并解析进度信息"""
                try:
                    for line in iter(pipe.readline, ''):
                        line = line.strip()
                        if line:
                            logger.debug(f"iseq {pipe_name}: {line}")

                            # 尝试解析进度信息
                            # iseq 可能输出类似: "Downloading: 45.2MB/125.8MB (36%)" 或 "45.2MB downloaded"
                            self._parse_progress_line(line, progress_bar)

                except Exception as e:
                    logger.debug(f"Error monitoring {pipe_name}: {e}")
                finally:
                    pipe.close()

            # 启动监控线程
            stdout_thread = threading.Thread(target=monitor_output, args=(process.stdout, "stdout"))
            stderr_thread = threading.Thread(target=monitor_output, args=(process.stderr, "stderr"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # 等待进程完成
            return_code = process.wait()

            # 等待监控线程完成
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)

            # 停止文件监控
            if file_monitor_thread:
                file_monitor_thread.join(timeout=1)

            # 完成进度条
            progress_bar.finish()

            if return_code != 0:
                # 收集错误信息
                error_output = []
                if process.stderr:
                    try:
                        error_output.append(process.stderr.read())
                    except:
                        pass

                error_msg = " ".join(error_output) if error_output else f"Process exited with code {return_code}"
                logger.error("iseq download failed: %s", error_msg)
                raise RuntimeError(f"iseq download failed: {error_msg}")

        except Exception as e:
            if 'progress_bar' in locals():
                progress_bar.stop()
            logger.error("iseq download failed: %s", str(e))
            raise RuntimeError(f"iseq download failed: {str(e)}") from e

        # Find downloaded files with better detection
        downloaded_files = []
        missing_accessions = []

        for srr in srr_list:
            # Look for files with multiple naming patterns
            patterns = [
                f"{srr}*.fastq",      # Standard FASTQ
                f"{srr}*.fastq.gz",   # Gzipped FASTQ
                f"{srr}*.fq",         # Short extension
                f"{srr}*.fq.gz",      # Short extension gzipped
                f"{srr}*.sra",        # SRA format (fallback)
                f"{srr}*.sralite"     # SRA lite format
            ]

            srr_files = []
            for pattern in patterns:
                files = list(out_dir.glob(pattern))
                if files:
                    srr_files.extend(files)
                    logger.info(f"Found {len(files)} files for {srr} with pattern {pattern}")

            if srr_files:
                downloaded_files.extend(srr_files)
                logger.info(f"Found {len(srr_files)} total files for {srr}")
            else:
                missing_accessions.append(srr)
                logger.warning(f"No files found for {srr}")

        # Only report success if we actually found files
        if downloaded_files:
            logger.info(f"iseq download completed successfully - found {len(downloaded_files)} files")
        else:
            logger.error(f"iseq download failed - no files found for any accessions: {srr_list}")
            raise RuntimeError(f"iseq download failed - no files found for accessions: {missing_accessions}")

        return [Path(f) for f in downloaded_files]

    # ---------- SRA: prefetch ----------
    def prefetch(self, srr_list: Sequence[str]) -> Sequence[Path]:
        """
        Prefetch SRAs to .sra files.
        Returns: list of .sra paths.

        新增：支持镜像自动切换，通过配置中的 mirror_config 参数
        """
        # Delegate to your prefetch implementation. We assume it exposes a function like:
        # _sra_prefetch.prefetch_batch(srr_list, out_root=..., threads=..., mirror_config=...)
        if not hasattr(_sra_prefetch, "prefetch_batch"):
            raise RuntimeError("sra_prefetch.prefetch_batch(...) not found. Please expose it.")
        return _sra_prefetch.prefetch_batch(
            srr_list=srr_list,
            out_root=str(self.cfg.prefetch_root),
            threads=self.cfg.threads,
            prefetch_config=self.cfg.prefetch_config  # 使用基本预取配置
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
            gzip_output=self.cfg.gzip_fastq,
            tmp_root=str(self.cfg.fasterq_root / "tmp")  # Use proper tmp directory within fasterq_root
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
        index_root: Optional[Path] = None,
        accession_for_species: Optional[str] = None,   # 所有样本同一 GSE 时可统一传；否则保持 None
        max_workers: Optional[int] = None,             # 同时跑多少个样本；None=串行，日志更清晰
    ) -> list[Tuple[str, str, Optional[str]]]:
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
        # fastq_qc 返回的是5元组：(srr, fq1, fq2, json, html)，我们只需要前三个
        pairs: List[Tuple[str, str, str]] = [
            (srr, str(Path(fq1)), str(Path(fq2))) for srr, fq1, fq2, *_ in clean_fastqs
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
            simple=(self.cfg.simple_counts if simple is None else bool(simple)),
            gtf=None,  # 运行时 gtf 通过 command(...) 传入，优先级最高
            by=(by or self.cfg.by),
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

    # ---------- iSeq: Company FASTQ Data Processing ----------
    def process_company_fastq(
        self,
        fastq_input: str | Path | List[str | Path],
        *,
        sample_prefix: Optional[str] = None,
        output_subdir: str = "iseq"
    ) -> Dict[str, Any]:
        """
        处理公司提供的FASTQ数据

        Args:
            fastq_input: FASTQ文件路径或路径列表，可以是文件或目录
            sample_prefix: 样本ID前缀，默认使用配置中的值
            output_subdir: 输出子目录

        Returns:
            处理结果字典，包含样本信息和文件路径
        """
        if not self.cfg.iseq_enabled:
            raise ValueError("iSeq processing is not enabled. Set iseq_enabled=True in config.")

        # 设置输出目录
        iseq_root = self.cfg.work_root / output_subdir
        iseq_root.mkdir(parents=True, exist_ok=True)

        # 处理输入路径
        if isinstance(fastq_input, (str, Path)):
            input_paths = [Path(fastq_input)]
        else:
            input_paths = [Path(p) for p in fastq_input]

        # 处理公司数据
        result = self.iseq_handler.process_company_data(
            input_paths[0] if len(input_paths) == 1 else input_paths,
            output_dir=iseq_root,
            sample_prefix=sample_prefix or self.cfg.iseq_sample_prefix
        )

        # 转换为标准格式，与SRA流程兼容
        sample_pairs = []
        for original_id, r1_path, r2_path in result['sample_pairs']:
            # 使用标准化的样本ID
            standardized_id = result['sample_id_mapping'].get(original_id, {}).get('standardized_id', original_id)
            sample_pairs.append((standardized_id, r1_path, r2_path))

        return {
            'type': 'iseq',
            'sample_pairs': sample_pairs,
            'metadata': result['metadata_df'],
            'mapping': result['sample_id_mapping'],
            'output_dir': iseq_root
        }

    def run_from_fastq(
        self,
        fastq_pairs: Sequence[Tuple[str, Path, Optional[Path]]],
        *,
        with_align: bool = True,
        align_index: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        从FASTQ文件开始运行完整流程

        Args:
            fastq_pairs: [(sample_id, fq1_path, fq2_path), ...]
            with_align: 是否进行比对步骤
            align_index: 比对索引路径

        Returns:
            处理结果字典
        """
        # 直接进行QC步骤
        fastqs_qc = self.fastp(fastq_pairs)

        result: dict[str, Any] = {
            "type": "fastq_direct",
            "fastq_input": fastq_pairs,
            "fastq_qc": fastqs_qc,
        }

        if with_align:
            bams = self.star_align(fastqs_qc, index_root=align_index)
            counts = self.featurecounts(bams, gtf=self.cfg.gtf)
            result["bam"] = bams
            result["counts"] = counts
        else:
            # 如果不比对，可以直接进行定量（需要kallisto等工具）
            logger.warning("Direct quantification from FASTQ without alignment is not implemented yet.")

        return result

    # ---------- Unified Pipeline Entry ----------
    def run_pipeline(
        self,
        input_data: Union[str, Path, List[str], List[Path], Sequence[str]],
        *,
        input_type: Literal["sra", "fastq", "company"] = "auto",
        with_align: bool = True,
        align_index: Optional[Path] = None,
        sample_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        统一的管道入口，支持多种输入类型

        Args:
            input_data: 输入数据，可以是：
                - SRA: SRR编号列表或GEO accession
                - FASTQ: FASTQ文件路径或路径列表
                - Company: 公司数据目录或文件列表
            input_type: 输入类型，可选 "sra", "fastq", "company", "auto"
            with_align: 是否进行比对步骤
            align_index: 比对索引路径
            sample_prefix: 样本前缀（仅对公司数据有效）

        Returns:
            处理结果字典
        """
        # 自动检测输入类型
        if input_type == "auto":
            input_type = self._detect_input_type(input_data)

        logger.info(f"Detected input type: {input_type}")

        if input_type == "sra":
            # SRA数据 - 使用原有流程
            # 支持txt文件输入，直接读取其中的accession列表
            if isinstance(input_data, str) and Path(input_data).exists() and Path(input_data).suffix == '.txt':
                # 如果是txt文件，读取其中的accession列表
                srr_list = self._read_sra_accessions_from_file(Path(input_data))
            elif isinstance(input_data, str):
                srr_list = [input_data]  # 单个字符串包装成列表
            elif isinstance(input_data, (list, tuple)):
                srr_list = list(input_data)  # 转换列表格式
            else:
                srr_list = list(input_data)  # 其他序列类型

            return self.run(srr_list=srr_list, with_align=with_align, align_index=align_index)

        elif input_type == "company":
            # 公司数据 - 先处理数据，再运行流程
            iseq_result = self.process_company_fastq(
                input_data,
                sample_prefix=sample_prefix
            )
            return self.run_from_fastq(
                iseq_result['sample_pairs'],
                with_align=with_align,
                align_index=align_index
            )

        elif input_type == "fastq":
            # 直接FASTQ文件 - 需要用户指定样本ID
            fastq_pairs = self._parse_fastq_input(input_data)
            return self.run_from_fastq(
                fastq_pairs,
                with_align=with_align,
                align_index=align_index
            )

        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def _detect_input_type(self, input_data: Any) -> str:
        """自动检测输入类型"""
        if isinstance(input_data, str):
            # 单个字符串
            if input_data.startswith(('SRR', 'ERR', 'DRR')):
                return "sra"
            elif Path(input_data).exists():
                # 存在的路径
                path = Path(input_data)
                if path.is_file() and any(str(path).endswith(ext) for ext in ['.fq', '.fastq', '.fq.gz', '.fastq.gz']):
                    return "fastq"
                elif path.suffix in ['.txt', '.csv']:
                    # 检查是否是包含SRA accession的文本文件
                    if self._is_sra_accession_file(path):
                        return "sra"
                    else:
                        return "company"
                elif path.is_dir():
                    return "company"
                else:
                    return "fastq"
            else:
                # 可能是GEO accession
                return "sra"

        elif isinstance(input_data, Path):
            # Path对象
            if input_data.exists():
                if input_data.is_file() and any(str(input_data).endswith(ext) for ext in ['.fq', '.fastq', '.fq.gz', '.fastq.gz']):
                    return "fastq"
                else:
                    return "company"
            else:
                return "sra"

        elif isinstance(input_data, (list, tuple)):
            # 列表或元组
            if len(input_data) == 0:
                raise ValueError("Empty input data")

            first_item = input_data[0]
            if isinstance(first_item, str) and first_item.startswith(('SRR', 'ERR', 'DRR')):
                return "sra"
            elif any(str(item).endswith(('.fq', '.fastq', '.fq.gz', '.fastq.gz')) for item in input_data):
                return "fastq"
            else:
                return "company"

        else:
            raise ValueError(f"Cannot detect input type for: {type(input_data)}")

    def _is_sra_accession_file(self, file_path: Path) -> bool:
        """检查文件是否包含SRA accession列表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 检查前几行是否包含SRA/ERR/DRR accession
            sra_pattern = re.compile(r'^(SRR|ERR|DRR)\d+$')
            valid_lines = 0

            for line in lines[:20]:  # 检查前20行
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过空行和注释行
                    if sra_pattern.match(line):
                        valid_lines += 1
                    else:
                        return False  # 如果有一行不是SRA格式，返回False

            # 至少有一行有效的SRA accession
            return valid_lines > 0

        except Exception:
            return False

    def _read_sra_accessions_from_file(self, file_path: Path) -> List[str]:
        """从txt文件中读取SRA accession列表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            srr_list = []
            sra_pattern = re.compile(r'^(SRR|ERR|DRR)\d+$')

            for line in lines:
                line = line.strip()
                # 跳过空行和注释行
                if line and not line.startswith('#'):
                    if sra_pattern.match(line):
                        srr_list.append(line)
                    else:
                        logger.warning(f"跳过无效的行: {line}")

            if not srr_list:
                raise ValueError(f"文件 {file_path} 中没有找到有效的SRA accession")

            logger.info(f"从文件 {file_path} 中读取到 {len(srr_list)} 个SRA accession")
            return srr_list

        except Exception as e:
            raise RuntimeError(f"读取SRA accession文件失败 {file_path}: {e}")

    def _parse_fastq_input(self, input_data: Union[str, Path, List[str], List[Path]]) -> List[Tuple[str, Path, Optional[Path]]]:
        """解析FASTQ输入数据"""
        if isinstance(input_data, (str, Path)):
            # 单个文件
            file_path = Path(input_data)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # 自动提取样本ID
            sample_id = self.iseq_handler.extract_sample_id(file_path)

            # 检查是否为双端测序文件
            if self._is_paired_end_file(file_path):
                # 需要找到对应的R2文件
                r2_path = self._find_paired_file(file_path, "R2")
                return [(sample_id, file_path, r2_path)]
            else:
                return [(sample_id, file_path, None)]

        elif isinstance(input_data, (list, tuple)):
            # 多个文件
            fastq_files = [Path(f) for f in input_data]

            # 使用iSeq处理器进行配对
            return self.iseq_handler.group_paired_end(fastq_files)

        else:
            raise ValueError(f"Unsupported input format: {type(input_data)}")

    def _is_paired_end_file(self, file_path: Path) -> bool:
        """检查文件是否为双端测序的R1文件"""
        filename = file_path.name
        return bool(re.search(r'[._][Rr]1[._]', filename))

    def _find_paired_file(self, r1_path: Path, direction: str) -> Optional[Path]:
        """查找配对的R2文件"""
        r1_name = r1_path.name

        # 构建R2文件名
        if direction == "R2":
            r2_name = re.sub(r'([._])[Rr]1([._])', r'\1R2\2', r1_name)
            if r2_name == r1_name:  # 如果没有替换成功
                r2_name = r1_name.replace("_R1", "_R2").replace(".R1", ".R2")
        else:
            return None

        r2_path = r1_path.parent / r2_name
        return r2_path if r2_path.exists() else None

    def run(self, srr_list: Sequence[str], *, with_align: bool = False, align_index: Optional[Path] = None) -> Dict[str, Any]:
        """
        Convenience runner: download -> fasterq -> (fastp) -> [align] -> featureCounts

        Supports both prefetch and iseq download methods.

        Returns a dict of paths keyed by step.
        """
        # Choose download method based on configuration
        if self.cfg.download_method == "iseq":
            logger.info(f"Using iseq for download with {len(srr_list)} accessions")
            downloaded_files = self.iseq_download(srr_list)

            # Check if we got FASTQ files directly from iseq
            if downloaded_files:
                # Check if any files are FASTQ format
                fastq_files = [f for f in downloaded_files if str(f).endswith(('.fastq', '.fastq.gz', '.fq', '.fq.gz'))]
                sra_files = [f for f in downloaded_files if str(f).endswith(('.sra', '.sralite'))]

                if fastq_files:
                    # iseq provided FASTQ files, skip fasterq-dump step
                    logger.info(f"iseq provided {len(fastq_files)} FASTQ files, skipping fasterq-dump step")

                    # Create fasterq results from iseq output
                    fastqs = []
                    for srr in srr_list:
                        # Find files for this specific SRR
                        srr_files = [f for f in fastq_files if srr in str(f)]

                        if len(srr_files) >= 2:
                            # Try to find R1/R2 paired files
                            r1_candidates = [f for f in srr_files if any(marker in str(f) for marker in ['_1', '.1.', '_R1', '.R1.'])]
                            r2_candidates = [f for f in srr_files if any(marker in str(f) for marker in ['_2', '.2.', '_R2', '.R2.'])]

                            if r1_candidates and r2_candidates:
                                # Use the first matching R1/R2 files
                                fastqs.append((srr, Path(r1_candidates[0]), Path(r2_candidates[0])))
                            else:
                                # Fallback: use first two files
                                fastqs.append((srr, Path(srr_files[0]), Path(srr_files[1])))
                        elif srr_files:
                            # Single-end or single file
                            fastqs.append((srr, Path(srr_files[0]), None))
                        else:
                            logger.warning(f"No FASTQ files found for {srr} in iseq output")

                    result: dict[str, Any] = {
                        "prefetch": downloaded_files,  # Store iseq downloads as prefetch equivalent
                        "fastq": fastqs,
                    }
                elif sra_files:
                    # iseq downloaded SRA files, need fasterq-dump conversion
                    logger.info(f"iseq provided {len(sra_files)} SRA files, will run fasterq-dump conversion")
                    sras = downloaded_files
                    fastqs = self.fasterq(srr_list)
                    result: dict[str, Any] = {
                        "prefetch": sras,
                        "fastq": fastqs,
                    }
                else:
                    # Unknown file types, try fasterq-dump anyway
                    logger.warning(f"iseq provided files of unknown type, will attempt fasterq-dump conversion")
                    sras = downloaded_files
                    fastqs = self.fasterq(srr_list)
                    result: dict[str, Any] = {
                        "prefetch": sras,
                        "fastq": fastqs,
                    }
            else:
                # No files found, this should have been caught by iseq_download
                logger.error("iseq download returned no files")
                raise RuntimeError("iseq download failed - no files were downloaded")
        else:
            # Default prefetch method
            logger.info(f"Using prefetch for download with {len(srr_list)} accessions")
            sras = self.prefetch(srr_list)
            fastqs = self.fasterq(srr_list)
            result: dict[str, Any] = {
                "prefetch": sras,
                "fastq": fastqs,
            }

        # Continue with common processing
        fastqs_qc = self.fastp(fastqs)
        result["fastq_qc"] = fastqs_qc

        if with_align:
            # 使用现有的STAR比对功能
            bam_triples = self.star_align(fastqs_qc)
            # 提取BAM路径（bam_triples格式: [(srr, bam_path, index_dir), ...]）
            bams = [(srr, Path(bam_path)) for srr, bam_path, _ in bam_triples]
        else:
            # 跳过比对步骤，直接返回空结果
            logger.info("跳过比对步骤 (with_align=False)")
            bam_triples = []
            bams = []

        counts = self.featurecounts(bam_triples, gtf=self.cfg.gtf)
        result["bam"] = bams
        result["counts"] = counts
        return result
