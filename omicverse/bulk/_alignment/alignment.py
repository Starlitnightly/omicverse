# Author: Zhi Luo
# omicverse/bulk/alignment.py

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

# Configure logging.
logger = logging.getLogger(__name__)


class DownloadProgressBar:
    """Progress bar helper for real-time download updates."""

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
        """Update the progress bar."""
        with self.lock:
            self.current_size = current_size
            current_time = time.time()

            # Update the display every 0.5 seconds to avoid spamming stdout.
            if current_time - self.last_update >= 0.5:
                self._display_progress(current_time)
                self.last_update = current_time
                self.last_size = current_size

    def _display_progress(self, current_time):
        """Render the progress bar."""
        if not self.active:
            return

        elapsed = current_time - self.start_time
        if elapsed == 0:
            return

        # Compute the speed in bytes per second.
        speed = (self.current_size - self.last_size) / (current_time - self.last_update) if current_time > self.last_update else 0

        # Format the file size.
        current_str = self._format_size(self.current_size)
        speed_str = self._format_size(speed) + "/s"

        if self.total_size and self.total_size > 0:
            # Show a percentage progress bar when total size is known.
            percentage = (self.current_size / self.total_size) * 100
            bar_length = 40
            filled_length = int(bar_length * self.current_size // self.total_size)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            total_str = self._format_size(self.total_size)
            eta = (self.total_size - self.current_size) / speed if speed > 0 else 0
            eta_str = self._format_time(eta)

            print(f'\r{self.desc}: {percentage:.1f}%|{bar}| {current_str}/{total_str} [{speed_str}, ETA: {eta_str}]', end='', flush=True)
        else:
            # Without total size, only display the current size and speed.
            print(f'\r{self.desc}: {current_str} [{speed_str}]', end='', flush=True)

    def _format_size(self, size):
        """Format file sizes into human-readable strings."""
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
        """Format seconds into a human-readable string."""
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
        """Finalize the progress bar."""
        with self.lock:
            self.active = False
            # Show the final state.
            total_time = time.time() - self.start_time
            total_str = self._format_size(self.current_size)
            avg_speed = self.current_size / total_time if total_time > 0 else 0
            avg_speed_str = self._format_size(avg_speed) + "/s"

            if self.total_size and self.current_size >= self.total_size:
                print(f'\r{self.desc}: 100%|{"█" * 40}| {total_str}/{total_str} [{avg_speed_str}, Total: {self._format_time(total_time)}]')
            else:
                print(f'\r{self.desc}: {total_str} [{avg_speed_str}, Total: {self._format_time(total_time)}]')
            print()  # New line.

    def stop(self):
        """Stop the progress bar."""
        self.active = False


@dataclass
class AlignmentConfig:
    # IO roots
    work_root: Path = Path("work")
    meta_root: Path = field(init=False)
    prefetch_root: Path = field(init=False)
    fasterq_root: Path = field(init=False)
    qc_root: Path = field(init=False)
    align_root: Path = field(init=False)     # Placeholder if you later add HISAT2/STAR.
    counts_root: Path = field(init=False)

    star_index_root: Path = field(init=False)        # Index root (aligned with star_tools).
    star_align_root: Path  = field(init=False)   # STAR output root.

    # STAR alignment configuration
    star_memory_limit: str = "100G"            # BAM sorting memory limit (e.g., "100G", "85899345920" for bytes)
    star_extra_args: List[str] = field(default_factory=list)  # Additional STAR arguments

    # Library layout configuration for GEO prefetch data
    library_layout: Literal["auto", "single", "paired"] = "auto"  # "auto" = detect, "single", "paired"

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
        self.align_root = self.work_root / "align"  # Reserved for alternative aligners.
        self.counts_root = self.work_root / "counts"
        self.star_align_root = self.work_root / "star"
        self.star_index_root = self.work_root / "index"

        # Keep STAR BAM sort memory aligned with the generic memory setting
        # unless the user has explicitly overridden star_memory_limit.
        # Heuristic: if star_memory_limit is still at its class default ("100G")
        # and memory was customized from its default ("8G"), propagate it.
        try:
            if self.star_memory_limit == "100G" and self.memory != "8G":
                self.star_memory_limit = self.memory
        except AttributeError:
            # In case older configs lack one of these attributes, fail silently.
            pass

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

        # Initialize the iSeq handler.
        self.iseq_handler = ISeqHandler(
            sample_id_pattern=self.cfg.iseq_sample_pattern,
            paired_end=True,  # Default to paired-end.
            validate_files=True
        )

     # ---------- Fetch Metadata ----------
    def fetch_metadata(
        self,
        accession: str,
        meta_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        organism_filter: Optional[str] = None,   # e.g. "Homo sapiens"
        layout_filter: Optional[str] = None,     # "PAIRED" / "SINGLE"
    ):
        """
        Given a GEO accession (GSE/GSM), fetch SOFT, persist the meta JSON,
        then run EDirect to generate the RunInfo CSV and return SRR IDs plus paths.
        """
        _tools_check.check()
        # 1) Prepare directories.
        meta_root = Path(meta_dir) if meta_dir else (Path(self.cfg.work_root) / "meta")
        #sra_meta_root = Path(out_dir) if out_dir else (Path(self.cfg.work_root) / "meta")
        meta_root.mkdir(parents=True, exist_ok=True)
        #sra_meta_root.mkdir(parents=True, exist_ok=True)
    
        # 2) Generate/update the JSON metadata (note the use of out_dir).
        _geo.geo_accession_to_meta_json(accession, out_dir=str(meta_root))

        # 3) Generate the RunInfo CSV (uses accession + meta_dir/out_dir).
        info = _ed.gse_meta_to_runinfo_csv(
            accession=accession,
            meta_dir=str(meta_root),
            out_dir=str(meta_root),
            organism_filter=organism_filter,
            layout_filter=layout_filter,
        )
        runinfo_csv = Path(info["csv"]) if info.get("csv") else None
    
        # 4) Extract the SRR list.
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
            # Deduplicate while preserving order.
            seen = set()
            srrs = [x for x in srrs if not (x in seen or seen.add(x))]
    
        # 5) Optional: structured metadata (full SOFT parse).
        try:
            meta_struct = _geo.parse_geo_soft_to_struct(_geo.fetch_geo_text(accession))
        except Exception:
            meta_struct = None
    
        return {
            "meta_json": meta_root / f"{accession}_meta.json",
            "runinfo_csv": runinfo_csv,
            "srr_list": srrs,
            "edirect_info": info,   # Includes term_used/rows and related fields.
            "meta_struct": meta_struct,
        }

    def _parse_progress_line(self, line: str, progress_bar: DownloadProgressBar):
        """Parse progress information emitted by iseq."""
        try:
            line_lower = line.lower()

            # Match common progress formats.
            # Pattern 1: "45.2MB/125.8MB (36%)" or "45.2MB / 125.8MB (36%)"
            progress_pattern = r'(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*\((\d+)%\)'
            match = re.search(progress_pattern, line, re.IGNORECASE)

            if match:
                current_size_str, current_unit, total_size_str, total_unit, percentage = match.groups()

                # Convert into bytes.
                current_bytes = self._parse_size_to_bytes(float(current_size_str), current_unit)
                total_bytes = self._parse_size_to_bytes(float(total_size_str), total_unit)

                # Update the progress bar accordingly.
                progress_bar.total_size = total_bytes
                progress_bar.update(current_bytes)
                return

            # Pattern 2: "Downloading: 45.2MB of 125.8MB"
            download_pattern = r'(\d+(?:\.\d+)?)\s*([KMGT]?B)\s+of\s+(\d+(?:\.\d+)?)\s*([KMGT]?B)'
            match = re.search(download_pattern, line, re.IGNORECASE)

            if match:
                current_size_str, current_unit, total_size_str, total_unit = match.groups()

                current_bytes = self._parse_size_to_bytes(float(current_size_str), current_unit)
                total_bytes = self._parse_size_to_bytes(float(total_size_str), total_unit)

                progress_bar.total_size = total_bytes
                progress_bar.update(current_bytes)
                return

            # Pattern 3: "45.2MB downloaded" or "Downloaded: 45.2MB"
            downloaded_pattern = r'(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*(?:downloaded|download)'
            match = re.search(downloaded_pattern, line, re.IGNORECASE)

            if match:
                size_str, unit = match.groups()
                current_bytes = self._parse_size_to_bytes(float(size_str), unit)
                progress_bar.update(current_bytes)
                return

            # Pattern 4: percentage-only "36%" or "Progress: 36%".
            percent_pattern = r'(\d+)%'
            match = re.search(percent_pattern, line)

            if match and progress_bar.total_size:
                percentage = int(match.group(1))
                current_bytes = int(progress_bar.total_size * percentage / 100)
                progress_bar.update(current_bytes)
                return

        except Exception as e:
            # Silent failure; do not impact the main execution.
            logger.debug(f"Failed to parse progress line '{line}': {e}")

    def _monitor_file_sizes(self, out_dir: Path, srr_list: Sequence[str], progress_bar: DownloadProgressBar):
        """Monitor file size changes inside the download directory."""
        try:
            logger.debug("Starting file size monitoring...")
            start_time = time.time()
            monitored_files = {}
            last_total_size = 0

            while progress_bar.active:
                current_time = time.time()

                # Check file sizes every two seconds.
                if current_time - start_time >= 2:
                    total_size = 0
                    found_files = 0

                    # Inspect files belonging to each SRR.
                    for srr in srr_list:
                        # Examine possible file patterns.
                        patterns = [
                            f"{srr}*.fastq*",      # FASTQ files (potentially compressed).
                            f"{srr}*.fq*",         # Short extensions.
                            f"{srr}*.sra*",        # SRA archives.
                            f"{srr}*.sralite"      # SRA lite archives.
                        ]

                        srr_size = 0
                        for pattern in patterns:
                            files = list(out_dir.glob(pattern))
                            for file_path in files:
                                if file_path.is_file():
                                    try:
                                        file_size = file_path.stat().st_size
                                        srr_size += file_size

                                        # Track file size changes.
                                        file_key = str(file_path)
                                        if file_key not in monitored_files:
                                            monitored_files[file_key] = file_size
                                            logger.debug(f"Found new file: {file_path.name} ({self._format_size(file_size)})")
                                        elif monitored_files[file_key] != file_size:
                                            # Size changed, indicating an active download.
                                            monitored_files[file_key] = file_size

                                    except Exception as e:
                                        logger.debug(f"Error getting file size for {file_path}: {e}")

                        if srr_size > 0:
                            found_files += 1
                            total_size += srr_size

                    # Update the progress bar when growth is detected.
                    if total_size > last_total_size:
                        progress_bar.update(total_size)
                        last_total_size = total_size
                        logger.debug(f"Total download progress: {self._format_size(total_size)} ({found_files}/{len(srr_list)} files)")

                    start_time = current_time

                time.sleep(0.5)  # Reduce CPU usage.

        except Exception as e:
            logger.debug(f"File monitoring error: {e}")
        finally:
            logger.debug("File size monitoring stopped")

    def _parse_size_to_bytes(self, size: float, unit: str) -> int:
        """Convert a size string into bytes."""
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
                # Validate the current database selection.
                current_db = iseq_options.get('database', self.cfg.iseq_database)
                if current_db in ["ena", "gsa"]:
                    cmd.append("-a")
                else:
                    logger.warning(f"Aspera is not supported for database {current_db}; skipping -a.")
            elif "-a" in cmd:
                cmd.remove("-a")

        if iseq_options.get('parallel'):
            cmd.extend(["-p", str(iseq_options['parallel'])])

        if iseq_options.get('database'):
            new_db = iseq_options['database']
            cmd.extend(["-d", new_db])
            # Remove protocol arguments when switching away from ENA.
            if new_db != "ena" and "-r" in cmd:
                idx = cmd.index("-r")
                cmd.pop(idx)  # remove flag
                if idx < len(cmd):
                    cmd.pop(idx)  # remove associated value

        # The protocol parameter only applies to the ENA database.
        if iseq_options.get('protocol'):
            current_db = iseq_options.get('database', self.cfg.iseq_database)
            if current_db == "ena":
                cmd.extend(["-r", iseq_options['protocol']])
            else:
                logger.warning(f"Protocol is only valid for ENA; database is {current_db}, skipping -r.")

        #if iseq_options.get('quiet', True):
        #    cmd.append("-Q")

        logger.info(f"Running iseq command: {' '.join(cmd)}")

        # Run iseq with real-time progress monitoring
        try:
            # Ensure the subprocess inherits the merged environment.
            from . import tools_check as _tools_check

            logger.info("Starting iseq download with real-time progress monitoring...")

            # Create the progress bar.
            progress_bar = DownloadProgressBar(desc=f"📥 Downloading {len(srr_list)} SRA file(s)")

            # Kick off file-size monitoring.
            file_monitor_thread = None
            if len(srr_list) <= 5:  # Only enable file-size monitoring when the list is small.
                file_monitor_thread = threading.Thread(
                    target=self._monitor_file_sizes,
                    args=(out_dir, srr_list, progress_bar)
                )
                file_monitor_thread.daemon = True
                file_monitor_thread.start()

            # Launch the process.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=_tools_check.merged_env()
            )

            # Monitor stdout/stderr in real time.
            def monitor_output(pipe, pipe_name):
                """Monitor process output and parse progress information."""
                try:
                    for line in iter(pipe.readline, ''):
                        line = line.strip()
                        if line:
                            logger.debug(f"iseq {pipe_name}: {line}")

                            # Attempt to decode progress information.
                            # iseq often prints lines like "Downloading: 45.2MB/125.8MB (36%)" or "45.2MB downloaded".
                            self._parse_progress_line(line, progress_bar)

                except Exception as e:
                    logger.debug(f"Error monitoring {pipe_name}: {e}")
                finally:
                    pipe.close()

            # Spin up monitoring threads.
            stdout_thread = threading.Thread(target=monitor_output, args=(process.stdout, "stdout"))
            stderr_thread = threading.Thread(target=monitor_output, args=(process.stderr, "stderr"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Wait for the process to finish.
            return_code = process.wait()

            # Wait for watcher threads to flush.
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)

            # Stop file monitoring.
            if file_monitor_thread:
                file_monitor_thread.join(timeout=1)

            # Complete the progress bar.
            progress_bar.finish()

            if return_code != 0:
                # Collect error output.
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
        Returns a list of .sra paths.
        Supports automatic mirror selection via the mirror_config setting.
        """
        # Delegate to your prefetch implementation. We assume it exposes a function like:
        # _sra_prefetch.prefetch_batch(srr_list, out_root=..., threads=..., mirror_config=...)
        if not hasattr(_sra_prefetch, "prefetch_batch"):
            raise RuntimeError("sra_prefetch.prefetch_batch(...) not found. Please expose it.")
        return _sra_prefetch.prefetch_batch(
            srr_list=srr_list,
            out_root=str(self.cfg.prefetch_root),
            threads=self.cfg.threads,
            prefetch_config=self.cfg.prefetch_config  # Use the base prefetch configuration.
        )

    # ---------- SRA: fasterq-dump ----------
    def fasterq(self, srr_list: Sequence[str]) -> Sequence[Tuple[str, Path, Optional[Path]]]:
        """
        Convert SRA to FASTQ(.gz) with library layout awareness.
        Supports both paired-end and single-end library layouts.
        Returns: list of tuples (srr, fq1_path, fq2_path) where fq2_path is None for single-end.
        """
        if not hasattr(_sra_fasterq, "fasterq_batch"):
            raise RuntimeError("sra_fasterq.fasterq_batch(...) not found. Please expose it.")

        # Adjust fasterq behavior based on library layout configuration
        if self.cfg.library_layout != "auto":
            # User specified library layout, pass hint to fasterq processing
            logger.info(f"Using user-specified library layout: {self.cfg.library_layout}")

        # Pass the library layout configuration to fasterq processing
        return _sra_fasterq.fasterq_batch(
            srr_list=srr_list,
            out_root=str(self.cfg.fasterq_root),
            threads=self.cfg.threads,
            mem=self.cfg.memory,
            gzip_output=self.cfg.gzip_fastq,
            tmp_root=str(self.cfg.fasterq_root / "tmp"),
            library_layout=self.cfg.library_layout  # Pass the library layout configuration
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
        clean_fastqs: Sequence[Tuple[str, Path, Path | None]],  # [(srr, fq1, fq2 | None)] - fq2 is None for single-end
        *,
        gencode_release: str = "v44",
        sjdb_overhang: Optional[int] = 149,
        index_root: Optional[Path] = None,
        accession_for_species: Optional[str] = None,   # Provide a shared accession when every sample belongs to the same GSE.
        max_workers: Optional[int] = None,             # Number of concurrent samples; None keeps logging clearer.
    ) -> list[Tuple[str, str, Optional[str]]]:
        """
        Run STAR in batch mode via star_step.make_star_step and return
          [(srr, bam_path, index_dir|None), ...].
        - Idempotent: skip when <SRR>/Aligned.sortedByCoord.out.bam already exists (>1 MB).
        - index_dir is returned when star_tools exposes it; otherwise None to keep downstream GTF inference consistent.
        """
        if not hasattr(_star_step, "make_star_step"):
            raise RuntimeError("star_step.make_star_step(...) not found")

        # Construct a batch-ready step using the existing factory interface.
        step = _star_step.make_star_step(
            index_root=str(self.cfg.star_index_root),
            out_root=str(self.cfg.star_align_root),
            threads=int(self.cfg.threads),
            gencode_release=gencode_release,
            sjdb_overhang=sjdb_overhang,
            accession_for_species=accession_for_species,
            max_workers=max_workers,   # None = serial; callers can opt-in to 2/4/etc concurrent runs.
            memory_limit=str(self.cfg.star_memory_limit),  # BAM sorting memory limit from configuration
        )

        # Normalize inputs as [(srr, str(fq1), str(fq2)), ...] handling single-end (fq2=None) cases.
        # fastq_qc returns 5-tuples (srr, fq1, fq2, json, html); fq2 is None for single-end data.
        pairs: List[Tuple[str, str, str]] = []
        for entry in clean_fastqs:
            srr, fq1, fq2_or_none, *_ = entry  # fq2_or_none is None for single-end
            # Handle single-end case where fq2 is None
            fq2 = str(Path(fq2_or_none)) if fq2_or_none is not None else ""  # Empty string for STAR single-end
            pairs.append((srr, str(Path(fq1)), fq2))

        # Execute the batch command to obtain [(srr, bam, index_dir|None), ...].
        products = step["command"](pairs, logger=None)
        # Remains fully compatible with the pipeline's triplet normalization.
        return products

    # ---------- Counting via featureCounts ----------
    def featurecounts(
        self,
        bam_triples: Sequence[Tuple[str, str | Path, Optional[str]] | Tuple[str, str | Path, Optional[str], Optional[bool]]],   # [(srr, bam, index_dir|None[, is_paired])]
        *,
        gtf: Optional[str | Path] = None,         # Explicit GTF path takes highest priority.
        simple: Optional[bool] = None,            # None→cfg.featurecounts_simple
        by: Optional[str] = None,                 # None→cfg.featurecounts_by
        threads: Optional[int] = None,            # None→cfg.threads
        max_workers: Optional[int] = None,        # Reserved to pass through when count_tools adds parallel support.
    ) -> Dict[str, object]:
        """
        Batch execution of featureCounts.
        Returns {"tables": [(srr, table_path), ...], "matrix": <path|None>, "failed": []}.
        Idempotent: skip when <counts_root>/<SRR>/<SRR>.counts.txt exists and is non-empty.
        """
        if not hasattr(_count_step, "make_featurecounts_step"):
            raise RuntimeError("count_step.make_featurecounts_step(...) not found")

        out_root = Path(self.cfg.counts_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # ---------- Built-in GTF inference ----------
        def _infer_gtf_from_bams(triples: Sequence[Tuple[str, str | Path, Optional[str]] | Tuple[str, str | Path, Optional[str], Optional[bool]]]) -> Optional[str]:
            # 1) Prefer GTF discovery from each sample's index_dir when available.
            for rec in triples:
                _srr, _bam, idx_dir = rec[:3]
                if not idx_dir:
                    continue
                idx = Path(idx_dir)
                # (a) Search *.gtf in the directory and its parent.
                for base in {idx, idx.parent}:
                    for p in base.glob("*.gtf"):
                        return str(p.resolve())
                # (b) Look for *.gtf under a sibling _cache directory.
                for base in {idx.parent, idx.parent.parent}:
                    cache = base / "_cache"
                    if cache.exists():
                        hits = list(cache.rglob("*.gtf"))
                        if hits:
                            return str(hits[0].resolve())
                # (c) Finally, search one level higher.
                for p in idx.parent.parent.glob("*.gtf"):
                    return str(p.resolve())

            # 2) Otherwise, fall back to the configured star_index_root.
            idx_root = Path(getattr(self.cfg, "star_index_root", "index"))
            if idx_root.exists():
                hits = list(idx_root.rglob("*.gtf"))
                if hits:
                    return str(hits[0].resolve())

            # 3) Lastly, honor the FC_GTF_HINT environment variable.
            env_hint = os.environ.get("FC_GTF_HINT")
            if env_hint and Path(env_hint).exists():
                return str(Path(env_hint).resolve())

            return None

        # Infer the GTF when it is not supplied explicitly.
        if gtf is None:
            inferred = _infer_gtf_from_bams(bam_triples)
            if inferred:
                print(f"[INFO] featureCounts: inferred GTF -> {inferred}")
                gtf = inferred
            else:
                raise RuntimeError(
                    "[featureCounts] Unable to locate a GTF automatically; provide gtf= or set FC_GTF_HINT."
                )

        # ---------- Build the step factory and enforce idempotency ----------
        step = _count_step.make_featurecounts_step(
            out_root=str(out_root),
            simple=(self.cfg.simple_counts if simple is None else bool(simple)),
            gtf=None,  # gtf is supplied at command time and takes precedence.
            by=(by or self.cfg.by),
            threads=int(threads or self.cfg.threads),
            gtf_path=str(gtf),  # Acts as a fallback; the command-level GTF wins internally.
        )

        def _table_path_for(srr: str) -> Path:
            # Change the extension here if your count_tools output is .csv and adjust the outputs template accordingly.
            return out_root / srr / f"{srr}.counts.txt"

        # Idempotent shortcut: skip when every output already exists.
        outs_by_srr: List[Tuple[str, Path]] = [
            (str(rec[0]), _table_path_for(str(rec[0]))) for rec in bam_triples
        ]
        if all(step["validation"]([str(p)]) for _, p in outs_by_srr):
            print("[SKIP] featureCounts for all")
            tables = [(srr, str(p)) for srr, p in outs_by_srr]
            return {"tables": tables, "matrix": None, "failed": []}

        # Assemble (srr, bam[, is_paired]) pairs and execute.
        layout_hint = None if self.cfg.library_layout == "auto" else (self.cfg.library_layout == "paired")
        bam_pairs = []
        for rec in bam_triples:
            if len(rec) == 4:
                srr, bam, _idx, is_paired = rec  # type: ignore[misc]
            elif len(rec) == 3:
                srr, bam, _idx = rec  # type: ignore[misc]
                is_paired = None
            else:
                raise ValueError(f"featurecounts expects (srr, bam, idx_dir[, is_paired]) tuples; got {rec}")

            # User hint overrides auto detection.
            if layout_hint is not None:
                is_paired = layout_hint

            bam_pairs.append((str(srr), str(bam), is_paired))

        ret = step["command"](
            bam_pairs,
            logger=None,
            gtf=str(gtf),  # Explicit runtime GTF has top priority.
        )

        tables = [(rec[0], str(_table_path_for(str(rec[0])))) for rec in bam_pairs]
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
        Process vendor-provided FASTQ data and align its structure with the SRA workflow.

        Args:
            fastq_input: FASTQ path or collection of paths (files or directories).
            sample_prefix: Optional sample identifier prefix; falls back to the configured value.
            output_subdir: Subdirectory under work_root where outputs are written.

        Returns:
            A result dictionary containing sample metadata and resolved file paths.
        """
        if not self.cfg.iseq_enabled:
            raise ValueError("iSeq processing is not enabled. Set iseq_enabled=True in config.")

        # Configure the output directory.
        iseq_root = self.cfg.work_root / output_subdir
        iseq_root.mkdir(parents=True, exist_ok=True)

        # Normalize the input paths.
        if isinstance(fastq_input, (str, Path)):
            input_paths = [Path(fastq_input)]
        else:
            input_paths = [Path(p) for p in fastq_input]

        # Process the vendor data.
        result = self.iseq_handler.process_company_data(
            input_paths[0] if len(input_paths) == 1 else input_paths,
            output_dir=iseq_root,
            sample_prefix=sample_prefix or self.cfg.iseq_sample_prefix
        )

        # Convert into the standard structure expected by the SRA workflow.
        sample_pairs = []
        for original_id, r1_path, r2_path in result['sample_pairs']:
            # Use the standardized sample identifier.
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
        Execute the full pipeline starting from FASTQ inputs.

        Args:
            fastq_pairs: Iterable of (sample_id, fq1_path, fq2_path or None).
            with_align: Whether to include the alignment stage.
            align_index: Optional alignment index overriding the configuration.

        Returns:
            A dictionary describing the processing outcomes.
        """
        # Run QC immediately.
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
            # Direct quantification without alignment requires external tools (e.g., kallisto) and is not implemented.
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
        Unified pipeline entry point supporting SRA, FASTQ, and vendor data.

        Args:
            input_data: Source data, e.g. SRR list/GEO accession, FASTQ paths, or vendor directories/files.
            input_type: Explicit input type ("sra", "fastq", "company") or "auto" to detect automatically.
            with_align: Whether to perform alignment.
            align_index: Optional alignment index path.
            sample_prefix: Optional prefix for vendor sample IDs.

        Returns:
            A dictionary describing the processing outcome.
        """
        # Auto-detect the input type when requested.
        if input_type == "auto":
            input_type = self._detect_input_type(input_data)

        logger.info(f"Detected input type: {input_type}")

        if input_type == "sra":
            # SRA data — reuse the existing workflow.
            # Support plain-text files containing accession lists.
            if isinstance(input_data, str) and Path(input_data).exists() and Path(input_data).suffix == '.txt':
                # When a txt file is supplied, read accessions from it.
                srr_list = self._read_sra_accessions_from_file(Path(input_data))
            elif isinstance(input_data, str):
                srr_list = [input_data]  # Wrap a single accession in a list.
            elif isinstance(input_data, (list, tuple)):
                srr_list = list(input_data)  # Normalize to list.
            else:
                srr_list = list(input_data)  # Convert other iterable types.

            return self.run(srr_list=srr_list, with_align=with_align, align_index=align_index)

        elif input_type == "company":
            # Vendor data: preprocess, then run the pipeline.
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
            # Direct FASTQ input requires explicit sample IDs.
            fastq_pairs = self._parse_fastq_input(input_data)
            return self.run_from_fastq(
                fastq_pairs,
                with_align=with_align,
                align_index=align_index
            )

        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def _detect_input_type(self, input_data: Any) -> str:
        """Automatically infer the input type."""
        if isinstance(input_data, str):
            # Single string input.
            if input_data.startswith(('SRR', 'ERR', 'DRR')):
                return "sra"
            elif Path(input_data).exists():
                # Path exists locally.
                path = Path(input_data)
                if path.is_file() and any(str(path).endswith(ext) for ext in ['.fq', '.fastq', '.fq.gz', '.fastq.gz']):
                    return "fastq"
                elif path.suffix in ['.txt', '.csv']:
                    # Determine whether the text file contains SRA accessions.
                    if self._is_sra_accession_file(path):
                        return "sra"
                    else:
                        return "company"
                elif path.is_dir():
                    return "company"
                else:
                    return "fastq"
            else:
                # Assume a GEO accession.
                return "sra"

        elif isinstance(input_data, Path):
            # Path object input.
            if input_data.exists():
                if input_data.is_file() and any(str(input_data).endswith(ext) for ext in ['.fq', '.fastq', '.fq.gz', '.fastq.gz']):
                    return "fastq"
                else:
                    return "company"
            else:
                return "sra"

        elif isinstance(input_data, (list, tuple)):
            # List or tuple input.
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
        """Check whether the file contains SRA accessions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Inspect initial lines for SRR/ERR/DRR accessions.
            sra_pattern = re.compile(r'^(SRR|ERR|DRR)\d+$')
            valid_lines = 0

            for line in lines[:20]:  # Examine the first 20 lines.
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comment lines.
                    if sra_pattern.match(line):
                        valid_lines += 1
                    else:
                        return False  # Reject when a line is not in SRA format.

            # Ensure at least one valid accession.
            return valid_lines > 0

        except Exception:
            return False

    def _read_sra_accessions_from_file(self, file_path: Path) -> List[str]:
        """Load SRA accessions from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            srr_list = []
            sra_pattern = re.compile(r'^(SRR|ERR|DRR)\d+$')

            for line in lines:
                line = line.strip()
                # Skip empty lines and comments.
                if line and not line.startswith('#'):
                    if sra_pattern.match(line):
                        srr_list.append(line)
                    else:
                        logger.warning(f"Skipping invalid accession line: {line}")

            if not srr_list:
                raise ValueError(f"No valid SRA accessions found in {file_path}")

            logger.info(f"Parsed {len(srr_list)} SRA accessions from {file_path}")
            return srr_list

        except Exception as e:
            raise RuntimeError(f"Failed to read SRA accession file {file_path}: {e}")

    def _parse_fastq_input(self, input_data: Union[str, Path, List[str], List[Path]]) -> List[Tuple[str, Path, Optional[Path]]]:
        """Parse FASTQ inputs into (sample_id, R1, R2?) tuples."""
        if isinstance(input_data, (str, Path)):
            # Single file input.
            file_path = Path(input_data)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Derive the sample ID automatically.
            sample_id = self.iseq_handler.extract_sample_id(file_path)

            # Determine whether this is an R1 paired-end file.
            if self._is_paired_end_file(file_path):
                # Locate the matching R2 file.
                r2_path = self._find_paired_file(file_path, "R2")
                return [(sample_id, file_path, r2_path)]
            else:
                return [(sample_id, file_path, None)]

        elif isinstance(input_data, (list, tuple)):
            # Multiple files input.
            fastq_files = [Path(f) for f in input_data]

            # Use the iSeq handler to pair R1/R2 files.
            return self.iseq_handler.group_paired_end(fastq_files)

        else:
            raise ValueError(f"Unsupported input format: {type(input_data)}")

    def _is_paired_end_file(self, file_path: Path) -> bool:
        """Return True when the filename indicates a paired-end R1 read."""
        filename = file_path.name
        return bool(re.search(r'[._][Rr]1[._]', filename))

    def _find_paired_file(self, r1_path: Path, direction: str) -> Optional[Path]:
        """Locate the paired R2 file for the given R1 path."""
        r1_name = r1_path.name

        # Build the R2 filename.
        if direction == "R2":
            r2_name = re.sub(r'([._])[Rr]1([._])', r'\1R2\2', r1_name)
            if r2_name == r1_name:  # Fallback when no replacement occurred.
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
            # Reuse the existing STAR alignment helper.
            bam_triples = self.star_align(fastqs_qc)
            # Extract BAM paths from the bam_triples structure [(srr, bam_path, index_dir), ...].
            bams = [(srr, Path(bam_path)) for srr, bam_path, _ in bam_triples]
            # Pass paired-end hints forward to featureCounts when available.
            paired_flags = {srr: bool(fq2) for srr, _c1, fq2, *_rest in fastqs_qc}
            bam_triples = [(srr, bam_path, idx_dir, paired_flags.get(srr)) for srr, bam_path, idx_dir in bam_triples]
        else:
            # Skip the alignment step and return an empty result.
            logger.info("Skipping alignment step because with_align=False")
            bam_triples = []
            bams = []

        counts = self.featurecounts(bam_triples, gtf=self.cfg.gtf)
        result["bam"] = bams
        result["counts"] = counts
        return result
