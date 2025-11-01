"""
OmicVerse enhanced alignment pipeline.
Supports multiple input types: SRA data and FASTQ data.
"""

import logging
import sys
import types
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, fields
# Configure logging
logger = logging.getLogger(__name__)
from .alignment import Alignment, AlignmentConfig
from .iseq_handler import ISeqHandler
from .pipeline_config import EnhancedAlignmentConfig, load_config, get_example_configs
from .tools_check import check_all_tools, check_tool_availability



__version__ = "2.0.0"
__author__ = "Zhi Luo"

# Public exports
__all__ = [
    # Core classes
    'Alignment',
    'AlignmentConfig',
    'EnhancedAlignmentConfig',
    'ISeqHandler',

    # Data preprocess pipline
    'geo_data_preprocess',
    'fq_data_preprocess',

    # Configuration helpers
    'load_config',
    'get_example_configs',

    # Tool check helpers
    'check_all_tools',
    'check_tool_availability',


]


def _register_alignment_namespace() -> None:
    """Expose alignment helpers under the public ``<package>.alignment.bulk`` path."""
    root_pkg = __name__.split(".", 1)[0]
    pkg_name = f"{root_pkg}.alignment"
    bulk_name = f"{pkg_name}.bulk"

    alignment_pkg = sys.modules.get(pkg_name)
    if alignment_pkg is None:
        alignment_pkg = types.ModuleType(pkg_name)
        alignment_pkg.__path__ = []
        alignment_pkg.__package__ = pkg_name
        sys.modules[pkg_name] = alignment_pkg
    else:
        # Ensure the namespace behaves like a package
        if not hasattr(alignment_pkg, "__path__"):
            alignment_pkg.__path__ = []
        if getattr(alignment_pkg, "__package__", None) is None:
            alignment_pkg.__package__ = pkg_name

    bulk_module = sys.modules.get(bulk_name)
    if bulk_module is None:
        bulk_module = types.ModuleType(bulk_name)
        bulk_module.__package__ = bulk_name
        sys.modules[bulk_name] = bulk_module

    allowed_names = set(__all__)
    for name, obj in globals().items():
        if name.startswith("_"):
            continue
        if isinstance(obj, (types.FunctionType, type)):
            allowed_names.add(name)

    exported_items = {
        name: globals()[name]
        for name in allowed_names
        if name in globals()
    }

    for name, obj in exported_items.items():
        setattr(bulk_module, name, obj)
        module_name = getattr(obj, "__module__", None)
        if module_name is None:
            continue
        if not isinstance(obj, (type, types.FunctionType)):
            # Preserve module metadata for non-callable objects
            continue
        namespace_prefix = f"{root_pkg}."
        if not module_name.startswith(namespace_prefix):
            continue
        try:
            obj.__module__ = bulk_name
        except (AttributeError, TypeError):
            # Some objects (e.g., functools.partial) may not allow reassignment
            pass

    existing_bulk_all = set(getattr(bulk_module, "__all__", []))
    bulk_module.__all__ = sorted(existing_bulk_all | set(exported_items.keys()))
    bulk_module.__doc__ = __doc__

    alignment_pkg.bulk = bulk_module
    existing_alignment_all = set(getattr(alignment_pkg, "__all__", []))
    alignment_pkg.__all__ = sorted(existing_alignment_all | {"bulk"})

    parent_pkg = sys.modules.get(root_pkg)
    if parent_pkg is not None:
        setattr(parent_pkg, "alignment", alignment_pkg)

# Convenience factory functions
def create_pipeline(
    config_source=None,
    *,
    work_dir: str = "work",
    threads: int = 8,
    genome: str = "human",
    input_type: str = "auto"
) -> Alignment:
    """
    Convenience helper for instantiating a pipeline.

    Args:
        config_source: Config source (file path or dictionary).
        work_dir: Working directory.
        threads: Thread count.
        genome: Genome identifier.
        input_type: Input type string.

    Returns:
        Alignment instance.
    """
    if config_source is not None:
        from .pipeline_config import load_config
        config = load_config(config_source)
    else:
        # Use the default configuration.
        config = AlignmentConfig(
            work_root=Path(work_dir),
            threads=threads,
            genome=genome
        )

    return Alignment(config)

def geo_data_preprocess(
    input_data,
    *,
    config=None,
    input_type: str = "auto",
    with_align: bool = True,
    work_dir: str = "work",
    threads: int = 8,
    genome: str = "human",
    sample_prefix: str = None,
    download_method: str = "prefetch",  # Download method: "prefetch" or "iseq".
    # iseq-specific parameters
    iseq_gzip: bool = True,
    iseq_aspera: bool = False,
    iseq_database: str = "sra",
    iseq_protocol: str = "ftp",
    iseq_parallel: int = 4,
    iseq_threads: int = 8
) -> dict:
    """
    Convenience helper that runs the analysis end-to-end.

    Args:
        input_data: Input payload.
        config: Configuration object or source.
        input_type: Input type string.
        with_align: Whether to perform alignment.
        work_dir: Working directory.
        threads: Thread count.
        genome: Genome identifier.
        sample_prefix: Sample ID prefix (vendor data only).
        download_method: Download method, "prefetch" (default) or "iseq".
            - prefetch: Use NCBI SRA Toolkit (prefetch + fasterq-dump).
            - iseq: Use the iseq tool (multi-database, Aspera acceleration, gzip downloads, etc.).
        iseq_gzip: Download FASTQ as gzip (iseq mode only, default True).
        iseq_aspera: Enable Aspera acceleration (iseq mode only, default False).
        iseq_database: Database to target: ena or sra (iseq mode only, default ena).
        iseq_protocol: Protocol to use: ftp or https (iseq mode only, default ftp).
        iseq_parallel: Number of parallel downloads (iseq mode only, default 4).
        iseq_threads: Threads for iseq processing (iseq mode only, default 8).

    Returns:
        Analysis result dictionary.
    """
    # Create or load the configuration.
    if config is None:
        pipeline_config = AlignmentConfig(
            work_root=Path(work_dir),
            threads=threads,
            genome=genome,
            download_method=download_method,  # Download method selection.
            # iseq-specific configuration.
            iseq_gzip=iseq_gzip,
            iseq_aspera=iseq_aspera,
            iseq_database=iseq_database,
            iseq_protocol=iseq_protocol,
            iseq_parallel=iseq_parallel,
            iseq_threads=iseq_threads
        )
    elif isinstance(config, (str, Path)):
        from .pipeline_config import load_config
        pipeline_config = load_config(config)
        # Ensure the download method attribute exists.
        if not hasattr(pipeline_config, 'download_method'):
            pipeline_config.download_method = download_method
    else:
        pipeline_config = config
        # Ensure the download method attribute exists.
        if not hasattr(pipeline_config, 'download_method'):
            pipeline_config.download_method = download_method

    # Create the pipeline.
    pipeline = Alignment(pipeline_config)

    # Check tool availability.
    if not check_all_tools():
        raise RuntimeError("Required tools are not available. Please install missing tools.")

    # When using the iseq download method, ensure axel is available (Jupyter Lab compatibility).
    if download_method == "iseq":
        from . import tools_check as _tools_check
        logger.info("Detected download_method=iseq; verifying axel dependency…")
        axel_available, axel_path = _tools_check.check_axel(auto_install=True)
        if not axel_available:
            logger.warning(f"axel unavailable: {axel_path}. iseq may not function fully, continuing anyway…")
        else:
            logger.info(f"axel available: {axel_path}")

    # Execute the pipeline.
    return pipeline.run_pipeline(
        input_data=input_data,
        input_type=input_type,
        with_align=with_align,
        sample_prefix=sample_prefix
    )

# Configuration template
def get_config_template():
    """Retrieve configuration templates."""
    return {
        "Basic SRA analysis": {
            "work_dir": "work_sra",
            "threads": 16,
            "genome": "human",
            "input_type": "sra"
        },
        "FASTQ file analysis": {
            "work_dir": "work_fastq",
            "threads": 12,
            "genome": "mouse",
            "input_type": "fastq"
        },
        "Quick test": {
            "work_dir": "work_test",
            "threads": 4,
            "genome": "human",
            "input_type": "auto"
        }
    }

# Version information
def get_version_info():
    """Retrieve version information."""
    return {
        "version": __version__,
        "author": __author__,
        "features": [
            "Supports SRA data download and processing",
            "Supports vendor FASTQ data",
            "Supports direct FASTQ file input",
            "Automatic input type detection",
            "Unified sample ID management",
            "Enhanced tool checks and installation guidance",
            "Flexible configuration system"
        ]
    }



def _filter_to_acfg_fields(d: Dict[str, Any]):
    valid = {f.name for f in fields(AlignmentConfig)}
    ok = {k: v for k, v in d.items() if k in valid}
    unknown = [k for k in d if k not in valid]
    return ok, unknown

def _resolve_acfg(
    config: Optional[Union[AlignmentConfig, Dict[str, Any], str, Path]] = None,
    **overrides
) -> tuple[AlignmentConfig, list]:
    # 1) Direct object
    if isinstance(config, AlignmentConfig):
        ok, unknown = _filter_to_acfg_fields(overrides)
        for k, v in ok.items():
            setattr(config, k, v)
        if not isinstance(config.work_root, Path):
            config.work_root = Path(config.work_root)
        if getattr(config, "gtf", None) and isinstance(config.gtf, str):
            config.gtf = Path(config.gtf)
        return config, unknown

    base: Dict[str, Any] = {}

    # 2) Config file path
    if isinstance(config, (str, Path)):
        try:
            from .pipeline_config import load_config
        except ImportError:
            load_config = None
        if load_config is None:
            raise RuntimeError("config is a path, but pipeline_config.load_config is unavailable.")
        loaded = load_config(config)
        if isinstance(loaded, AlignmentConfig):
            base = {f.name: getattr(loaded, f.name) for f in fields(AlignmentConfig)}
        elif isinstance(loaded, dict):
            base = dict(loaded)
        else:
            raise TypeError(f"Unsupported config file content type: {type(loaded)}")

    # 3) Merge overrides with dict / None
    if isinstance(config, dict):
        base.update(config)
    base.update(overrides or {})

    ok, unknown = _filter_to_acfg_fields(base)
    acfg = AlignmentConfig(**ok)
    if not isinstance(acfg.work_root, Path):
        acfg.work_root = Path(acfg.work_root)
    if getattr(acfg, "gtf", None) and isinstance(acfg.gtf, str):
        acfg.gtf = Path(acfg.gtf)
    return acfg, unknown
# ======================================================================


def _pair_fastqs_flat(fastq_files: List[str]) -> List[Tuple[str, str, Optional[str]]]:
    """
    Pair a flat FASTQ list into [(sample_id, R1, R2), ...].

    Rule priority: detect _R1/_R2, .R1/.R2, _1/_2; otherwise treat as single-end (R1 only).
    """
    if not fastq_files:
        raise ValueError("No fastq files provided.")

    from collections import defaultdict
    files = [Path(p) for p in fastq_files]
    buckets = defaultdict(lambda: {"R1": None, "R2": None})

    for p in files:
        name = p.name
        stem = name
        for suf in (".fastq.gz", ".fq.gz", ".fastq", ".fq", ".gz"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]

        sample_id, role = None, None
        if "_R1" in stem:
            sample_id, role = stem.replace("_R1", ""), "R1"
        elif "_R2" in stem:
            sample_id, role = stem.replace("_R2", ""), "R2"
        elif ".R1" in stem:
            sample_id, role = stem.replace(".R1", ""), "R1"
        elif ".R2" in stem:
            sample_id, role = stem.replace(".R2", ""), "R2"
        elif stem.endswith("_1"):
            sample_id, role = stem[:-2], "R1"
        elif stem.endswith("_2"):
            sample_id, role = stem[:-2], "R2"
        else:
            # Treat as single-end
            sample_id, role = stem, "R1"

        if role == "R1":
            buckets[sample_id]["R1"] = str(p)
        elif role == "R2":
            buckets[sample_id]["R2"] = str(p)

    pairs: List[Tuple[str, str, Optional[str]]] = []
    for sid, d in buckets.items():
        if d["R1"] is None:
            raise ValueError(f"Sample {sid} missing R1 FASTQ.")
        pairs.append((sid, d["R1"], d["R2"]))
    return pairs


def fq_data_preprocess(
    fastq_files: List[str],
    *,
    config: Optional[Union[AlignmentConfig, Dict[str, Any], str, Path]] = None,
    input_type: str = "fastq",         # Align with geo style, defaults to fastq
    with_align: bool = True,
    work_dir: str = "work",
    threads: int = 8,
    genome: str = "human",
    sample_prefix: str = None,         # Retained for interface consistency; not used in this function
    # Preserve geo-style extensibility: override other AlignmentConfig fields via kwargs
    **kwargs
) -> dict:
    """
    FASTQ entry point following the same style as geo_data_preprocess:
    - Skip prefetch / fasterq-dump
    - Accept a flat FASTQ list and auto pair
    - Subsequent steps (fastp / STAR / featureCounts) match geo
    - Returns the same structure as geo_data_preprocess
    """
    # Parameters + kwargs -> override AlignmentConfig, same strategy as geo
    overrides = dict(
        work_root=Path(work_dir),
        threads=threads,
        genome=genome,
        # kwargs can override fastp_enabled / memory / gtf / simple_counts, etc.
    )
    overrides.update(kwargs or {})

    # Resolve final configuration
    pipeline_config, unknown = _resolve_acfg(config, **overrides)
    if unknown:
        logger.warning(f"[fq_data_preprocess] Ignored unknown config keys: {unknown}")

    # Tool check (aligned with geo)
    if not check_all_tools():
        raise RuntimeError("Required tools are not available. Please install missing tools.")

    # Create pipeline
    pipeline = Alignment(pipeline_config)

    # Flat list -> paired list: (sample_id, fq1, fq2?) entries
    # Reuse Alignment internal parsing logic (uses iseq_handler R1/R2 pairing rules)
    fastq_pairs: List[Tuple[str, Path, Optional[Path]]] = pipeline._parse_fastq_input(fastq_files)

    # Run: start from FASTQ and enter the unified workflow (fastp -> STAR -> featureCounts)
    return pipeline.run_from_fastq(
        fastq_pairs,
        with_align=with_align
    )


_register_alignment_namespace()

if __name__ == "__main__":
    # Print version information
    version_info = get_version_info()
    print(f"OmicVerse Enhanced Pipeline v{version_info['version']}")
    print(f"Author: {version_info['author']}")
    print("\nKey features:")
    for feature in version_info['features']:
        print(f"  - {feature}")

    print("\nUsage examples:")
    print("  from bulk._alignment import run_analysis")
    print("  result = run_analysis('SRR123456', work_dir='my_analysis')")
    print("\n  # Vendor data")
    print("  result = run_analysis('/path/to/fastq/files', input_type='company')")
    print("\n  # FASTQ files")
    print("  result = run_analysis(['sample1_R1.fastq.gz', 'sample1_R2.fastq.gz'], input_type='fastq')")
