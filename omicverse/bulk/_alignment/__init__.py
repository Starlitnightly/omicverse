"""
OmicVerse 增强比对管道
支持多种输入类型：SRA数据、FASTQ数据
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, fields
# 设置日志
logger = logging.getLogger(__name__)
from .alignment import Alignment, AlignmentConfig
from .iseq_handler import ISeqHandler
from .pipeline_config import EnhancedAlignmentConfig, load_config, get_example_configs
from .tools_check import check_all_tools, check_tool_availability



__version__ = "2.0.0"
__author__ = "Zhi Luo"

# 主要导出类
__all__ = [
    # 核心类
    'Alignment',
    'AlignmentConfig',
    'EnhancedAlignmentConfig',
    'ISeqHandler',

    # Data preprocess pipline
    'geo_data_preprocess',
    'fq_data_preprocess',

    # 配置相关
    'load_config',
    'get_example_configs',

    # 工具检查
    'check_all_tools',
    'check_tool_availability',


]

# 便捷的工厂函数
def create_pipeline(
    config_source=None,
    *,
    work_dir: str = "work",
    threads: int = 8,
    genome: str = "human",
    input_type: str = "auto"
) -> Alignment:
    """
    创建管道实例的便捷函数

    Args:
        config_source: 配置来源（文件路径或配置字典）
        work_dir: 工作目录
        threads: 线程数
        genome: 基因组类型
        input_type: 输入类型

    Returns:
        Alignment实例
    """
    if config_source is not None:
        from .pipeline_config import load_config
        config = load_config(config_source)
    else:
        # 使用基础配置
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
    download_method: str = "prefetch",  # 新增：下载方法，可选 "prefetch" 或 "iseq"
    # iseq 特定参数
    iseq_gzip: bool = True,
    iseq_aspera: bool = False,
    iseq_database: str = "sra",
    iseq_protocol: str = "ftp",
    iseq_parallel: int = 4,
    iseq_threads: int = 8
) -> dict:
    """
    一键运行分析的便捷函数

    Args:
        input_data: 输入数据
        config: 配置对象或配置源
        input_type: 输入类型
        with_align: 是否进行比对
        work_dir: 工作目录
        threads: 线程数
        genome: 基因组类型
        sample_prefix: 样本前缀（仅对公司数据）
        download_method: 下载方法，可选 "prefetch" (默认) 或 "iseq"
            - prefetch: 使用 NCBI SRA Toolkit (prefetch + fasterq-dump)
            - iseq: 使用 iseq 工具 (支持多数据库、Aspera 加速、直接下载 gzip 等)
        iseq_gzip: 下载gzip格式的FASTQ文件 (iseq模式有效, 默认: True)
        iseq_aspera: 使用Aspera加速下载 (iseq模式有效, 默认: False)
        iseq_database: 选择数据库: ena, sra (iseq模式有效, 默认: ena)
        iseq_protocol: 选择协议: ftp, https (iseq模式有效, 默认: ftp)
        iseq_parallel: 并行下载数 (iseq模式有效, 默认: 4)
        iseq_threads: 处理线程数 (iseq模式有效, 默认: 8)

    Returns:
        分析结果
    """
    # 创建或加载配置
    if config is None:
        pipeline_config = AlignmentConfig(
            work_root=Path(work_dir),
            threads=threads,
            genome=genome,
            download_method=download_method,  # 新增：下载方法
            # iseq 特定配置
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
        # 确保下载方法被设置
        if not hasattr(pipeline_config, 'download_method'):
            pipeline_config.download_method = download_method
    else:
        pipeline_config = config
        # 确保下载方法被设置
        if not hasattr(pipeline_config, 'download_method'):
            pipeline_config.download_method = download_method

    # 创建管道
    pipeline = Alignment(pipeline_config)

    # 检查工具
    if not check_all_tools():
        raise RuntimeError("Required tools are not available. Please install missing tools.")

    # 如果使用iseq下载方法，额外检查axel（Jupyter Lab环境适配）
    if download_method == "iseq":
        from . import tools_check as _tools_check
        logger.info("检测到使用iseq下载方法，正在检查axel依赖...")
        axel_available, axel_path = _tools_check.check_axel(auto_install=True)
        if not axel_available:
            logger.warning(f"axel不可用: {axel_path}。iseq可能无法正常工作，但将继续尝试...")
        else:
            logger.info(f"axel可用: {axel_path}")

    # 运行分析
    return pipeline.run_pipeline(
        input_data=input_data,
        input_type=input_type,
        with_align=with_align,
        sample_prefix=sample_prefix
    )

# 配置模板
def get_config_template():
    """获取配置模板"""
    return {
        "基础SRA分析": {
            "work_dir": "work_sra",
            "threads": 16,
            "genome": "human",
            "input_type": "sra"
        },
        "FASTQ文件分析": {
            "work_dir": "work_fastq",
            "threads": 12,
            "genome": "mouse",
            "input_type": "fastq"
        },
        "快速测试": {
            "work_dir": "work_test",
            "threads": 4,
            "genome": "human",
            "input_type": "auto"
        }
    }

# 版本信息
def get_version_info():
    """获取版本信息"""
    return {
        "version": __version__,
        "author": __author__,
        "features": [
            "支持SRA数据下载和处理",
            "支持公司FASTQ数据",
            "支持直接FASTQ文件输入",
            "自动输入类型检测",
            "统一样本ID管理",
            "增强的工具检查和安装指引",
            "灵活的配置系统"
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
    # 1) 直接对象
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

    # 2) 配置文件路径
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

    # 3) dict / None 合并 overrides
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
    将扁平 fastq 列表配对为 [(sample_id, R1, R2), ...]
    规则：优先识别 _R1/_R2、.R1/.R2、_1/_2；其余视为单端（只有 R1）。
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
            # 视为单端
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
    input_type: str = "fastq",         # 与 geo 风格一致，默认即 fastq
    with_align: bool = True,
    work_dir: str = "work",
    threads: int = 8,
    genome: str = "human",
    sample_prefix: str = None,         # 为了接口一致性保留，不在此函数中使用
    # 保持与 geo 一致的扩展方式：其余 AlignmentConfig 字段通过 kwargs 覆盖
    **kwargs
) -> dict:
    """
    与 geo_data_preprocess 同风格的 FASTQ 入口：
    - 不做 prefetch / fasterq-dump；
    - 接受扁平 fastq 列表并自动配对；
    - 后续步骤（fastp / STAR / featureCounts）与 geo 一致；
    - 返回结构与 geo_data_preprocess 一致。
    """
    # 形参 + kwargs → 覆盖 AlignmentConfig，同 geo 的策略
    overrides = dict(
        work_root=Path(work_dir),
        threads=threads,
        genome=genome,
        # 可以用 kwargs 继续覆盖 fastp_enabled / memory / gtf / simple_counts 等
    )
    overrides.update(kwargs or {})

    # 解析最终配置
    pipeline_config, unknown = _resolve_acfg(config, **overrides)
    if unknown:
        logger.warning(f"[fq_data_preprocess] Ignored unknown config keys: {unknown}")

    # 工具检查（与 geo 对齐）
    if not check_all_tools():
        raise RuntimeError("Required tools are not available. Please install missing tools.")

    # 创建管道
    pipeline = Alignment(pipeline_config)

    # 扁平列表 → 配对：(sample_id, fq1, fq2?) 列表
    # 复用 Alignment 内部的解析逻辑（会用 iseq_handler 以 R1/R2 规则配对）
    fastq_pairs: List[Tuple[str, Path, Optional[Path]]] = pipeline._parse_fastq_input(fastq_files)

    # 运行：直接从 FASTQ 进入统一流程（fastp → STAR → featureCounts）
    return pipeline.run_from_fastq(
        fastq_pairs,
        with_align=with_align
    )

if __name__ == "__main__":
    # 打印版本信息
    version_info = get_version_info()
    print(f"OmicVerse Enhanced Pipeline v{version_info['version']}")
    print(f"Author: {version_info['author']}")
    print("\n主要功能:")
    for feature in version_info['features']:
        print(f"  - {feature}")

    print("\n使用示例:")
    print("  from bulk._alignment import run_analysis")
    print("  result = run_analysis('SRR123456', work_dir='my_analysis')")
    print("\n  # 公司数据")
    print("  result = run_analysis('/path/to/fastq/files', input_type='company')")
    print("\n  # FASTQ文件")
    print("  result = run_analysis(['sample1_R1.fastq.gz', 'sample1_R2.fastq.gz'], input_type='fastq')")