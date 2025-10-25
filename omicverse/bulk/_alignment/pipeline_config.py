"""
增强的管道配置文件
支持多种输入类型和灵活的配置选项
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
import json
import yaml

@dataclass
class EnhancedAlignmentConfig:
    """增强的比对管道配置"""

    # 基础配置
    work_root: Path = Path("work")
    threads: int = 8
    memory: str = "8G"
    genome: Literal["human", "mouse", "custom"] = "human"

    # 输入类型配置
    input_type: Literal["auto", "sra", "fastq", "company"] = "auto"

    # SRA配置
    prefetch_enabled: bool = True
    prefetch_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_size": "100G",
        "transport": "fasp",
        "timeout": 300
    })

    # iSeq/公司数据配置
    iseq_enabled: bool = False
    iseq_sample_prefix: str = "Sample"
    iseq_sample_pattern: str = "auto"  # auto, filename, directory
    iseq_validate_files: bool = True
    iseq_paired_end: bool = True

    # FASTQ处理配置
    fastq_extensions: List[str] = field(default_factory=lambda: [
        ".fq", ".fastq", ".fq.gz", ".fastq.gz"
    ])

    # QC配置
    fastp_enabled: bool = True
    fastp_params: Dict[str, Any] = field(default_factory=lambda: {
        "qualified_quality_phred": 20,
        "length_required": 50,
        "detect_adapter_for_pe": True
    })

    # 比对配置
    align_enabled: bool = True
    star_index_root: Path = Path("index")
    star_align_root: Path = Path("work/star")
    star_params: Dict[str, Any] = field(default_factory=lambda: {
        "gencode_release": "v44",
        "sjdb_overhang": 149,
        "outFilterMultimapNmax": 20,
        "alignSJoverhangMin": 8,
        "alignSJDBoverhangMin": 1
    })

    # 定量配置
    featurecounts_enabled: bool = True
    counts_root: Path = Path("work/counts")
    featurecounts_params: Dict[str, Any] = field(default_factory=lambda: {
        "gtf": None,  # 将自动检测
        "simple": True,
        "by": "gene_id",
        "t": "exon",
        "g": "gene_id"
    })

    # 输出配置
    gzip_fastq: bool = True
    keep_intermediate: bool = True

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # 并行配置
    max_workers: Optional[int] = None  # None表示使用所有可用CPU

    # 错误处理配置
    continue_on_error: bool = False
    retry_attempts: int = 3
    retry_delay: int = 5

    @classmethod
    def from_file(cls, config_file: str | Path) -> "EnhancedAlignmentConfig":
        """从配置文件加载"""
        config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        if config_file.suffix.lower() == '.json':
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        elif config_file.suffix.lower() in ['.yml', '.yaml']:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")

        # 转换路径字符串为Path对象
        for field_name, field_type in cls.__annotations__.items():
            if field_type == Path and field_name in config_data:
                config_data[field_name] = Path(config_data[field_name])

        return cls(**config_data)

    def to_file(self, output_file: str | Path, format: str = "json") -> None:
        """保存配置到文件"""
        output_file = Path(output_file)

        # 转换Path对象为字符串
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                config_dict[field_name] = str(field_value)
            else:
                config_dict[field_name] = field_value

        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() in ["yml", "yaml"]:
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported output format: {format}")

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []

        # 检查线程数
        if self.threads < 1:
            errors.append("threads must be >= 1")

        # 检查内存格式
        if not self.memory.endswith(('G', 'M', 'K')):
            errors.append("memory must end with G, M, or K")

        # 检查路径
        path_fields = ['work_root', 'star_index_root', 'star_align_root', 'counts_root']
        for field in path_fields:
            path_value = getattr(self, field)
            if not isinstance(path_value, Path):
                errors.append(f"{field} must be a Path object")

        # 检查基因组
        if self.genome not in ["human", "mouse", "custom"]:
            errors.append(f"genome must be one of: human, mouse, custom")

        # 检查输入类型
        if self.input_type not in ["auto", "sra", "fastq", "company"]:
            errors.append(f"input_type must be one of: auto, sra, fastq, company")

        # 检查iSeq配置
        if self.iseq_enabled and not self.iseq_sample_pattern in ["auto", "filename", "directory"]:
            errors.append(f"iseq_sample_pattern must be one of: auto, filename, directory")

        return errors

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """获取特定工具的配置"""
        tool_configs = {
            'star': self.star_params,
            'fastp': self.fastp_params,
            'featurecounts': self.featurecounts_params,
            'prefetch': self.prefetch_params
        }
        return tool_configs.get(tool_name, {})

    def update_tool_config(self, tool_name: str, config: Dict[str, Any]) -> None:
        """更新特定工具的配置"""
        tool_configs = {
            'star': 'star_params',
            'fastp': 'fastp_params',
            'featurecounts': 'featurecounts_params',
            'prefetch': 'prefetch_params'
        }

        if tool_name in tool_configs:
            attr_name = tool_configs[tool_name]
            current_config = getattr(self, attr_name)
            current_config.update(config)
            setattr(self, attr_name, current_config)

# 默认配置文件模板
def create_default_config_template(output_file: str | Path) -> None:
    """创建默认配置文件模板"""
    config = EnhancedAlignmentConfig()
    config.to_file(output_file, format="yaml")
    print(f"Default config template created: {output_file}")

def load_config(config_source: str | Path | Dict[str, Any]) -> EnhancedAlignmentConfig:
    """
    从多种来源加载配置

    Args:
        config_source: 配置来源，可以是文件路径或配置字典

    Returns:
        配置对象
    """
    if isinstance(config_source, dict):
        return EnhancedAlignmentConfig(**config_source)
    elif isinstance(config_source, (str, Path)):
        return EnhancedAlignmentConfig.from_file(config_source)
    else:
        raise ValueError(f"Unsupported config source type: {type(config_source)}")

# 示例配置
def get_example_configs() -> Dict[str, EnhancedAlignmentConfig]:
    """获取示例配置"""

    # SRA数据处理配置
    sra_config = EnhancedAlignmentConfig(
        work_root=Path("work_sra"),
        input_type="sra",
        threads=16,
        genome="human"
    )

    # 公司数据处理配置
    company_config = EnhancedAlignmentConfig(
        work_root=Path("work_company"),
        input_type="company",
        threads=8,
        genome="human",
        iseq_enabled=True,
        iseq_sample_prefix="TumorSample",
        iseq_sample_pattern="auto"
    )

    # FASTQ文件处理配置
    fastq_config = EnhancedAlignmentConfig(
        work_root=Path("work_fastq"),
        input_type="fastq",
        threads=12,
        genome="mouse"
    )

    # 快速测试配置
    test_config = EnhancedAlignmentConfig(
        work_root=Path("work_test"),
        threads=4,
        genome="human",
        fastp_params={"length_required": 30},
        star_params={"outFilterMultimapNmax": 10}
    )

    return {
        "sra": sra_config,
        "company": company_config,
        "fastq": fastq_config,
        "test": test_config
    }

if __name__ == "__main__":
    # 创建默认配置模板
    create_default_config_template("pipeline_config.yaml")

    # 显示示例配置
    examples = get_example_configs()
    for name, config in examples.items():
        print(f"\n{name.upper()} 配置:")
        print(f"  工作目录: {config.work_root}")
        print(f"  线程数: {config.threads}")
        print(f"  基因组: {config.genome}")
        print(f"  输入类型: {config.input_type}")
        if config.iseq_enabled:
            print(f"  iSeq样本前缀: {config.iseq_sample_prefix}")