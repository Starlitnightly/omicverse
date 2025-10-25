"""
iSeq Handler - 处理公司提供的FASTQ数据
支持本地FASTQ文件的样本ID分配、验证和格式转换
"""
from __future__ import annotations

import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence
import logging

logger = logging.getLogger(__name__)

class ISeqHandler:
    """处理公司提供的FASTQ数据"""

    def __init__(self,
                 sample_id_pattern: str = "auto",
                 paired_end: bool = True,
                 validate_files: bool = True):
        """
        初始化iSeq处理器

        Args:
            sample_id_pattern: 样本ID提取模式，可选 "auto", "filename", "directory"
            paired_end: 是否为双端测序
            validate_files: 是否验证文件完整性
        """
        self.sample_id_pattern = sample_id_pattern
        self.paired_end = paired_end
        self.validate_files = validate_files

    def discover_fastq_files(self,
                           input_path: str | Path,
                           file_extensions: List[str] = None) -> List[Path]:
        """
        发现指定路径下的FASTQ文件

        Args:
            input_path: 输入路径（文件或目录）
            file_extensions: 文件扩展名列表，默认 [".fq", ".fastq", ".fq.gz", ".fastq.gz"]

        Returns:
            FASTQ文件路径列表
        """
        if file_extensions is None:
            file_extensions = [".fq", ".fastq", ".fq.gz", ".fastq.gz"]

        input_path = Path(input_path)
        fastq_files = []

        if input_path.is_file():
            # 单个文件
            if any(str(input_path).endswith(ext) for ext in file_extensions):
                fastq_files.append(input_path)
        elif input_path.is_dir():
            # 目录递归搜索
            for ext in file_extensions:
                fastq_files.extend(input_path.rglob(f"*{ext}"))
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if not fastq_files:
            raise ValueError(f"No FASTQ files found in {input_path}")

        # 排序确保一致性
        fastq_files.sort()
        logger.info(f"Discovered {len(fastq_files)} FASTQ files")
        return fastq_files

    def extract_sample_id(self, file_path: Path) -> str:
        """
        从文件路径提取样本ID

        Args:
            file_path: FASTQ文件路径

        Returns:
            样本ID
        """
        if self.sample_id_pattern == "auto":
            return self._auto_extract_sample_id(file_path)
        elif self.sample_id_pattern == "filename":
            return self._extract_from_filename(file_path)
        elif self.sample_id_pattern == "directory":
            return self._extract_from_directory(file_path)
        else:
            raise ValueError(f"Unknown pattern: {self.sample_id_pattern}")

    def _auto_extract_sample_id(self, file_path: Path) -> str:
        """自动提取样本ID"""
        # 尝试多种模式
        filename = file_path.stem  # 移除扩展名

        # 模式1: 移除常见的测序后缀 (_R1, _R2, .R1, .R2, _1, _2)
        cleaned = re.sub(r'[._][Rr]?[12]$', '', filename)

        # 模式2: 如果还包含下划线，取第一部分
        if '_' in cleaned:
            cleaned = cleaned.split('_')[0]

        # 模式3: 检查是否符合常见的样本ID格式
        if re.match(r'^[A-Za-z0-9_-]+$', cleaned):
            return cleaned
        else:
            # 如果都不匹配，使用文件名（移除特殊字符）
            return re.sub(r'[^A-Za-z0-9_-]', '', cleaned)

    def _extract_from_filename(self, file_path: Path) -> str:
        """从文件名提取样本ID"""
        filename = file_path.stem
        # 移除测序后缀
        return re.sub(r'[._][Rr]?[12]$', '', filename)

    def _extract_from_directory(self, file_path: Path) -> str:
        """从目录名提取样本ID"""
        return file_path.parent.name

    def group_paired_end(self, fastq_files: List[Path]) -> List[Tuple[str, Path, Optional[Path]]]:
        """
        将FASTQ文件按双端测序配对

        Args:
            fastq_files: FASTQ文件列表

        Returns:
            配对结果列表: [(sample_id, R1_path, R2_path), ...]
        """
        if not self.paired_end:
            # 单端测序
            return [(self.extract_sample_id(fq), fq, None) for fq in fastq_files]

        # 双端测序配对
        sample_groups: Dict[str, List[Path]] = {}

        for fq_file in fastq_files:
            sample_id = self.extract_sample_id(fq_file)
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(fq_file)

        # 配对验证
        paired_results = []
        for sample_id, files in sample_groups.items():
            if len(files) == 1:
                # 单文件，可能是单端或缺失R2
                logger.warning(f"Sample {sample_id} has only one file: {files[0]}")
                paired_results.append((sample_id, files[0], None))
            elif len(files) == 2:
                # 双文件，需要区分R1和R2
                r1_file, r2_file = self._identify_r1_r2(files)
                paired_results.append((sample_id, r1_file, r2_file))
            else:
                # 多于2个文件，需要更复杂的配对逻辑
                logger.warning(f"Sample {sample_id} has {len(files)} files, using first two")
                r1_file, r2_file = self._identify_r1_r2(files[:2])
                paired_results.append((sample_id, r1_file, r2_file))

        return paired_results

    def _identify_r1_r2(self, files: List[Path]) -> Tuple[Path, Path]:
        """识别R1和R2文件"""
        # 根据文件名中的模式识别
        r1_candidates = [f for f in files if re.search(r'[._][Rr]1[._]', str(f))]
        r2_candidates = [f for f in files if re.search(r'[._][Rr]2[._]', str(f))]

        if len(r1_candidates) == 1 and len(r2_candidates) == 1:
            return r1_candidates[0], r2_candidates[0]

        # 如果没有明确的R1/R2标记，按字母顺序
        files_sorted = sorted(files, key=lambda x: str(x))
        if len(files_sorted) >= 2:
            return files_sorted[0], files_sorted[1]

        # 兜底方案
        return files[0], files[1] if len(files) > 1 else None

    def validate_fastq_files(self, fastq_pairs: List[Tuple[str, Path, Optional[Path]]]) -> bool:
        """
        验证FASTQ文件的有效性

        Args:
            fastq_pairs: 配对的FASTQ文件列表

        Returns:
            是否所有文件都有效
        """
        all_valid = True

        for sample_id, r1_path, r2_path in fastq_pairs:
            # 验证R1文件
            if not self._validate_single_file(r1_path):
                logger.error(f"Invalid R1 file for sample {sample_id}: {r1_path}")
                all_valid = False
                continue

            # 验证R2文件（如果存在）
            if r2_path and not self._validate_single_file(r2_path):
                logger.error(f"Invalid R2 file for sample {sample_id}: {r2_path}")
                all_valid = False
                continue

            logger.info(f"Validated sample {sample_id}: R1={r1_path}, R2={r2_path}")

        return all_valid

    def _validate_single_file(self, file_path: Path) -> bool:
        """验证单个FASTQ文件"""
        if not file_path.exists():
            return False

        if file_path.stat().st_size == 0:
            return False

        # 简单的格式验证：检查文件头
        try:
            if str(file_path).endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline().strip()
            else:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()

            # FASTQ文件应该以@开头
            return first_line.startswith('@')
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    def create_sample_metadata(self,
                             fastq_pairs: List[Tuple[str, Path, Optional[Path]]],
                             output_file: Optional[Path] = None) -> pd.DataFrame:
        """
        创建样本元数据表

        Args:
            fastq_pairs: 配对的FASTQ文件列表
            output_file: 输出文件路径（可选）

        Returns:
            样本元数据DataFrame
        """
        metadata = []

        for sample_id, r1_path, r2_path in fastq_pairs:
            meta = {
                'sample_id': sample_id,
                'r1_path': str(r1_path),
                'r2_path': str(r2_path) if r2_path else None,
                'paired_end': r2_path is not None,
                'file_size_r1_mb': r1_path.stat().st_size / (1024 * 1024),
                'file_size_r2_mb': r2_path.stat().st_size / (1024 * 1024) if r2_path else 0,
            }
            metadata.append(meta)

        df = pd.DataFrame(metadata)

        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Sample metadata saved to {output_file}")

        return df

    def process_company_data(self,
                           input_path: str | Path,
                           output_dir: str | Path,
                           sample_prefix: str = "Sample") -> Dict[str, Any]:
        """
        处理公司提供的FASTQ数据的主函数

        Args:
            input_path: 输入FASTQ文件或目录路径
            output_dir: 输出目录
            sample_prefix: 样本ID前缀

        Returns:
            处理结果字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing company FASTQ data from {input_path}")

        # 1. 发现FASTQ文件
        fastq_files = self.discover_fastq_files(input_path)

        # 2. 配对双端测序数据
        fastq_pairs = self.group_paired_end(fastq_files)

        # 3. 验证文件
        if self.validate_files:
            if not self.validate_fastq_files(fastq_pairs):
                raise ValueError("Some FASTQ files failed validation")

        # 4. 创建样本元数据
        metadata_file = output_dir / "sample_metadata.csv"
        metadata_df = self.create_sample_metadata(fastq_pairs, metadata_file)

        # 5. 创建标准化的样本ID映射
        sample_id_mapping = {}
        for i, (original_id, r1_path, r2_path) in enumerate(fastq_pairs):
            standardized_id = f"{sample_prefix}_{i+1:03d}"
            sample_id_mapping[standardized_id] = {
                'original_id': original_id,
                'r1_path': str(r1_path),
                'r2_path': str(r2_path) if r2_path else None,
                'paired_end': r2_path is not None
            }

        # 6. 保存映射文件
        mapping_file = output_dir / "sample_id_mapping.json"
        import json
        with open(mapping_file, 'w') as f:
            json.dump(sample_id_mapping, f, indent=2)

        logger.info(f"Processed {len(fastq_pairs)} samples")
        logger.info(f"Results saved to {output_dir}")

        return {
            'sample_pairs': fastq_pairs,
            'metadata_df': metadata_df,
            'sample_id_mapping': sample_id_mapping,
            'metadata_file': metadata_file,
            'mapping_file': mapping_file,
            'output_dir': output_dir
        }