# count_step.py featureCounts 步骤
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence

# 批量 wrapper：内部调用 featureCounts，支持 simple=True 只留 gene_id,count
from .count_tools import feature_counts_batch

def make_featurecounts_step(
    out_root: str = "work/counts",
    simple: bool = True,            # 只保留 gene_id,count
    gtf: str | None = None,         # 留空表示“使用 ensure_star_index 时下载到的 GENCODE GTF”（内部可从 index_root 推断）
    by: str = "auto",               # "srr" 或 "accession" 或 "auto"
    threads: int = 8,
    gtf_path: str | None = None,
):
    """
    输入：BAM 列表（[(srr, bam), ...]）
    输出：
      - 每样本：work/counts/{SRR}/{SRR}.counts.txt（或 .csv）
      - 可选：合并矩阵：work/counts/matrix.{by}.csv
    验证：每样本计数文件存在且行数>0
    """
    def _cmd(bam_pairs: Sequence[tuple[str, str]], logger=None, gtf: str | None = None):
        """
        bam_pairs: [(srr, bam_path), ...]
        gtf:       ✅ 新增：运行时也可显式传入 GTF（优先级最高）
        """
        os.makedirs(out_root, exist_ok=True)

        # ✅ 统一选择 GTF： 运行时 gtf > 工厂入参 gtf_path > 环境变量 FC_GTF_HINT
        gtf_use = gtf or gtf_path or os.environ.get("FC_GTF_HINT")
        if not gtf_use or not os.path.exists(gtf_use):
            raise RuntimeError(
                "[featureCounts] GTF not provided and FC_GTF_HINT not set or file missing.\n"
                f"  - gtf (runtime): {gtf}\n"
                f"  - gtf_path (factory): {gtf_path}\n"
                f"  - FC_GTF_HINT (env): {os.environ.get('FC_GTF_HINT')}"
            )

        # 调用你的批量计数函数（保持原有参数不变）
        return feature_counts_batch(
            bam_items=list(bam_pairs),   # [(srr, bam)]
            out_dir=out_root,
            gtf=gtf_use,
            simple=simple,
            by=by,
            threads=threads,
            max_workers=None,            # 如需并行可按需加
        )


    return {
        "name": "featurecounts",
        "command": _cmd,  # 接收 [(srr, bam), ...]
        "outputs": [f"{out_root}" + "/{SRR}/{SRR}.counts.txt"],  # 取决于你内部命名：txt/csv 二选一
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fs),
        "takes": "BAM_PATHS",
        "yields": "COUNT_TABLES"
    }