# count_tools.py featureCounts 批量版本
from __future__ import annotations
import os, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


def _feature_counts_one(bam_path: str, out_dir: str, gtf: str, threads: int = 8, simple: bool = True):
    # -------------- 新增安全判断 --------------
    if gtf is None:
        gtf = os.environ.get("FC_GTF_HINT")
    if not gtf or not os.path.exists(gtf):
        raise RuntimeError(
            f"[featureCounts] Missing valid GTF file for {bam_path}. "
            "请确认 pipeline 已设置 FC_GTF_HINT 或传入 gtf 参数。"
        )
    # -----------------------------------------
    
    srr = Path(bam_path).stem.replace(".bam", "")
    out_prefix = Path(out_dir) / srr
    out_file = f"{out_prefix}.counts.txt"

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return srr, out_file

    cmd = [
        "featureCounts",
        "-T", str(threads),
        "-a", gtf,
        "-o", out_file,
        bam_path
    ]
    subprocess.run(cmd, check=True)
    
    # 简化输出（自动识别计数列）
    if simple and os.path.exists(out_file):
        df = pd.read_csv(out_file, sep="\t", comment="#")
        # featureCounts 的注释列
        annot_cols = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
        # 找出计数列（通常只有 1 列；列名是 BAM 名称/路径）
        count_cols = [c for c in df.columns if c not in annot_cols]
        if len(count_cols) == 0:
            raise ValueError(f"No count columns found in {out_file}. Got columns: {list(df.columns)}")
        if len(count_cols) > 1:
            # 多 BAM 情况下这里按需处理；本函数是“单个 bam”，理论上==1
            # 保险起见，取最后一列当计数列
            counts_col = count_cols[-1]
        else:
            counts_col = count_cols[0]
    
        df_simple = df[["Geneid", counts_col]].rename(
            columns={"Geneid": "gene_id", counts_col: srr}
        )
        df_simple.to_csv(out_file, sep="\t", index=False)
    
    return srr, out_file


def feature_counts_batch(
    bam_items: list[tuple[str, str]],  # [(srr, bam_path)]
    out_dir: str,
    gtf: str | None = None,
    simple: bool = True,
    by: str = "auto",
    threads: int = 8,
    max_workers: int | None = None
):
    """
    Run featureCounts on multiple BAM files.
    """
    os.makedirs(out_dir, exist_ok=True)
     # -------------- 新增安全判断 --------------
    if gtf is None:
        gtf = os.environ.get("FC_GTF_HINT")
    if not gtf or not os.path.exists(gtf):
        raise RuntimeError(
            "[featureCounts_batch] GTF not provided and FC_GTF_HINT not found. "
            "请在 pipeline 推断 GTF 或手动传入。"
        )
    # -----------------------------------------

    results, errors = [], []

    with ProcessPoolExecutor(max_workers=max_workers or min(8, os.cpu_count() // threads)) as ex:
        futures = {
            ex.submit(_feature_counts_one, bam, out_dir, gtf, threads, simple): srr
            for srr, bam in bam_items
        }
        for fut in as_completed(futures):
            srr = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[ERR] {srr}: {e}")

    # 结果汇总（更健壮：支持 Geneid 或 gene_id，自动识别计数列）
        # 结果汇总（保留 SRR→文件配对；在合并前就把计数列重命名为 SRR，避免重复列名）
    pairs = [(srr, f) for (srr, f) in results if os.path.exists(f)]
    if len(pairs) > 1:
        merged_df = None
        for srr, f in pairs:
            # 跳过注释行，避免列名被干扰
            df = pd.read_csv(f, sep="\t", comment="#")

            # 基因列：优先 gene_id；否则 Geneid；都没有则用第 1 列
            if "gene_id" in df.columns:
                gene_col = "gene_id"
            elif "Geneid" in df.columns:
                gene_col = "Geneid"
            else:
                gene_col = df.columns[0]

            # 计数列：去除元数据列后剩下的列（通常只有 1 列，为该样本计数）
            meta_cols = {"Chr", "Start", "End", "Strand", "Length", gene_col}
            count_cols = [c for c in df.columns if c not in meta_cols]
            if not count_cols:
                # 兜底：最后一列当作计数列
                count_col = df.columns[-1]
            else:
                # 常见只有 1 列；若有多列，取最后一列（通常为计数）
                count_col = count_cols[-1]

            # 只保留 gene_id + 该样本计数列，并把计数列名改成 SRR（唯一）
            df_simple = df[[gene_col, count_col]].copy()
            df_simple.columns = ["gene_id", srr]

            if merged_df is None:
                merged_df = df_simple
            else:
                merged_df = merged_df.merge(df_simple, on="gene_id", how="outer")

        # 现在 merged_df 的每个计数列名都是 SRR，天然唯一，无需再做后续的正则重命名
        out_path = Path(out_dir) / f"matrix.{by}.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"[OK] featureCounts merged matrix → {out_path}")
    '''table_files = [f for _, f in results if os.path.exists(f)]
    if len(table_files) > 1:
        merged_df = None
        for f in table_files:
            # 关键：跳过注释行，避免列名被干扰
            df = pd.read_csv(f, sep="\t", comment="#")
    
            # 基因列：优先 gene_id；否则用 Geneid；都没有就退而求其次第 1 列
            if "gene_id" in df.columns:
                gene_col = "gene_id"
            elif "Geneid" in df.columns:
                gene_col = "Geneid"
            else:
                gene_col = df.columns[0]  # 兜底
    
            # 计数列：排除注释列后余下的全是计数列（通常只有 1 列；列名是 BAM 路径/文件名）
            annot_cols = {gene_col, "Chr", "Start", "End", "Strand", "Length"}
            count_cols = [c for c in df.columns if c not in annot_cols]
            if len(count_cols) == 0:
                raise ValueError(f"No count columns found in {f}. Columns={list(df.columns)}")
    
            # 只留“基因 + 计数列”，并统一把基因列命名为 gene_id
            df = df[[gene_col] + count_cols].rename(columns={gene_col: "gene_id"})
    
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.merge(df, on="gene_id", how="outer")
    
        import re

        def _extract_srr(colname: str) -> str:
            m = re.search(r'(SRR\d{5,})', colname)
            if m:
                return m.group(1)
            return colname  # 如果没匹配到，保留原名
        
        # 对除 gene_id 外的列重命名
        new_cols = [merged_df.columns[0]] + [_extract_srr(c) for c in merged_df.columns[1:]]
        merged_df.columns = new_cols
        
        out_path = Path(out_dir) / f"matrix.{by}.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"[OK] featureCounts merged matrix → {out_path}")'''
    

    return {
        "tables": results,                 # 例如 [(srr, sample_table_path), ...]，按你现有结构
        "matrix": str(out_path),
        "failed": errors if 'errors' in locals() else [],
    }

def run_featurecounts_auto(
    bam_files: List[str | Path],
    index_dir: str | Path,
    out_dir: str = "results",
    accession_id: Optional[str] = None,   # e.g. "GSE157103"
    srr_id: Optional[str] = None,         # e.g. "SRR12544419"（单样本时）
    threads: int = 12,
    output_csv: bool = True,
    simple: bool = True,                # ✅ 新增：是否精简输出
    featurecounts_bin: Optional[str] = None,  # 可留空，自动用 PATH 解析
) -> Path:
    """
    - 自动从 STAR index 定位 GTF（并在需要时解压 .gtf.gz）
    - 根据 accession 或 SRR 自动命名输出文件
    - 默认将 featureCounts 的制表符输出转换为 CSV
    """
    bam_files = [str(Path(b)) for b in bam_files]
    index_dir = Path(index_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 自动找 GTF
    gtf_path = get_gtf_for_index(index_dir)  # ← 核心：无需手动提供

    # 2) 智能命名
    if accession_id and not srr_id:
        prefix = f"{accession_id}_counts"
    elif srr_id:
        prefix = f"{srr_id}_counts"
    else:
        prefix = f"counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_txt = Path(out_dir) / f"{prefix}.txt"   # 先让 featureCounts 按默认写 .txt（TSV）

    # 3) 定位 featureCounts 可执行文件
    if featurecounts_bin is None:
        import shutil
        featurecounts_bin = shutil.which("featureCounts") or "featureCounts"

    # 4) 运行 featureCounts
    cmd = [
        featurecounts_bin,
        "-T", str(threads),
        "-t", "exon",
        "-g", "gene_id",
        "-a", str(gtf_path),
        "-o", str(out_txt),
    ] + bam_files
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

    # 读取输出
    df = pd.read_csv(out_txt, sep="\t", comment="#")

    # 若 simple=True，仅保留 Geneid + counts 列
    if simple:
        gene_col = "Geneid" if "Geneid" in df.columns else df.columns[0]
        # 去除注释列，只留基因和样本计数 用于下游 OV pair的比对格式
        keep_cols = [gene_col] + [
            c for c in df.columns
            if c not in ["Chr", "Start", "End", "Strand", "Length"] and c != gene_col
        ]
        df = df[keep_cols]
        print(f"[INFO] Simplified output: retained {len(keep_cols)} columns")

    # 导出 CSV 或 TXT
    if output_csv:
        out_csv = out_txt.with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] featureCounts output → {out_csv}")
        return out_csv.resolve()
    else:
        df.to_csv(out_txt, sep="\t", index=False)
        print(f"[OK] featureCounts output → {out_txt}")
        return out_txt.resolve()