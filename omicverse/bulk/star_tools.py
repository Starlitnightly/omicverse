from __future__ import annotations
import os, time,re, sys, gzip, shutil, subprocess, json, requests
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Union
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from .geo_meta_fetcher import parse_geo_soft_to_struct, fetch_geo_text



# ================= 获GEO取数据信息  ================= 

HEADERS = {"User-Agent": "Mozilla/5.0 (Python requests)"}

def fetch_geo_text(accession: str) -> str:
    """
    accession: GSE12345 or GSMxxxx. (For SRR, you typically have no GEO record; pass the GSE.)
    """
    
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&form=text&view=full"
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text
# ================= 检测数据来源  ================= 

def detect_species_build_from_geo(accession: str) -> Tuple[str, str]:
    """
    Returns (species_key, build_key), e.g. ('human', 'GRCh38')
    - If build is not stated, choose a robust default by species.
    """
    txt = fetch_geo_text(accession)
    # try to find organism lines
    org_lines = re.findall(r"!(?:Series|Sample|Platform)_(?:sample_)?organism.*=\s*(.+)", txt)
    org = "|".join(set([o.strip() for o in org_lines])) if org_lines else ""

    # normalize organism
    org_lower = org.lower()
    if "homo sapiens" in org_lower or "human" in org_lower:
        species, build = "human", "GRCh38"
    elif "mus musculus" in org_lower or "mouse" in org_lower:
        species, build = "mouse", "GRCm39"
    else:
        # default fallbacks; customize as needed
        species, build = "human", "GRCh38"

    # Try to detect genome build strings in free text (rare in GEO)
    build_hits = re.findall(r"(GRCh\d+|hg\d+|GRCm\d+|mm\d+)", txt, flags=re.I)
    if build_hits:
        # pick the first reasonable hit
        b = build_hits[0].upper()
        # normalize a few common aliases
        if b.startswith("HG"):
            # hg38->GRCh38, hg19->GRCh37
            build = "GRCh38" if "38" in b else ("GRCh37" if "19" in b else build)
        elif b.startswith("MM"):
            build = "GRCm39" if "39" in b else ("GRCm38" if "10" in b else build)
        else:
            build = b
    return species, build

# ================= ·download helper ================= 
'''
def download_file(url, dest, chunk=1 << 20, max_retries=5):
    """
    Robust file downloader with HTTP Range resume support.
    - If an incomplete .part file exists, it will resume from where it stopped.
    - Retries on connection drop (ChunkedEncodingError, Timeout, etc.).
    """
    headers = {
    "User-Agent": "Python-Resumable-Downloader/1.0",
    "Accept-Encoding": "identity",   # 避免服务器对主体再 gzip，保证 Range 的字节对齐
    }
    dest = os.path.abspath(dest)
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Figure out where to resume
    resume_byte_pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    mode = "ab" if resume_byte_pos > 0 else "wb"

    
    if resume_byte_pos > 0:
        headers["Range"] = f"bytes={resume_byte_pos}-"

    # Do streaming with retries
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=600) as r:
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) + resume_byte_pos
                pbar = tqdm(
                    total=total, initial=resume_byte_pos, unit="B", unit_scale=True,
                    desc=os.path.basename(dest)
                )

                with open(tmp, mode) as f:
                    for chunk_bytes in r.iter_content(chunk_size=chunk):
                        if not chunk_bytes:
                            continue
                        f.write(chunk_bytes)
                        pbar.update(len(chunk_bytes))
                pbar.close()

            # Rename only if full download succeeded
            os.replace(tmp, dest)
            print(f"[OK] {os.path.basename(dest)} downloaded successfully ({total/1e6:.1f} MB)")
            return dest

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
            print("   Retrying in 10 s …")
            import time; time.sleep(10)
            # next iteration will resume again
            continue

    raise RuntimeError(f"❌ Failed to download {url} after {max_retries} attempts")
'''
def gunzip_to(src_gz: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rb") as gzin, open(dst, "wb") as out:
        shutil.copyfileobj(gzin, out)

class ResumeError(RuntimeError): pass

def head_info(url, timeout=30):
    r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=timeout)
    r.raise_for_status()
    h = r.headers
    return {
        "length": int(h.get("Content-Length") or 0),
        "accept_ranges": (h.get("Accept-Ranges") or "").lower() == "bytes",
        "etag": h.get("ETag"),
        "last_modified": h.get("Last-Modified"),
        "final_url": r.url,
    }

def curl_available():
    return shutil.which("curl") is not None

def download_file(url: str, dest: str, chunk: int = 1<<20, max_retries: int = 8, use_curl_fallback: bool = True):
    """
    具备断点续传、元信息校验、重试与可选 curl 回退的稳健下载器。
    - 部分下载写到 dest.part，元信息写到 dest.meta.json
    - 支持 Range 续传（206）。若服务端忽略 Range（200），可选走 curl -C - 回退。
    """
    dest = os.path.abspath(dest)
    tmp = dest + ".part"
    meta = dest + ".meta.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # 1) 获取服务器元信息
    info = head_info(url)
    total = info["length"]
    supports_range = info["accept_ranges"]
    etag = info["etag"]
    last_mod = info["last_modified"]

    # 2) 读取本地元信息
    local = {"downloaded": 0, "total": total, "etag": etag, "last_modified": last_mod}
    if os.path.exists(meta):
        try:
            local = json.load(open(meta, "r"))
        except Exception:
            pass

    # 如果服务器文件特征（etag/last-mod/总长）与本地不一致，删除旧片段
    mismatch = (
        (local.get("etag") and etag and local["etag"] != etag) or
        (local.get("last_modified") and last_mod and local["last_modified"] != last_mod) or
        (local.get("total", 0) and total and local["total"] != total)
    )
    if mismatch:
        for p in (tmp, dest, meta):
            if os.path.exists(p):
                os.remove(p)
        local = {"downloaded": 0, "total": total, "etag": etag, "last_modified": last_mod}

    # 若已有完整文件，直接返回
    if os.path.exists(dest) and (total == 0 or os.path.getsize(dest) == total):
        return dest

    # 初始化/刷新 meta
    local["total"] = total
    local["etag"] = etag
    local["last_modified"] = last_mod
    json.dump(local, open(meta, "w"))

    # 当前已下载字节
    downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if downloaded > total > 0:
        # 异常状态，清理重新来
        os.remove(tmp)
        downloaded = 0

    # 3) 如果不支持 Range，且需要续传，考虑 curl 回退或重下
    if downloaded > 0 and not supports_range:
        if use_curl_fallback and curl_available():
            print("⚠️ Server does not advertise Range support. Falling back to curl -C - …")
            return curl_resume(url, dest, meta)
        else:
            print("⚠️ Server does not support resume; restarting from scratch.")
            if os.path.exists(tmp):
                os.remove(tmp)
            downloaded = 0

    # 4) 带 Range 的下载循环
    headers = dict(HEADERS)
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                # 期望 206（续传）；如果返回 200 且我们想续传 → 服务器忽略 Range
                if downloaded > 0 and r.status_code == 200:
                    if use_curl_fallback and curl_available():
                        print("⚠️ Server ignored Range (200). Falling back to curl -C - …")
                        return curl_resume(url, dest, meta)
                    else:
                        print("⚠️ Server ignored Range; restarting from scratch.")
                        if os.path.exists(tmp): os.remove(tmp)
                        downloaded = 0

                r.raise_for_status()

                # 进度条
                total_disp = total if total else None
                pbar = tqdm(total=total_disp, initial=downloaded, unit="B", unit_scale=True,
                            desc=os.path.basename(dest))

                # 以追加或写入模式打开
                with open(tmp, "ab" if downloaded else "wb") as f:
                    for chunk_bytes in r.iter_content(chunk_size=chunk):
                        if not chunk_bytes:
                            continue
                        f.write(chunk_bytes)
                        downloaded += len(chunk_bytes)
                        pbar.update(len(chunk_bytes))
                        # 每写一段就刷新 meta（防止崩溃后能继续）
                        local["downloaded"] = downloaded
                        json.dump(local, open(meta, "w"))
                pbar.close()

            # 完成：原子替换
            os.replace(tmp, dest)
            print(f"[OK] {os.path.basename(dest)} downloaded ({downloaded/1e6:.1f} MB)")
            return dest

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(min(30, 5 * attempt))  # 逐步退避

    raise ResumeError(f"Failed to download after {max_retries} attempts: {url}")

def curl_resume(url: str, dest: str, meta_path: str):
    """使用 curl -L -C - 断点续传作为回退方案。"""
    tmp = dest + ".part"
    # curl 直接对最终文件续传，因此我们对齐逻辑：先把 .part 挪成最终名（若存在）
    if os.path.exists(tmp) and not os.path.exists(dest):
        os.rename(tmp, dest)

    cmd = ["curl", "-L", "-C", "-", "-o", dest, url]
    print(">>", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise ResumeError("curl failed to resume download.")
    # 校验尺寸并补 meta
    try:
        info = head_info(url)
        meta = {"downloaded": os.path.getsize(dest), "total": info["length"],
                "etag": info["etag"], "last_modified": info["last_modified"]}
        json.dump(meta, open(meta_path, "w"))
    except Exception:
        pass
    print(f"[OK] {os.path.basename(dest)} downloaded via curl")
    return dest

# ================= ·检测STAR INDEX是否存在 ================= 
def get_gtf_for_index(index_dir: Path) -> Path:
    """
    根据 STAR index 目录自动定位 GTF：
    1) 若存在 reference.json（建议在 ensure_star_index 构建索引时写入），直接读取；
    2) 按约定路径扫描：index/<species>/<build>/STAR → 缓存在 index/_cache/<species>/<build>/*.gtf(.gz)
    3) 兜底：在 index 根目录向下递归找最近的 .gtf 或 .gtf.gz

    返回：解压后的 .gtf 的绝对路径（若原文件是 .gtf.gz，会自动解压并返回解压路径）
    """
    index_dir = Path(index_dir).resolve()
    # 1) 优先用 reference.json（如果你在 ensure_star_index 里写过这个档案）
    ref_json = index_dir.parent / "reference.json"
    if ref_json.exists():
        try:
            meta = json.loads(ref_json.read_text())
            gtf = Path(meta.get("gtf", ""))
            if gtf.exists():
                if gtf.suffix == ".gz":
                    # 解压到同级无后缀路径（可按需改到 cache）
                    dst = gtf.with_suffix("") 
                    if not dst.exists():
                        _gunzip_to(gtf, dst)
                    return dst.resolve()
                return gtf.resolve()
        except Exception:
            pass

    # 2) 依约定推导 cache 目录：index/<species>/<build>/STAR
    #    → index/_cache/<species>/<build>
    #    index_dir.parts: [..., "index", species, build, "STAR"]
    parts = index_dir.parts
    # 尝试定位 "index" 的位置
    try:
        i = parts.index("index")
        species = parts[i+1]
        build   = parts[i+2]
        cache_dir = Path(*parts[:i+1]) / "_cache" / species / build
    except Exception:
        # 若不符合约定结构，fallback 到 index 的三层上级再去找 _cache
        cache_dir = index_dir
        for _ in range(3):
            if cache_dir.name == "STAR":
                cache_dir = cache_dir.parent.parent / "_cache" / cache_dir.parent.name / cache_dir.parent.parent.name
                break
            cache_dir = cache_dir.parent

    gtf = None
    # 2a) cache 下找 .gtf or .gtf.gz
    if cache_dir and cache_dir.exists():
        gz_list  = sorted(cache_dir.glob("*.gtf.gz"))
        gtf_list = sorted(cache_dir.glob("*.gtf"))
        if gtf_list:
            gtf = gtf_list[0]
        elif gz_list:
            gtf_gz = gz_list[0]
            dst = cache_dir / re.sub(r"\.gz$", "", gtf_gz.name)
            if not dst.exists():
                _gunzip_to(gtf_gz, dst)
            gtf = dst

    # 3) 兜底：在 index 根目录附近递归找
    if gtf is None:
        root = index_dir
        for _ in range(4):  # 往上最多四层
            root = root.parent
        candidates = list(root.rglob("*.gtf")) or list(root.rglob("*.gtf.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"Cannot locate GTF for STAR index at: {index_dir}\n"
                f"Please ensure ensure_star_index() downloaded GENCODE and cache exists under index/_cache/<species>/<build>/"
            )
        gtf_cand = candidates[0]
        if str(gtf_cand).endswith(".gtf.gz"):
            dst = gtf_cand.with_suffix("")
            if not dst.exists():
                _gunzip_to(gtf_cand, dst)
            gtf = dst
        else:
            gtf = gtf_cand

    return Path(gtf).resolve()
def ensure_star_index(
    species: str,
    build: str,
    index_root: Path = Path("index"),
    gencode_release: str = "v44",
    threads: int = 12,
    sjdb_overhang: Optional[int] = 149,) -> Path:
    """
    Ensures STAR index exists for (species, build).
    Downloads GENCODE genome+GTF, builds index if missing.
    Returns the STAR index dir path.
    """
    # Normalize
    species = species.lower()
    build = build.upper()
    
    # normalize common aliases 确保名称大小写规范
    if build in ("GRCH38", "HG38"):
        build = "GRCh38"
    elif build in ("GRCH37", "HG19"):
        build = "GRCh37"
    elif build in ("GRCM39", "MM39"):
        build = "GRCm39"
    elif build in ("GRCM38", "MM10"):
        build = "GRCm38"

    # Where to place the index
    idx_dir = index_root / species / build / "STAR"
    genome_params = idx_dir / "genomeParameters.txt"
    if genome_params.exists():
        return idx_dir

    # Choose GENCODE URLs (adjust if you prefer Ensembl / iGenomes)
    if species == "human" and build == "GRCh38":
        # GENCODE primary assembly (no ALT contigs)
        base = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{gencode_release.lstrip('v')}"
        fasta_url = f"{base}/GRCh38.primary_assembly.genome.fa.gz"
        gtf_url   = f"{base}/gencode.{gencode_release}.annotation.gtf.gz"
    elif species == "mouse" and build in ("GRCm39","GRCM39"):
        base = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_{gencode_release.lstrip('v')}"
        fasta_url = f"{base}/GRCm39.primary_assembly.genome.fa.gz"
        gtf_url   = f"{base}/gencode.{gencode_release}.annotation.gtf.gz"
    else:
        raise ValueError(f"No URL templates for species={species}, build={build}. Add one above.")

    # Download to a cache folder
    cache = index_root / "_cache" / species / build
    fa_gz = cache / Path(fasta_url).name
    gtf_gz = cache / Path(gtf_url).name
    fa = cache / fa_gz.with_suffix("").name  # .fa
    gtf = cache / gtf_gz.with_suffix("").name  # .gtf

    print(f"[INFO] Preparing reference for {species}/{build} (GENCODE {gencode_release})")
    if not fa_gz.exists():
        download_file(fasta_url, fa_gz)
    if not gtf_gz.exists():
        download_file(gtf_url, gtf_gz)

    if not fa.exists():
        print(f"[INFO] Decompressing {fa_gz.name} → {fa.name}")
        gunzip_to(fa_gz, fa)
    if not gtf.exists():
        print(f"[INFO] Decompressing {gtf_gz.name} → {gtf.name}")
        gunzip_to(gtf_gz, gtf)

    # Build STAR index
    idx_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "STAR",
        "--runThreadN", str(threads),
        "--runMode", "genomeGenerate",
        "--genomeDir", str(idx_dir),
        "--genomeFastaFiles", str(fa),
        "--sjdbGTFfile", str(gtf),
    ]
    if sjdb_overhang is not None:
        cmd += ["--sjdbOverhang", str(sjdb_overhang)]  # readLength-1 (150bp reads → 149)

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not genome_params.exists():
        raise RuntimeError(f"STAR index build completed but {genome_params} not found.")

    ref_json = idx_dir.parent / "reference.json"  # index/<species>/<build>/reference.json
    try:
        ref_json.write_text(json.dumps({
            "species": species,
            "build": build,
            "gencode_release": gencode_release,
            "fasta": str(fa.resolve()),
            "gtf":   str(gtf.resolve()),
            "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {"fasta_url": fasta_url, "gtf_url": gtf_url}
        }, indent=2))
        print(f"[INFO] Reference metadata written → {ref_json}")
    except Exception as e:
        print(f"[WARN] Could not write reference.json: {e}")
    return idx_dir

# ================= RUN STAR =======================
'''
def run_star(
    fq1: Path,
    fq2: Optional[Path],
    index_dir: Path,
    out_prefix: Path,
    threads: int = 12,
    extra_args: Optional[list] = None,
):
    if not (index_dir / "genomeParameters.txt").exists():
        raise RuntimeError(f"STAR index incomplete: {index_dir}")

    cmd = [
        "STAR",
        "--runThreadN", str(threads),
        "--genomeDir", str(index_dir),
        "--readFilesIn", str(fq1)
    ]
    if fq2:
        cmd += [str(fq2)]

    # macOS uses 'gunzip -c' (or 'gzcat'); use gunzip universally for .gz
    if str(fq1).endswith(".gz") or (fq2 and str(fq2).endswith(".gz")):
        cmd += ["--readFilesCommand", "gunzip", "-c"]

    cmd += ["--outSAMtype", "BAM", "SortedByCoordinate",
            "--outFileNamePrefix", str(out_prefix)]
    if extra_args:
        cmd += list(extra_args)

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def star_align_auto(
    accession: str,
    fq1: str,
    fq2: Optional[str],
    index_root: str = "index",
    out_prefix: str = "work/star/sample.",
    threads: int = 12,
    gencode_release: str = "v44",
    sjdb_overhang: Optional[int] = 149,
):
    species, build = detect_species_build_from_geo(accession)
    print(f"[INFO] Detected: species={species}, build={build}")

    idx = ensure_star_index(
        species, build,
        index_root=Path(index_root),
        gencode_release=gencode_release,
        threads=threads,
        sjdb_overhang=sjdb_overhang
    )

    run_star(
        fq1=Path(fq1),
        fq2=Path(fq2) if fq2 else None,
        index_dir=idx,
        out_prefix=Path(out_prefix),
        threads=threads
    )
    print("[OK] STAR alignment finished.")
'''
def _infer_sample_id_from_fq(fq1: Path) -> str:
    s = fq1.name
    m = re.search(r"(SRR\d+)", s, re.I)
    if m:
        return m.group(1)
    # 常见命名 S1_1.fastq / S1.R1.fq.gz / S1.clean.fq.gz
    s = re.sub(r"\.f(ast)?q(\.gz)?$", "", s)
    s = re.sub(r"(_|\.)(R?1|R?2|clean)$", "", s, flags=re.I)
    return s

def run_star(
    fq1: Path,
    fq2: Optional[Path],
    index_dir: Path,
    out_prefix: Path,
    threads: int = 12,
    extra_args: Optional[List[str]] = None,
    sample: Optional[str] = None,
) -> Path:
    """
    运行 STAR，比对结果写入：<base>/<sample>/<sample>.* 并返回 BAM 路径。
    - out_prefix：可以传你原来的 "work/star/sample."（我们会基于它构造真正的 per-sample 前缀）
    - sample：可显式传 SRR；不传则从 fq1 文件名自动推断
    """
    # 检查是否已有结果

    sample = sample or _infer_sample_id_from_fq(fq1)
    # 以 out_prefix 的父目录为 base，例如 "work/star/sample." -> base="work/star"
    base_dir = out_prefix.parent
    out_dir = base_dir / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    star_prefix = out_dir / f"{sample}."  # <sample>.*
    bam_path = out_dir / f"{sample or out_prefix.stem}.Aligned.sortedByCoord.out.bam"
    if bam_path.exists() and bam_path.stat().st_size > 1024 * 1024:  # >1MB
        print(f"[SKIP] Detected existing alignment → {bam_path}")
        return bam_path

    if not (index_dir / "genomeParameters.txt").exists():
        raise RuntimeError(f"STAR index incomplete: {index_dir}")

    cmd = [
        "STAR",
        "--runThreadN", str(threads),
        "--genomeDir", str(index_dir),
        "--readFilesIn", str(fq1)
    ]
    if fq2:
        cmd += [str(fq2)]

    # 通用 .gz 解压（macOS 用 gunzip -c）
    if str(fq1).endswith(".gz") or (fq2 and str(fq2).endswith(".gz")):
        cmd += ["--readFilesCommand", "gunzip", "-c"]

    cmd += [
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFileNamePrefix", str(star_prefix),
        "--outTmpDir", str(out_dir / "tmp"),
        "--outSAMattrRGline", f"ID:{sample}", f"SM:{sample}", "PL:ILLUMINA"
    ]
    if extra_args:
        cmd += list(extra_args)

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    bam = star_prefix.with_name(f"{sample}.Aligned.sortedByCoord.out.bam")
    return bam

def star_align_auto(
    accession: str,
    fq1: str,
    fq2: Optional[str],
    index_root: str = "index",
    out_prefix: str = "work/star/sample.",  # 基底目录仍沿用这个写法
    threads: int = 12,
    gencode_release: str = "v44",
    sjdb_overhang: Optional[int] = 149,
    sample: Optional[str] = None,           # 新增：可显式传 SRR
) -> Path:
    species, build = detect_species_build_from_geo(accession)
    print(f"[INFO] Detected: species={species}, build={build}")

    idx = ensure_star_index(
        species, build,
        index_root=Path(index_root),
        gencode_release=gencode_release,
        threads=threads,
        sjdb_overhang=sjdb_overhang
    )

    bam = run_star(
        fq1=Path(fq1),
        fq2=Path(fq2) if fq2 else None,
        index_dir=idx,
        out_prefix=Path(out_prefix),
        threads=threads,
        sample=sample  # 传递 SRR（不传则自动从 fq1 推断）
    )
    print(f"[OK] STAR alignment finished. BAM -> {bam}")
    return bam, idx

def _star_align_one(
    accession: str,
    fq1: str,
    fq2: str,
    index_dir: Path,
    out_prefix: str,
    threads: int,
    sample: str,
):
    """
    执行单个 STAR 对齐任务。
    """
    os.makedirs(Path(out_prefix).parent, exist_ok=True)
    bam_out = str(Path(out_prefix).with_suffix(".Aligned.sortedByCoord.out.bam"))

    if os.path.exists(bam_out) and os.path.getsize(bam_out) > 0:
        return sample, bam_out

    cmd = [
        "STAR",
        "--runThreadN", str(threads),
        "--genomeDir", str(index_dir),
        "--readFilesIn", fq1, fq2,
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFileNamePrefix", out_prefix,
        "--readFilesCommand", "zcat" if fq1.endswith(".gz") else "cat",
        "--outSAMunmapped", "Within"
    ]
    subprocess.run(cmd, check=True)
    return sample, bam_out


def star_align_auto_parallel(
    samples: list[tuple[str, str, str]],  # [(srr, fq1, fq2)]
    accession: Optional[str],
    index_root: str,
    out_root: str,
    threads: int = 12,
    gencode_release: str = "v44",
    sjdb_overhang: int = 149,
    max_workers: Optional[int] = None
):
    """
    并行运行 STAR 对齐。
    """
    species, build = detect_species_build_from_geo(accession or samples[0][0])

    index_dir = ensure_star_index(species, build, gencode_release, index_root, threads)
    os.makedirs(out_root, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers or min(8, os.cpu_count() // threads)) as ex:
        futures = {}
        for srr, fq1, fq2 in samples:
            out_prefix = Path(out_root) / srr / "run."
            out_prefix.parent.mkdir(parents=True, exist_ok=True)
            fut = ex.submit(_star_align_one, accession, fq1, fq2, index_dir, str(out_prefix), threads, srr)
            futures[fut] = srr

        for fut in as_completed(futures):
            srr = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[ERR] {srr}: {e}")

    return results