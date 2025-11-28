from __future__ import annotations
import os, time,re, sys, gzip, shutil, subprocess, json, requests
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Union
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from .geo_meta_fetcher import parse_geo_soft_to_struct, fetch_geo_text

# ================= Fetch GEO metadata =================

HEADERS = {"User-Agent": "Mozilla/5.0 (Python requests)"}

def fetch_geo_text(accession: str) -> str:
    """
    accession: GSE12345 or GSMxxxx. (For SRR, you typically have no GEO record; pass the GSE.)
    """
    
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&form=text&view=full"
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text
# ================= Detect data origin =================

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

# ================= Download helper =================
'''
def download_file(url, dest, chunk=1 << 20, max_retries=5):
    """
    Robust file downloader with HTTP Range resume support.
    - If an incomplete .part file exists, it will resume from where it stopped.
    - Retries on connection drop (ChunkedEncodingError, Timeout, etc.).
    """
    headers = {
    "User-Agent": "Python-Resumable-Downloader/1.0",
    "Accept-Encoding": "identity",   # Avoid extra gzip so byte ranges remain aligned.
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
    Robust downloader with resume support, metadata validation, retries, and optional curl fallback.
    - Partial downloads are written to dest.part, with metadata stored in dest.meta.json.
    - Supports HTTP Range resumes (206). If the server ignores Range (200), optionally fall back to curl -C -.
    """
    dest = os.path.abspath(dest)
    tmp = dest + ".part"
    meta = dest + ".meta.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # 1) Fetch server metadata.
    info = head_info(url)
    total = info["length"]
    supports_range = info["accept_ranges"]
    etag = info["etag"]
    last_mod = info["last_modified"]

    # 2) Load local metadata.
    local = {"downloaded": 0, "total": total, "etag": etag, "last_modified": last_mod}
    if os.path.exists(meta):
        try:
            local = json.load(open(meta, "r"))
        except Exception:
            pass

    # Remove stale fragments when server metadata (etag/last-mod/size) differs from local records.
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

    # Return immediately if the destination file is already complete.
    if os.path.exists(dest) and (total == 0 or os.path.getsize(dest) == total):
        return dest

    # Initialize or refresh the metadata file.
    local["total"] = total
    local["etag"] = etag
    local["last_modified"] = last_mod
    json.dump(local, open(meta, "w"))

    # Track how many bytes have been downloaded.
    downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if downloaded > total > 0:
        # Inconsistent state detected; reset and start over.
        os.remove(tmp)
        downloaded = 0

    # 3) If Range is unsupported but resume is required, consider curl fallback or restarting the download.
    if downloaded > 0 and not supports_range:
        if use_curl_fallback and curl_available():
            print("⚠️ Server does not advertise Range support. Falling back to curl -C - …")
            return curl_resume(url, dest, meta)
        else:
            print("⚠️ Server does not support resume; restarting from scratch.")
            if os.path.exists(tmp):
                os.remove(tmp)
            downloaded = 0

    # 4) Download loop using HTTP Range.
    headers = dict(HEADERS)
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                # Expect 206 for resume; if 200 arrives while resuming, the server ignored Range.
                if downloaded > 0 and r.status_code == 200:
                    if use_curl_fallback and curl_available():
                        print("⚠️ Server ignored Range (200). Falling back to curl -C - …")
                        return curl_resume(url, dest, meta)
                    else:
                        print("⚠️ Server ignored Range; restarting from scratch.")
                        if os.path.exists(tmp): os.remove(tmp)
                        downloaded = 0

                r.raise_for_status()

                # Progress bar display.
                total_disp = total if total else None
                pbar = tqdm(total=total_disp, initial=downloaded, unit="B", unit_scale=True,
                            desc=os.path.basename(dest))

                # Open in append or write mode as appropriate.
                with open(tmp, "ab" if downloaded else "wb") as f:
                    for chunk_bytes in r.iter_content(chunk_size=chunk):
                        if not chunk_bytes:
                            continue
                        f.write(chunk_bytes)
                        downloaded += len(chunk_bytes)
                        pbar.update(len(chunk_bytes))
                        # Refresh metadata after each chunk to support safe resumption.
                        local["downloaded"] = downloaded
                        json.dump(local, open(meta, "w"))
                pbar.close()

            # Atomically swap the temporary file into place once complete.
            os.replace(tmp, dest)
            print(f"[OK] {os.path.basename(dest)} downloaded ({downloaded/1e6:.1f} MB)")
            return dest

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(min(30, 5 * attempt))  # Exponential backoff capped at 30 seconds.

    raise ResumeError(f"Failed to download after {max_retries} attempts: {url}")

def curl_resume(url: str, dest: str, meta_path: str):
    """Fallback resume helper using curl -L -C -."""
    tmp = dest + ".part"
    # curl resumes directly onto the destination file; rename any .part beforehand.
    if os.path.exists(tmp) and not os.path.exists(dest):
        os.rename(tmp, dest)

    cmd = ["curl", "-L", "-C", "-", "-o", dest, url]
    print(">>", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise ResumeError("curl failed to resume download.")
    # Validate file size and refresh metadata.
    try:
        info = head_info(url)
        meta = {"downloaded": os.path.getsize(dest), "total": info["length"],
                "etag": info["etag"], "last_modified": info["last_modified"]}
        json.dump(meta, open(meta_path, "w"))
    except Exception:
        pass
    print(f"[OK] {os.path.basename(dest)} downloaded via curl")
    return dest

# ================= Detect STAR index assets =================
def get_gtf_for_index(index_dir: Path) -> Path:
    """
    Locate the GTF associated with a STAR index directory:
    1) If reference.json exists (recommended when ensure_star_index writes it), read it directly.
    2) Scan the conventional layout: index/<species>/<build>/STAR with cached copies under index/_cache/<species>/<build>/*.gtf(.gz).
    3) As a fallback, recursively search from the index root for the nearest .gtf or .gtf.gz.

    Returns the absolute path to the extracted .gtf (gzipped files are decompressed automatically).
    """
    index_dir = Path(index_dir).resolve()
    # 1) Prefer reference.json when ensure_star_index generated it.
    ref_json = index_dir.parent / "reference.json"
    if ref_json.exists():
        try:
            meta = json.loads(ref_json.read_text())
            gtf = Path(meta.get("gtf", ""))
            if gtf.exists():
                if gtf.suffix == ".gz":
                    # Decompress alongside the original (adjust to cache if desired).
                    dst = gtf.with_suffix("") 
                    if not dst.exists():
                        _gunzip_to(gtf, dst)
                    return dst.resolve()
                return gtf.resolve()
        except Exception:
            pass

    # 2) Derive the cache directory based on the conventional layout.
    #    index_dir.parts: [..., "index", species, build, "STAR"]
    parts = index_dir.parts
    # Locate the "index" segment within the path.
    try:
        i = parts.index("index")
        species = parts[i+1]
        build   = parts[i+2]
        cache_dir = Path(*parts[:i+1]) / "_cache" / species / build
    except Exception:
        # If the structure differs, fall back to searching several levels above for _cache.
        cache_dir = index_dir
        for _ in range(3):
            if cache_dir.name == "STAR":
                cache_dir = cache_dir.parent.parent / "_cache" / cache_dir.parent.name / cache_dir.parent.parent.name
                break
            cache_dir = cache_dir.parent

    gtf = None
    # 2a) Search for .gtf or .gtf.gz within the cache directory.
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

    # 3) Final fallback: recursively search near the index root.
    if gtf is None:
        root = index_dir
        for _ in range(4):  # Search up to four levels upward.
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
    
    # Normalize common aliases and enforce consistent casing.
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
    # Handle common naming patterns like S1_1.fastq / S1.R1.fq.gz / S1.clean.fq.gz.
    s = re.sub(r"\.f(ast)?q(\.gz)?$", "", s)
    s = re.sub(r"(_|\.)(R?1|R?2|clean)$", "", s, flags=re.I)
    return s

def _parse_memory_limit_to_bytes(value: Union[str, int, float]) -> int:
    """
    Convert human-readable memory like '48G' or '800M' into integer bytes.
    STAR expects --limitBAMsortRAM as a byte count; passing '100G' directly
    would otherwise be interpreted incorrectly and can trigger false
    'not enough memory for BAM sorting' errors.
    """
    if isinstance(value, (int, float)):
        return int(value)

    s = str(value).strip().upper()
    # Accept bare integers (already bytes).
    if s.isdigit():
        return int(s)

    import re
    m = re.match(r"^(\d+)([KMG])B?$", s)
    if not m:
        # Fallback: keep numeric portion or 0.
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 0

    num = int(m.group(1))
    unit = m.group(2)
    factor = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return num * factor


def run_star(
    fq1: Path,
    fq2: Optional[Path],
    index_dir: Path,
    out_prefix: Path,
    threads: int = 12,
    memory_limit: Union[str, int, float] = "100G",  # BAM sorting memory limit
    extra_args: Optional[List[str]] = None,
    sample: Optional[str] = None,
) -> Path:
    """
    Run STAR, writing outputs to <base>/<sample>/<sample>.* and return the BAM path.
    - out_prefix: accept the legacy "work/star/sample." prefix (used to derive the per-sample prefix).
    - sample: optionally pass an SRR; otherwise infer from fq1.
    - memory_limit: memory limit for BAM sorting (e.g., "100G", "85899345920" for bytes)
    """
    # Skip when results already exist.

    sample = sample or _infer_sample_id_from_fq(fq1)
    # Use out_prefix's parent as the base directory (e.g., "work/star/sample." → base="work/star").
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

    # Generic .gz decompression (macOS uses gunzip -c).
    if str(fq1).endswith(".gz") or (fq2 and str(fq2).endswith(".gz")):
        cmd += ["--readFilesCommand", "gunzip", "-c"]

    # Ensure memory_limit is passed as a byte count, not "100G".
    mem_bytes = _parse_memory_limit_to_bytes(memory_limit)

    cmd += [
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFileNamePrefix", str(star_prefix),
        "--outTmpDir", str(out_dir / "tmp"),
        "--outSAMattrRGline", f"ID:{sample}", f"SM:{sample}", "PL:ILLUMINA",
        "--limitBAMsortRAM", str(mem_bytes),
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
    out_prefix: str = "work/star/sample.",  # Base directory naming remains consistent with prior usage.
    threads: int = 12,
    gencode_release: str = "v44",
    sjdb_overhang: Optional[int] = 149,
    sample: Optional[str] = None,           # Optional explicit SRR value.
    memory_limit: str = "100G",             # BAM sorting memory limit
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
        memory_limit=memory_limit,  # Add memory limit for BAM sorting
        sample=sample  # Propagate the SRR (infer from fq1 when not provided).
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
    Execute a single STAR alignment task.
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
    Run STAR alignments in parallel.
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
