# Author: Zhi Luo

import os, json, csv, time, shutil, subprocess
from pathlib import Path
from typing import Optional, List
import pandas as pd

# -------------------- EDirect helpers --------------------
class RunInfoError(RuntimeError): ...
def _which(cmd: str) -> Optional[str]:
    """Locate the edirect executable; prefer PATH, then fall back to ~/edirect/."""
    p = shutil.which(cmd)
    if p: return p
    home = os.path.expanduser("~")
    cand = os.path.join(home, "edirect", cmd)
    return cand if os.path.exists(cand) else None

def _run_edirect(term: str, timeout: int = 240) -> str:
    """Invoke esearch|efetch and return the runinfo CSV text; raise when it fails."""
    esearch = _which("esearch"); efetch = _which("efetch")
    if not esearch or not efetch:
        raise RunInfoError("Entrez Direct not found (esearch/efetch). Add $HOME/edirect to PATH.")
    # Use bash -lc to support pipelines and wrap term with quotes.
    cmd = f'''{esearch} -db sra -query "{term}" | {efetch} -format runinfo'''
    try:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunInfoError(f"EDirect timeout for {term} after {timeout}s")
    except Exception as e:
        raise RunInfoError(f"EDirect execution failed for {term}: {str(e)}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout)[:400]
        raise RunInfoError(f"EDirect failed for {term}: {err}")
    return proc.stdout

def _ok_csv(text: str) -> bool:
    """Return True when the CSV contains at least two rows and a Run column."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2: return False
    header = next(csv.reader([lines[0]]))
    return any(h.strip().lower() == "run" for h in header)

# -------------------- JSON meta parsing --------------------
def _load_meta(accession: str, meta_dir: str | Path = "meta") -> dict:
    p = Path(meta_dir) / f"{accession}_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"Meta JSON not found: {p}. Generate it first with geo_accession_to_meta_json().")
    return json.loads(p.read_text(encoding="utf-8"))

def _candidates_from_meta(meta: dict) -> List[str]:
    """
    Prefer PRJNA candidates (BioProject_all); otherwise fall back to SRP (SRAnum_all).
    If both are missing, fall back to single-value entries in extracted.
    """
    ex = meta.get("extracted", {})
    prjna_all = ex.get("BioProject_all") or []
    srp_all   = ex.get("SRAnum_all") or []
    # Handle cases where only single values exist.
    if ex.get("BioProject") and ex["BioProject"] not in prjna_all:
        prjna_all = [ex["BioProject"]] + prjna_all
    if ex.get("SRAnum") and ex["SRAnum"] not in srp_all:
        srp_all = [ex["SRAnum"]] + srp_all
    # Deduplicate while keeping the original order.
    seen, out = set(), []
    for x in prjna_all + srp_all:
        u = x.upper()
        if u not in seen:
            seen.add(u); out.append(u)
    return out

# -------------------- Main flow: GSE→load JSON→PRJNA/SRP→runinfo.csv --------------------
def gse_meta_to_runinfo_csv(
    accession: str,
    meta_dir: str | Path = "meta",
    out_dir: str | Path = "meta",
    retries: int = 3,
    backoff: float = 2.0,
    organism_filter: Optional[str] = None,  # e.g., "Homo sapiens"
    layout_filter: Optional[str] = None,    # "PAIRED" / "SINGLE"
) -> dict:
    """
    Read {meta_dir}/{GSE}_meta.json, extract PRJNA (or SRP) terms, and try them sequentially:
      esearch -db sra -query <TERM> | efetch -format runinfo
    On success, save {out_dir}/{GSE}_runinfo.csv and return the result summary.
    """
    gse = accession.strip().upper()
    meta = _load_meta(gse, meta_dir=meta_dir)
    terms = _candidates_from_meta(meta)
    if not terms:
        return {"available": False, "csv": None, "term_used": None, "rows": 0,
                "error": "No PRJNA/SRP found in meta JSON"}

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{gse}_runinfo.csv"

    # -------- New section: skip when the output file already exists --------
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        return {
            "available": True,
            "csv": str(out_csv.resolve()),
            "term_used": "SKIPPED (already exists)",
            "rows": int(len(df))
        }
    # ------------------------------------------------

    last_err = None
    for term in terms:
        # Retry on network instability or throttling.
        for attempt in range(1, retries + 1):
            try:
                text = _run_edirect(term)
                if not _ok_csv(text):
                    raise RunInfoError(f"Empty/malformed CSV for {term} (no Run column).")
                df = pd.read_csv(pd.io.common.StringIO(text))
                if df.empty:
                    raise RunInfoError(f"No rows for {term}.")

                # Optional filtering: organism + library layout.
                if organism_filter:
                    org_cols = [c for c in df.columns if c.lower() in ("scientificname", "organism")]
                    if org_cols:
                        mask = False
                        for c in org_cols:
                            mask = mask | df[c].astype(str).str.contains(organism_filter, case=False, na=False)
                        df = df[mask]
                if layout_filter and "LibraryLayout" in df.columns:
                    df = df[df["LibraryLayout"].astype(str).str.upper() == layout_filter.upper()]

                if df.empty:
                    raise RunInfoError(f"Rows removed by filters for {term}.")

                # Persist the CSV.
                df.to_csv(out_csv, index=False)
                return {
                    "available": True,
                    "csv": str(out_csv.resolve()),
                    "term_used": term,
                    "rows": int(len(df))
                }
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff * attempt)
        # Move to the next term (e.g., multiple PRJNAs or fallback to SRP).

    return {"available": False, "csv": None, "term_used": None, "rows": 0,
            "error": repr(last_err) if last_err else "Unknown error"}

'''
How to use

# Assume you already created meta/GSE157103_meta.json via geo_accession_to_meta_json("GSE157103", out_dir="meta").

info = gse_meta_to_runinfo_csv(
    "GSE157103",
    meta_dir="meta",
    out_dir="sra_meta",
    organism_filter=None,   # or "Homo sapiens"
    layout_filter=None      # or "PAIRED" / "SINGLE"
)
print(info)
# -> {'available': True, 'csv': '/.../sra_meta/GSE157103_runinfo.csv',
#     'term_used': 'PRJNA660067', 'rows':  ...}
'''
