# Author: Zhi Luo

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Python GEO-to-SRA fetcher)"}

# ---------- 1) Fetch GEO SOFT text ----------
def fetch_geo_text(accession: str, timeout: int = 120) -> str:
    """
    Retrieve the GEO Series/GSM SOFT text view (contains the full set of Series_* fields).
    e.g. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157103&form=text&view=full
    """
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&form=text&view=full"
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------- 2) Parse into structured data ----------
PRJNA_RE = re.compile(r"\bPRJNA\d+\b", re.I)
SRP_RE   = re.compile(r"\bSRP\d+\b", re.I)
GSM_RE   = re.compile(r"\bGSM\d+\b", re.I)

def _append(d: Dict[str, Any], key: str, val: str):
    """Collect repeated keys (e.g., Series_sample_id) into a list."""
    if key not in d:
        d[key] = val
    else:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
        d[key].append(val)

def parse_geo_soft_to_struct(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse every '!Series_' key-value pair in the SOFT text and collect key fields.
    Returns (series_all, extracted)
    - series_all: complete !Series_* key-value pairs (duplicate keys become lists)
    - extracted: normalized summary of the fields of interest
    """
    series_all: Dict[str, Any] = {}

    # Parse Series_* key-value pairs line by line (e.g., !Series_platform_id = GPLxxxx).
    for ln in text.splitlines():
        if ln.startswith("!Series_"):
            # Example: !Series_sample_id = GSM123456
            parts = ln.split("=", 1)
            if len(parts) == 2:
                key = parts[0].lstrip("!").strip()   # 'Series_sample_id'
                val = parts[1].strip()
                _append(series_all, key, val)

    # Extract PRJNA/SRP from Series_relation and the entire text.
    relations = series_all.get("Series_relation", [])
    if isinstance(relations, str):
        relations = [relations]

    prjna_candidates: List[str] = []
    srp_candidates: List[str] = []

    # 1) Pull from the relation lines.
    for item in relations:
        prjna_candidates += PRJNA_RE.findall(item)
        srp_candidates   += SRP_RE.findall(item)

    # 2) Fallback: scan the full text again.
    if not prjna_candidates:
        prjna_candidates += PRJNA_RE.findall(text)
    if not srp_candidates:
        srp_candidates   += SRP_RE.findall(text)

    # Deduplicate and normalize to uppercase.
    def _uniq_upper(xs): 
        seen, out = set(), []
        for x in xs:
            x = x.upper()
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    prjna_list = _uniq_upper(prjna_candidates)
    srp_list   = _uniq_upper(srp_candidates)

    # Series accession.
    geo_acc = None
    for ln in text.splitlines():
        if ln.startswith("!Series_geo_accession"):
            # !Series_geo_accession = GSE157103
            geo_acc = ln.split("=", 1)[1].strip()
            break

    # Collect all sample GSM accessions (fallback to scanning the text when they are not listed).
    sample_ids: List[str] = []
    ssid = series_all.get("Series_sample_id", [])
    if isinstance(ssid, str):
        ssid = [ssid]
    sample_ids += ssid
    sample_ids += GSM_RE.findall(text)
    sample_ids = _uniq_upper(sample_ids)

    # Organism and taxid summary at the Series level.
    # Allow duplicates and possible multiple species; normalize to a unique list.
    organism_vals = series_all.get("Series_sample_organism", [])
    if isinstance(organism_vals, str):
        organism_vals = [organism_vals]
    organism_vals = [v.strip() for v in organism_vals]

    taxid_vals = series_all.get("Series_sample_taxid", [])
    if isinstance(taxid_vals, str):
        taxid_vals = [taxid_vals]
    taxid_vals = [v.strip() for v in taxid_vals]

    # Assemble the extracted payload.
    extracted = {
        "geo_acc": geo_acc,
        "BioProject": prjna_list[0] if prjna_list else None,
        "BioProject_all": prjna_list or [],
        "SRAnum": srp_list[0] if srp_list else None,
        "SRAnum_all": srp_list or [],
        "sample_ids": sample_ids,
        "sample_organism": sorted(list({x for x in organism_vals if x})),
        "sample_taxid": sorted(list({x for x in taxid_vals if x})),
    }

    return series_all, extracted

# ---------- 3) One-stop helper: fetch from GSE and write JSON ----------
def geo_accession_to_meta_json(accession: str, out_dir: str | Path = ".") -> Path:
    """
    For a GSE/GSM accession, fetch the SOFT text and extract:
      - geo_acc, BioProject (PRJNA), SRAnum (SRP)
      - sample_ids (list)
      - sample_organism (deduplicated list)
      - sample_taxid (deduplicated list)
    Persist all !Series_* key-values as a dictionary.
    Output: {accession}_meta.json
    """
    out_path = Path(out_dir) / f"{accession}_meta.json"
    if out_path.exists():
        print("SKIPPED (already exists)")
        return out_path
    else:
        text = fetch_geo_text(accession)
        series_all, extracted = parse_geo_soft_to_struct(text)
    
        payload = {
            "accession": accession,
            "extracted": extracted,
            "series_all": series_all  # Preserve the structured version of all information (duplicate keys merged into lists).
        }
    
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

'''
How to use
# 1) Generate the meta JSON (retains all !Series_* fields and the extracted summary).
p = geo_accession_to_meta_json("GSE157103", out_dir="meta")
print("Saved:", p)

# 2) Read the JSON and extract PRJNA / SRP for downstream EDirect usage:
import json
d = json.loads(Path(p).read_text(encoding="utf-8"))
prjna = d["extracted"]["BioProject"] or (d["extracted"]["BioProject_all"][0] if d["extracted"]["BioProject_all"] else None)
srp   = d["extracted"]["SRA"] or (d["extracted"]["SRAnum_all"][0] if d["extracted"]["SRAnum_all"] else None)
print("PRJNA:", prjna, " SRP:", srp)
'''
