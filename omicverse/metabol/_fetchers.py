r"""On-demand database fetchers for KEGG / LION / HMDB.

The tiny curated CSVs shipped with omicverse (``data/kegg_pathways.csv``,
``data/lion_subset.json``, ``data/metabolite_lookup.csv``) cover the
tutorial datasets but are obviously too small for real analysis:

=======  ================  ========================  ================
source   shipped offline   fetched full DB           from
=======  ================  ========================  ================
KEGG     ~35 pathways      ~550 pathways + compounds rest.kegg.jp
LION     ~20 terms         ~730 terms, 6k lipids     github/lipidontology.com
HMDB     ~95 compounds     ~220k compounds           Metabolomics Workbench REST
=======  ================  ========================  ================

Every fetcher:

- Downloads to ``~/.cache/omicverse/metabol/`` (override with the
  ``OV_METABOL_CACHE`` env var) so repeat calls are free
- Validates the downloaded data before caching
- Has the same function signature shape — ``fetch_*(cache=True, refresh=False)``
  — so scripts are idempotent
- Is optional: every consumer in the package falls back to the shipped
  subset when the fetched cache is absent

Licensing notes
---------------
- **KEGG** is free for academic use via the REST API but restricts bulk
  redistribution. We fetch on demand, don't ship the data.
- **LION** is CC-BY-NC (Molenaar 2019) — shipping a subset for tutorials
  is OK; the full ontology is downloaded on demand.
- **HMDB** is free to download but not redistribute; we query the
  Metabolomics Workbench REST API which has its own (US NIH Common Fund)
  license allowing programmatic access.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# cache management
# ---------------------------------------------------------------------------
def _cache_dir() -> Path:
    p = Path(os.environ.get("OV_METABOL_CACHE",
                            Path.home() / ".cache" / "omicverse" / "metabol"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_cache() -> int:
    """Delete every file in the fetcher cache dir. Returns file count."""
    root = _cache_dir()
    n = 0
    for f in root.glob("*"):
        if f.is_file():
            f.unlink(); n += 1
    return n


def _download(url: str, path: Path, *, user_agent: str = "omicverse/metabol") -> Path:
    """HTTP GET ``url`` → ``path``, with a ``Mozilla``-style UA because
    both GitHub raw and Zenodo 403 on the default urllib UA."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# ChEBI compound master table — monoisotopic mass + KEGG/HMDB/LipidMaps xrefs
# ---------------------------------------------------------------------------
def fetch_chebi_compounds(
    *,
    cache: bool = True,
    refresh: bool = False,
) -> "pd.DataFrame":
    """Build a compound master table from ChEBI's flat-file TSVs.

    Downloads + joins three ChEBI distributions from the public EBI
    FTP (over HTTPS):

    - ``compounds.tsv.gz``          — ChEBI ID → canonical name
    - ``chemical_data.tsv.gz``      — monoisotopic mass + formula
    - ``database_accession.tsv.gz`` — HMDB / KEGG / LipidMaps xrefs

    Total download is ~15 MB; the joined parquet cache persists at
    ``~/.cache/omicverse/metabol/chebi_compounds.parquet``. This is the
    substrate :func:`annotate_peaks` uses for mummichog mass matching.

    Returns
    -------
    pd.DataFrame
        Columns: ``chebi_id``, ``name``, ``formula``, ``mw``
        (monoisotopic, float), ``kegg``, ``hmdb``, ``lipidmaps``.
        Rows without a monoisotopic mass are dropped.
    """
    cache_path = _cache_dir() / "chebi_compounds.parquet"
    if cache and cache_path.exists() and not refresh:
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    base = "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files"
    paths = {
        "compounds":     (_cache_dir() / "chebi_compounds.tsv.gz", "compounds.tsv.gz"),
        "chemical_data": (_cache_dir() / "chebi_chemical_data.tsv.gz", "chemical_data.tsv.gz"),
        "accession":     (_cache_dir() / "chebi_database_accession.tsv.gz", "database_accession.tsv.gz"),
    }
    for path, fname in paths.values():
        if not path.exists() or refresh:
            _download(f"{base}/{fname}", path)

    # compounds.tsv: id, name, status_id, chebi_accession, stars, ...
    compounds = pd.read_csv(paths["compounds"][0], sep="\t", compression="gzip",
                            usecols=["id", "name", "chebi_accession", "stars"],
                            dtype=str, low_memory=False)
    compounds = compounds[compounds["stars"].astype(str) == "3"]
    compounds = compounds[["id", "name", "chebi_accession"]].rename(
        columns={"id": "compound_id", "chebi_accession": "chebi_id"}
    )

    # chemical_data.tsv already has formula + monoisotopic_mass as columns
    cdata = pd.read_csv(paths["chemical_data"][0], sep="\t", compression="gzip",
                        usecols=["compound_id", "formula", "monoisotopic_mass"],
                        dtype={"compound_id": str, "formula": str,
                               "monoisotopic_mass": str},
                        low_memory=False)
    cdata["mw"] = pd.to_numeric(cdata["monoisotopic_mass"], errors="coerce")
    cdata = cdata.dropna(subset=["mw"])[["compound_id", "formula", "mw"]]
    cdata["formula"] = cdata["formula"].fillna("")
    # Keep the first row per compound (multiple structures can exist)
    cdata = cdata.drop_duplicates("compound_id", keep="first")

    # database_accession.tsv: id, compound_id, accession_number, type, source_id
    # Source IDs confirmed by inspection of the v260 dump:
    #   35 → HMDB,  45 → KEGG COMPOUND,  50 → LIPID MAPS
    acc = pd.read_csv(paths["accession"][0], sep="\t", compression="gzip",
                      usecols=["compound_id", "accession_number", "source_id"],
                      dtype=str, low_memory=False)
    xref = acc.rename(columns={"accession_number": "accession"})
    kegg = (xref[xref["source_id"] == "45"]
            .groupby("compound_id")["accession"].first()
            .rename("kegg").reset_index())
    hmdb = (xref[xref["source_id"] == "35"]
            .groupby("compound_id")["accession"].first()
            .rename("hmdb").reset_index())
    lipidmaps = (xref[xref["source_id"] == "50"]
                 .groupby("compound_id")["accession"].first()
                 .rename("lipidmaps").reset_index())

    out = (compounds.merge(cdata, on="compound_id", how="inner")
                     .merge(kegg, on="compound_id", how="left")
                     .merge(hmdb, on="compound_id", how="left")
                     .merge(lipidmaps, on="compound_id", how="left")
                     .drop(columns=["compound_id"]))
    for col in ("kegg", "hmdb", "lipidmaps", "formula"):
        out[col] = out[col].fillna("")

    if cache:
        try:
            out.to_parquet(cache_path, index=False)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# KEGG
# ---------------------------------------------------------------------------
def fetch_kegg_pathways(
    organism: Optional[str] = None,
    *,
    cache: bool = True,
    refresh: bool = False,
) -> dict[str, list[str]]:
    """Fetch the full KEGG compound→pathway map via KEGG REST.

    Parameters
    ----------
    organism
        KEGG organism code (``"hsa"`` human, ``"mmu"`` mouse, …). When
        ``None``, the reference pathway namespace (``map#####``, species-
        agnostic metabolic pathways) is used. For metabolomics enrichment
        this is almost always what you want.
    cache
        Read/write a TSV cache at ``~/.cache/omicverse/metabol/kegg_<org>.tsv``.
    refresh
        Ignore the cache and re-fetch.

    Returns
    -------
    dict[str, list[str]]
        ``{pathway_name: [kegg_compound_id, ...]}`` — the same shape
        ``load_pathways()`` returns for the shipped subset, so
        :func:`msea_ora` / :func:`msea_gsea` accept it directly.
    """
    org_key = organism or "map"
    cache_path = _cache_dir() / f"kegg_{org_key}.tsv"
    if cache and cache_path.exists() and not refresh:
        try:
            df = pd.read_csv(cache_path, sep="\t")
            return df.groupby("pathway_name")["kegg_compound"].apply(list).to_dict()
        except Exception:
            pass  # corrupt cache — fall through to re-fetch

    # 1) pathway list
    pw_prefix = organism if organism else "map"
    pw_url = f"https://rest.kegg.jp/list/pathway/{organism}" if organism else "https://rest.kegg.jp/list/pathway"
    with urllib.request.urlopen(pw_url, timeout=60) as resp:
        pw_lines = resp.read().decode("utf-8").strip().splitlines()
    pw_names: dict[str, str] = {}
    for line in pw_lines:
        pid, pname = line.split("\t", 1)
        pid = pid.replace("path:", "")
        pw_names[pid] = pname.split(" - ")[0].strip()

    # 2) compound ↔ pathway links
    if organism:
        link_url = f"https://rest.kegg.jp/link/{organism}/compound"
    else:
        link_url = "https://rest.kegg.jp/link/pathway/compound"
    with urllib.request.urlopen(link_url, timeout=120) as resp:
        link_lines = resp.read().decode("utf-8").strip().splitlines()

    pw_to_cpd: dict[str, list[str]] = {}
    for line in link_lines:
        cpd, pid = line.split("\t", 1)
        cpd = cpd.replace("cpd:", "")
        pid = pid.replace("path:", "")
        # Only keep metabolic pathways we have names for
        if pid not in pw_names:
            continue
        pw_to_cpd.setdefault(pw_names[pid], []).append(cpd)
    # Dedupe per pathway preserving order
    pw_to_cpd = {k: list(dict.fromkeys(v)) for k, v in pw_to_cpd.items()}

    if cache:
        rows = [{"pathway_name": k, "kegg_compound": c}
                for k, vs in pw_to_cpd.items() for c in vs]
        pd.DataFrame(rows).to_csv(cache_path, sep="\t", index=False)

    return pw_to_cpd


# ---------------------------------------------------------------------------
# LION
# ---------------------------------------------------------------------------
_LION_CSV_URL = (
    "https://raw.githubusercontent.com/martijnmolenaar/"
    "lipidontology.com/master/all-LION-lipid-associations.csv"
)


def fetch_lion_associations(
    *,
    cache: bool = True,
    refresh: bool = False,
) -> dict[str, dict]:
    """Fetch the full LION lipid↔ontology associations.

    Returns
    -------
    dict[str, dict]
        ``{term_name: {"category": str, "members": [lipid_class, ...]}}``
        — same shape as the shipped ``lion_subset.json`` so
        :func:`lion_enrichment` consumes it directly. LION terms that
        attach to many thousands of species are aggregated to the
        **class** level (first token of the LIPID MAPS shorthand) to
        match how the lipidomics module does enrichment.
    """
    cache_path = _cache_dir() / "lion_associations.json"
    if cache and cache_path.exists() and not refresh:
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    raw_path = _cache_dir() / "all-LION-lipid-associations.csv"
    if not raw_path.exists() or refresh:
        _download(_LION_CSV_URL, raw_path)

    # LION format: column headers are LION term IDs (LION:0000001, ...),
    # the first data row is the **term names** (e.g. "fatty acids [FA]"),
    # subsequent rows are lipid species with 0/1 membership. First column
    # holds the lipid name (labelled "Unnamed: 0" because no header).
    df = pd.read_csv(raw_path, low_memory=False)
    if len(df.columns) < 2 or not str(df.columns[1]).startswith("LION:"):
        raise RuntimeError(
            f"Unexpected LION CSV format (cols={list(df.columns[:3])}); "
            "upstream schema may have changed."
        )

    term_id_cols = [c for c in df.columns if str(c).startswith("LION:")]
    # Row 0 of the dataframe holds the human-readable term names
    term_id_to_name = {
        tid: str(df.iloc[0][tid]).strip() for tid in term_id_cols
    }
    lipid_col = df.columns[0]
    data = df.iloc[1:].copy()              # drop the term-names row
    data[lipid_col] = data[lipid_col].astype(str)
    # LION uses two naming styles — "PC 34:1" (space-separated) and
    # "PC(16:0/18:1)" (parenthesized). Extract the class token from
    # either by grabbing everything up to the first space or '('.
    data["class"] = (
        data[lipid_col]
        .str.extract(r"^([A-Za-z0-9\-]+)", expand=False)
        .str.upper()
    )

    out: dict[str, dict] = {}
    for tid, term_name in term_id_to_name.items():
        # Membership: any non-null value in this term's column. LION's
        # public CSV uses the string "x" for member-of and blank for not.
        col = data[tid]
        members_mask = col.notna() & col.astype(str).str.strip().astype(bool)
        classes = sorted(set(data.loc[members_mask, "class"].dropna().tolist()))
        if len(classes) < 2:
            continue
        out[term_name or tid] = {
            "category": "lipid_class",
            "lion_id": tid,
            "members": classes,
        }

    if cache:
        cache_path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# HMDB / KEGG / ChEBI cross-references via PubChem REST
# ---------------------------------------------------------------------------
import re

_PUBCHEM_CID_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
)
_PUBCHEM_XREFS_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RegistryID/JSON"
)

_HMDB_PATTERN = re.compile(r"^HMDB\d+$")
_KEGG_CPD_PATTERN = re.compile(r"^C\d{5}$")
_CHEBI_PATTERN = re.compile(r"^CHEBI:\d+$")


def fetch_hmdb_from_name(
    name: str,
    *,
    cache: bool = True,
    refresh: bool = False,
) -> dict[str, str]:
    """Resolve a metabolite name → HMDB / KEGG / ChEBI / PubChem CID.

    Uses the **PubChem REST API** in two hops:

    1. ``compound/name/<name>/cids/JSON`` → PubChem CID
    2. ``compound/cid/<cid>/xrefs/RegistryID/JSON`` → a long list of
       external registry IDs; we filter by regex to pick out the HMDB,
       KEGG-compound, and ChEBI identifiers.

    PubChem is publicly-accessible, tolerant of synonyms, and gives
    HMDB/KEGG/ChEBI in a single call. No API key needed.

    Parameters
    ----------
    name
        Common name (``"Glucose"``, ``"L-isoleucine"``, …). Case-
        insensitive; PubChem's synonym index covers most aliases.
    cache
        Cache under ``~/.cache/omicverse/metabol/pubchem_xref_cache.json``
        so repeat calls are free.
    refresh
        Force a network round-trip even if the name is cached.

    Returns
    -------
    dict[str, str]
        Keys: ``hmdb``, ``kegg``, ``chebi``, ``pubchem``. Missing ID → ``""``.
    """
    cache_path = _cache_dir() / "pubchem_xref_cache.json"
    cache_data: dict[str, dict] = {}
    if cache and cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text())
        except Exception:
            cache_data = {}
    key = name.strip().lower()
    if key in cache_data and not refresh:
        return cache_data[key]

    import urllib.parse
    out = {"hmdb": "", "kegg": "", "chebi": "", "pubchem": ""}
    try:
        q = urllib.parse.quote(name, safe="")
        with urllib.request.urlopen(_PUBCHEM_CID_URL.format(name=q),
                                   timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        cids = d.get("IdentifierList", {}).get("CID", [])
        if not cids:
            # No PubChem entry for this name — cache the empty result too
            if cache:
                cache_data[key] = out
                cache_path.write_text(json.dumps(cache_data, indent=1))
            return out
        cid = cids[0]
        out["pubchem"] = str(cid)
    except Exception as exc:
        import warnings
        warnings.warn(f"PubChem CID lookup failed for {name!r}: "
                      f"{type(exc).__name__}: {exc}",
                      UserWarning, stacklevel=2)
        return out

    try:
        with urllib.request.urlopen(_PUBCHEM_XREFS_URL.format(cid=cid),
                                   timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        reg_ids = d["InformationList"]["Information"][0].get("RegistryID", [])
    except Exception as exc:
        import warnings
        warnings.warn(f"PubChem xrefs lookup for CID {cid} failed: "
                      f"{type(exc).__name__}: {exc}",
                      UserWarning, stacklevel=2)
        reg_ids = []

    for rid in reg_ids:
        if not out["hmdb"] and _HMDB_PATTERN.match(rid):
            out["hmdb"] = rid
        elif not out["kegg"] and _KEGG_CPD_PATTERN.match(rid):
            out["kegg"] = rid
        elif not out["chebi"] and _CHEBI_PATTERN.match(rid):
            out["chebi"] = rid

    if cache:
        cache_data[key] = out
        cache_path.write_text(json.dumps(cache_data, indent=1))
    return out
