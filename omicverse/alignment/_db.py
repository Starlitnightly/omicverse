"""Reference-database helpers for 16S / amplicon pipelines.

Every function requires an explicit ``db_dir`` — no ``$HOME`` or other
implicit defaults. Downloads land directly under the caller-supplied path.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

from .._registry import register_function
from ._cli_utils import ensure_dir


_DEFAULT_TIMEOUT = 300


_SOURCES = {
    "rdp_16s_v18": {
        "url": "https://drive5.com/sintax/rdp_16s_v18.fa.gz",
        "filename": "rdp_16s_v18.fa.gz",
        "size_mb": 6.8,
        # Verified 2026-04-18 against drive5.com mirror.
        "sha256": "8dd858c00a89d43cca4289463adaa7f5ebba6e85c05daab950518e1b16f61ce0",
        "description": "RDP 16S SINTAX-formatted reference (v18). Small (6.8 MB), "
                       "well-tested, covers bacteria + archaea. Pre-formatted for "
                       "vsearch --sintax.",
    },
    "silva_16s_v123": {
        "url": "https://drive5.com/sintax/silva_16s_v123.fa.gz",
        "filename": "silva_16s_v123.fa.gz",
        "size_mb": 439.0,
        # 440 MB file; checksum deliberately not included here because
        # (a) the reference is deprecated (SILVA v123 is from 2015) and
        # (b) the upstream host does not publish a reference hash. Users
        # who need integrity checking should host a mirror and pass their
        # own fasta via `db_fasta=` instead of relying on this helper.
        "sha256": None,
        "description": "SILVA 16S SINTAX-formatted (v123). Comprehensive (~440 MB). "
                       "NOTE: v123 was released in 2015 — current SILVA is v138.1+. "
                       "Consider exporting a newer SILVA release to SINTAX format "
                       "and passing it via `db_fasta=` directly.",
    },
}


def _sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(
    url: str,
    dest: Path,
    overwrite: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
    expected_sha256: Optional[str] = None,
) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        if expected_sha256:
            actual = _sha256_file(dest)
            if actual != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch on cached {dest}:\n"
                    f"  expected: {expected_sha256}\n"
                    f"  actual  : {actual}\n"
                    f"Pass overwrite=True to re-download."
                )
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if expected_sha256:
            actual = _sha256_file(tmp)
            if actual != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for downloaded {url}:\n"
                    f"  expected: {expected_sha256}\n"
                    f"  actual  : {actual}\n"
                    f"Refusing to install a corrupted file."
                )
        tmp.rename(dest)
        return dest
    except BaseException:
        # Any interruption — network error, checksum mismatch, KeyboardInterrupt,
        # disk full — must not leave a stale .part behind that a subsequent run
        # could mistake for a valid cache.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _resolve_db_dir(db_dir: Optional[str]) -> Path:
    """Resolve db_dir strictly — no implicit $HOME fallback.

    Order:
      1. Explicit ``db_dir`` argument (must not be ``None`` and not empty).
      2. ``OMICVERSE_DB_DIR`` environment variable.
      3. Raise ``ValueError`` (never write to ``$HOME``).
    """
    if db_dir is not None and str(db_dir).strip():
        return Path(db_dir).expanduser().resolve()
    env_dir = os.environ.get("OMICVERSE_DB_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    raise ValueError(
        "db_dir must be specified (either as the `db_dir` argument or via the "
        "OMICVERSE_DB_DIR environment variable). omicverse never writes reference "
        "databases to $HOME — point db_dir at a path under /scratch or similar."
    )


@register_function(
    aliases=["fetch_sintax_ref", "download_16s_db"],
    category="alignment",
    description="Download a SINTAX-formatted 16S reference database. db_dir is required (no $HOME writes).",
    examples=[
        "ov.alignment.fetch_sintax_ref("
        "'rdp_16s_v18', db_dir='/scratch/.../db/rdp')",
    ],
    related=["alignment.vsearch"],
)
def fetch_sintax_ref(
    source: str = "rdp_16s_v18",
    db_dir: Optional[str] = None,
    overwrite: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Download a SINTAX-formatted 16S reference FASTA.

    Parameters
    ----------
    source
        One of ``'rdp_16s_v18'`` (small, 6.8 MB) or ``'silva_16s_v123'``
        (comprehensive, ~440 MB; but old — see notes below). Both are
        pre-formatted for vsearch ``--sintax``.
    db_dir
        **Required** (or set ``OMICVERSE_DB_DIR``). Target directory under
        which the reference is saved. No ``$HOME`` fallback.
    overwrite
        Re-download even if the file already exists.
    timeout
        Per-connection read timeout in seconds (default 300).

    Returns
    -------
    str
        Absolute path to the downloaded ``.fa.gz``.
    """
    if source not in _SOURCES:
        raise ValueError(
            f"Unknown source '{source}'. Known: {sorted(_SOURCES)}"
        )
    spec = _SOURCES[source]
    base = _resolve_db_dir(db_dir)
    sub = base / source
    ensure_dir(sub)
    dest = sub / spec["filename"]
    _download(
        spec["url"], dest,
        overwrite=overwrite,
        timeout=timeout,
        expected_sha256=spec.get("sha256"),
    )
    return str(dest)


@register_function(
    aliases=["fetch_silva", "fetch_silva_sintax"],
    category="alignment",
    description="Convenience alias: download SILVA (v123, 2015) SINTAX 16S reference. db_dir required. See docstring for freshness caveat.",
    examples=[
        "ov.alignment.fetch_silva(db_dir='/scratch/.../db/silva')",
    ],
    related=["alignment.fetch_sintax_ref"],
)
def fetch_silva(
    db_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Alias for ``fetch_sintax_ref('silva_16s_v123', db_dir=...)``.

    .. warning::
        ``silva_16s_v123`` dates from 2015. The current SILVA release is
        v138.1+ (2020). For publication-grade work, export a newer SILVA
        release to SINTAX format yourself and pass it to
        :func:`omicverse.alignment.vsearch.sintax` via ``db_fasta=``.
        A `DeprecationWarning` is emitted each time this alias is called.
    """
    warnings.warn(
        "fetch_silva() downloads SILVA v123 (2015). Current SILVA is v138.1+ "
        "(2020). For up-to-date taxonomy, export a newer SILVA release to "
        "SINTAX format and pass it via db_fasta= directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fetch_sintax_ref("silva_16s_v123", db_dir=db_dir, overwrite=overwrite)


@register_function(
    aliases=["fetch_rdp", "fetch_rdp_sintax"],
    category="alignment",
    description="Convenience alias: download RDP 16S v18 SINTAX reference (small, 6.8 MB). db_dir required.",
    examples=[
        "ov.alignment.fetch_rdp(db_dir='/scratch/.../db/rdp')",
    ],
    related=["alignment.fetch_sintax_ref"],
)
def fetch_rdp(
    db_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Alias for ``fetch_sintax_ref('rdp_16s_v18', db_dir=...)``."""
    return fetch_sintax_ref("rdp_16s_v18", db_dir=db_dir, overwrite=overwrite)
