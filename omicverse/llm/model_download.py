from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class ModelSourceType(str, Enum):
    HF = "hf"
    URL = "url"
    GDRIVE = "gdrive"


@dataclass
class ModelSpec:
    """Specification for a pretrained checkpoint source.

    Exactly one of the source fields should be populated depending on `source_type`.
    """

    # Common
    name: str
    source_type: ModelSourceType
    description: str = ""

    # Hugging Face Hub
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_allow_patterns: Optional[Sequence[str]] = None

    # Direct URL(s). If multiple, all will be fetched into the target directory.
    urls: Optional[Sequence[str]] = None
    url_filenames: Dict[str, str] = field(default_factory=dict)
    extract_archives: bool = False

    # Google Drive
    gdrive_id: Optional[str] = None  # file or folder id
    gdrive_is_folder: bool = False

    # Optional checksums: mapping of filename (relative to target_dir) -> sha256 hex digest
    checksums: Dict[str, str] = field(default_factory=dict)


# Minimal registry of common single-cell foundation models
# Note: scGPT checkpoints are hosted on Google Drive folders by authors.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Geneformer V2 (Hugging Face repo hosts multiple variants). We fetch the whole repo snapshot by default.
    "geneformer": ModelSpec(
        name="geneformer",
        source_type=ModelSourceType.HF,
        hf_repo_id="ctheodoris/Geneformer",
        description="Geneformer foundation model hub repo (multiple variants inside)",
    ),
    "geneformer-v2-316m": ModelSpec(
        name="geneformer-v2-316m",
        source_type=ModelSourceType.HF,
        hf_repo_id="ctheodoris/Geneformer",
        hf_allow_patterns=[
            "Geneformer-V2-316M/*",
            "geneformer/*gc316M.pkl",
        ],
        # Leave allow_patterns None to fetch complete snapshot; callers may override
        description="Geneformer V2 316M parameters (default in repo).",
    ),
    "geneformer-v2-104m": ModelSpec(
        name="geneformer-v2-104m",
        source_type=ModelSourceType.HF,
        hf_repo_id="ctheodoris/Geneformer",
        hf_allow_patterns=[
            "Geneformer-V2-104M/*",
            "geneformer/*gc104M.pkl",
        ],
        description="Geneformer V2 104M parameters.",
    ),
    "geneformer-v1-10m": ModelSpec(
        name="geneformer-v1-10m",
        source_type=ModelSourceType.HF,
        hf_repo_id="ctheodoris/Geneformer",
        description="Geneformer V1 10M parameters.",
    ),
    "scfoundation": ModelSpec(
        name="scfoundation",
        source_type=ModelSourceType.HF,
        hf_repo_id="genbio-ai/scFoundation",
        hf_allow_patterns=["models.ckpt"],
        description="scFoundation checkpoint (models.ckpt) from Hugging Face.",
    ),
    "uce": ModelSpec(
        name="uce",
        source_type=ModelSourceType.URL,
        urls=[
            "https://figshare.com/ndownloader/files/42706558",
            "https://figshare.com/ndownloader/files/42706555",
            "https://figshare.com/ndownloader/files/42706585",
            "https://figshare.com/ndownloader/files/42706576",
            "https://figshare.com/ndownloader/files/42715213",
        ],
        url_filenames={
            "https://figshare.com/ndownloader/files/42706558": "species_chrom.csv",
            "https://figshare.com/ndownloader/files/42706555": "species_offsets.pkl",
            "https://figshare.com/ndownloader/files/42706585": "token_to_pos.torch",
            "https://figshare.com/ndownloader/files/42706576": "4layer_model.torch",
            "https://figshare.com/ndownloader/files/42715213": "protein_embeddings.tar.gz",
        },
        extract_archives=True,
        description="UCE checkpoint bundle from figshare including protein embeddings archive.",
    ),
    # scCello on HF
    "sccello-zeroshot": ModelSpec(
        name="sccello-zeroshot",
        source_type=ModelSourceType.HF,
        hf_repo_id="katarinayuan/scCello-zeroshot",
        description="scCello zero-shot checkpoint on Hugging Face",
    ),
    "scmulan": ModelSpec(
        name="scmulan",
        source_type=ModelSourceType.URL,
        urls=["https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1"],
        url_filenames={
            "https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1": "ckpt_scMulan.pt",
        },
        description="scMulan checkpoint from the upstream Tsinghua sharing link.",
    ),
    # scGPT checkpoints (Google Drive folders provided by authors)
    "scgpt-whole-human": ModelSpec(
        name="scgpt-whole-human",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
        gdrive_is_folder=True,
        description="scGPT pretrained on 33M normal human cells (Google Drive folder)",
    ),
    "scgpt-continual-pretrained": ModelSpec(
        name="scgpt-continual-pretrained",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB",
        gdrive_is_folder=True,
        description="scGPT continual pretrained checkpoint for cell embedding tasks (Google Drive folder)",
    ),
    "scgpt-brain": ModelSpec(
        name="scgpt-brain",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx",
        gdrive_is_folder=True,
        description="scGPT brain checkpoint (Google Drive folder)",
    ),
    "scgpt-blood": ModelSpec(
        name="scgpt-blood",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU",
        gdrive_is_folder=True,
        description="scGPT blood & bone marrow checkpoint (Google Drive folder)",
    ),
    "scgpt-heart": ModelSpec(
        name="scgpt-heart",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r",
        gdrive_is_folder=True,
        description="scGPT heart checkpoint (Google Drive folder)",
    ),
    "scgpt-lung": ModelSpec(
        name="scgpt-lung",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG",
        gdrive_is_folder=True,
        description="scGPT lung checkpoint (Google Drive folder)",
    ),
    "scgpt-kidney": ModelSpec(
        name="scgpt-kidney",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK",
        gdrive_is_folder=True,
        description="scGPT kidney checkpoint (Google Drive folder)",
    ),
    "scgpt-pan-cancer": ModelSpec(
        name="scgpt-pan-cancer",
        source_type=ModelSourceType.GDRIVE,
        gdrive_id="13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa",
        gdrive_is_folder=True,
        description="scGPT pan-cancer checkpoint (Google Drive folder)",
    ),
}


def get_default_models_dir() -> Path:
    base = os.environ.get("OMICVERSE_HOME")
    if base:
        return Path(base).expanduser().resolve() / "models"
    # fall back to ~/.cache/omicverse/models
    return Path.home() / ".cache" / "omicverse" / "models"


def list_available_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_file(file_path: Union[str, Path]) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_checksums(target_dir: Path, checksums: Mapping[str, str]) -> None:
    for rel_path, expected in checksums.items():
        file_path = target_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file missing for checksum: {file_path}")
        actual = _sha256_file(file_path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Checksum mismatch for {file_path}\nExpected: {expected}\nActual:   {actual}"
            )


def _extract_downloaded_archives(target_dir: Path) -> None:
    """Extract any supported archives already present in ``target_dir``."""

    def _resolve_extraction_path(base_dir: Path, member_name: str, source: Path) -> Path:
        destination = (base_dir / member_name).resolve()
        if base_dir not in destination.parents and destination != base_dir:
            raise ValueError(
                f"Refusing to extract unsafe path '{member_name}' from archive: {source}"
            )
        return destination

    def _resolve_link_target(base_dir: Path, link_name: str, link_target: str, source: Path) -> None:
        if not link_target:
            return
        link_parent = (base_dir / link_name).resolve().parent
        resolved_target = (link_parent / link_target).resolve()
        if base_dir not in resolved_target.parents and resolved_target != base_dir:
            raise ValueError(
                f"Refusing to extract unsafe link target '{link_target}' from archive: {source}"
            )

    for archive_path in sorted(target_dir.iterdir()):
        if archive_path.is_dir():
            continue
        name = archive_path.name.lower()
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive_path, "r:*") as tar:
                for member in tar.getmembers():
                    _resolve_extraction_path(target_dir, member.name, archive_path)
                    if member.issym() or member.islnk():
                        _resolve_link_target(
                            target_dir,
                            member.name,
                            getattr(member, "linkname", ""),
                            archive_path,
                        )
                extractall_kwargs = {"path": target_dir}
                if "filter" in tarfile.TarFile.extractall.__code__.co_varnames:
                    extractall_kwargs["filter"] = "data"
                tar.extractall(**extractall_kwargs)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    _resolve_extraction_path(target_dir, member.filename, archive_path)
                zf.extractall(target_dir)


def _download_hf_snapshot(
    repo_id: str,
    target_dir: Path,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    token: Optional[str] = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF downloads. Install with: pip install huggingface_hub"
        ) from exc

    # snapshot_download handles resuming and symlinking from cache efficiently
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
        # use default cache dir; users can override via HF_HOME or HF_HUB_CACHE
    )
    return Path(snapshot_path)


def _download_url(
    url: str,
    target_file: Path,
    resume: bool = True,
) -> Path:
    """Download a single URL with optional resume support."""
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("requests is required for URL downloads. Install with: pip install requests") from exc

    temp_file = target_file.with_suffix(target_file.suffix + ".part")
    headers: Dict[str, str] = {}
    mode = "wb"
    existing_size = 0
    if resume and temp_file.exists():
        existing_size = temp_file.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        if total is not None and "Content-Range" not in r.headers and existing_size > 0:
            # Server ignored range; restart
            existing_size = 0
            mode = "wb"
        with open(temp_file, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    temp_file.rename(target_file)
    return target_file


def _download_gdrive(
    file_or_folder_id: str,
    target_dir: Path,
    is_folder: bool = False,
) -> Path:
    """Download from Google Drive using gdown.

    For folders, it will create a subdirectory under target_dir.
    """
    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise RuntimeError("gdown is required for Google Drive downloads. Install with: pip install gdown") from exc

    target_dir = _ensure_dir(target_dir)
    if is_folder:
        # gdown supports folder download via id
        out_dir = target_dir
        gdown.download_folder(id=file_or_folder_id, output=str(out_dir), quiet=False, use_cookies=False)
        return out_dir
    else:
        # Single file id
        # gdown will preserve filename by default
        gdown.download(id=file_or_folder_id, output=str(target_dir), quiet=False)
        return target_dir


def download_model(
    model: Union[str, ModelSpec],
    target_dir: Optional[Union[str, Path]] = None,
    *,
    hf_token: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_allow_patterns: Optional[Sequence[str]] = None,
    url_filenames: Optional[Mapping[str, str]] = None,
    verify_checksums: bool = True,
) -> Path:
    """Download a single-cell foundation model checkpoint.

    Parameters
    ----------
    model: str | ModelSpec
        Either a key from the internal registry (see list_available_models()), or a custom ModelSpec.
    target_dir: str | Path | None
        Destination directory. Defaults to ~/.cache/omicverse/models/<model_name>.
    hf_token: str | None
        Hugging Face access token if needed (for private repos).
    hf_revision: str | None
        Optional override for HF revision/tag/commit.
    hf_allow_patterns: Sequence[str] | None
        Optional include patterns for HF snapshot (e.g., ["*.safetensors", "config.json"]).
    url_filenames: Mapping[str, str] | None
        For direct URL sources, mapping of url -> desired filename in target_dir. If omitted, the filename
        will be inferred from the URL path.
    verify_checksums: bool
        If True and checksums are provided in the spec, verify after download.

    Returns
    -------
    Path
        Path to the directory containing the downloaded checkpoint files.
    """
    if isinstance(model, str):
        key = model.lower()
        if key not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model '{model}'. Available: {', '.join(list_available_models())}"
            )
        spec = MODEL_REGISTRY[key]
    else:
        spec = model

    final_target_dir = _ensure_dir(
        target_dir if target_dir is not None else get_default_models_dir() / spec.name
    )

    if spec.source_type == ModelSourceType.HF:
        repo_id = spec.hf_repo_id
        if not repo_id:
            raise ValueError("HF model spec missing hf_repo_id")
        revision = hf_revision if hf_revision is not None else spec.hf_revision
        allow = hf_allow_patterns if hf_allow_patterns is not None else spec.hf_allow_patterns
        _download_hf_snapshot(repo_id=repo_id, target_dir=final_target_dir, revision=revision, allow_patterns=allow, token=hf_token)
    elif spec.source_type == ModelSourceType.URL:
        if not spec.urls:
            raise ValueError("URL model spec missing urls")
        effective_url_filenames = dict(spec.url_filenames)
        if url_filenames:
            effective_url_filenames.update(url_filenames)
        for url in spec.urls:
            # infer filename or use provided mapping
            if url in effective_url_filenames:
                filename = effective_url_filenames[url]
            else:
                # Use the last non-empty segment of the URL path
                parsed_name = re.sub(r"[?#].*$", "", url.rstrip("/"))
                filename = parsed_name.split("/")[-1]
                if not filename:
                    raise ValueError(f"Cannot infer filename from URL: {url}. Provide url_filenames mapping.")
            _download_url(url=url, target_file=final_target_dir / filename, resume=True)
    elif spec.source_type == ModelSourceType.GDRIVE:
        if not spec.gdrive_id:
            raise ValueError("Google Drive spec missing gdrive_id")
        _download_gdrive(file_or_folder_id=spec.gdrive_id, target_dir=final_target_dir, is_folder=spec.gdrive_is_folder)
    else:
        raise ValueError(f"Unsupported source type: {spec.source_type}")

    if spec.extract_archives:
        _extract_downloaded_archives(final_target_dir)

    if verify_checksums and spec.checksums:
        _verify_checksums(final_target_dir, spec.checksums)

    return final_target_dir


__all__ = [
    "ModelSourceType",
    "ModelSpec",
    "MODEL_REGISTRY",
    "list_available_models",
    "download_model",
] 
