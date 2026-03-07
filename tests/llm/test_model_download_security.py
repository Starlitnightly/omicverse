from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DOWNLOAD_PATH = PROJECT_ROOT / "omicverse" / "llm" / "model_download.py"

_spec = importlib.util.spec_from_file_location("model_download", MODEL_DOWNLOAD_PATH)
assert _spec and _spec.loader
_model_download = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _model_download
_spec.loader.exec_module(_model_download)
_extract_downloaded_archives = _model_download._extract_downloaded_archives


def _build_tar_with_member(archive_path: Path, member_name: str, data: bytes = b"payload") -> None:
    with tarfile.open(archive_path, "w:gz") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


def test_extract_downloaded_archives_rejects_tar_path_traversal(tmp_path: Path) -> None:
    target_dir = tmp_path / "models"
    target_dir.mkdir()
    _build_tar_with_member(target_dir / "bundle.tar.gz", "../owned.txt", b"owned")

    outside_file = tmp_path / "owned.txt"
    with pytest.raises(ValueError, match="unsafe path"):
        _extract_downloaded_archives(target_dir)

    assert not outside_file.exists()


def test_extract_downloaded_archives_rejects_zip_path_traversal(tmp_path: Path) -> None:
    target_dir = tmp_path / "models"
    target_dir.mkdir()

    archive_path = target_dir / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../owned.txt", "owned")

    outside_file = tmp_path / "owned.txt"
    with pytest.raises(ValueError, match="unsafe path"):
        _extract_downloaded_archives(target_dir)

    assert not outside_file.exists()


def test_extract_downloaded_archives_allows_safe_archives(tmp_path: Path) -> None:
    target_dir = tmp_path / "models"
    target_dir.mkdir()

    _build_tar_with_member(target_dir / "bundle.tar.gz", "nested/model.bin", b"ok")
    with zipfile.ZipFile(target_dir / "bundle.zip", "w") as zf:
        zf.writestr("nested/config.json", "{}")

    _extract_downloaded_archives(target_dir)

    assert (target_dir / "nested" / "model.bin").exists()
    assert (target_dir / "nested" / "config.json").exists()
