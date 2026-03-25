from __future__ import annotations

import base64
import io
import logging
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("omicverse.jarvis.media_ingest")

OPENAI_SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_KNOWN_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

_DEFAULT_IMAGE_PROMPT = (
    "Analyze the user's attached image and respond in the user's language."
)


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    mime_type: str
    size: int
    request_block: dict
    original_filename: str = ""
    source: str = ""


def looks_like_image_name(name: str) -> bool:
    lower = (name or "").strip().lower()
    if not lower:
        return False
    if Path(lower).suffix.lower() in _KNOWN_IMAGE_EXTENSIONS:
        return True
    guessed, _ = mimetypes.guess_type(lower)
    return bool(guessed and guessed.startswith("image/"))


def compose_multimodal_user_text(text: str, image_note: str = "") -> str:
    base = (text or "").strip() or _DEFAULT_IMAGE_PROMPT
    if image_note:
        return f"{base}\n\n{image_note}"
    return base


def build_workspace_note(
    workspace: Path,
    images: Sequence[PreparedImage],
    *,
    header: str = "[Attached images saved in workspace]",
) -> str:
    if not images:
        return ""
    lines = [header]
    for item in images:
        try:
            rel_path = item.path.relative_to(workspace)
            lines.append(f"- workspace/{rel_path.as_posix()}")
        except Exception:
            lines.append(f"- {item.path}")
    return "\n".join(lines)


def prepare_image_bytes(
    data: bytes,
    *,
    target_dir: Path,
    filename: str = "",
    mime_type: str = "",
    prefix: str = "image",
    source: str = "",
) -> PreparedImage:
    normalized, normalized_mime, normalized_ext = _normalize_image_bytes(
        data,
        filename=filename,
        mime_type=mime_type,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _make_filename(prefix, filename, normalized_ext)
    path.write_bytes(normalized)
    return PreparedImage(
        path=path,
        mime_type=normalized_mime,
        size=len(normalized),
        request_block=_build_request_block(normalized, normalized_mime),
        original_filename=filename,
        source=source,
    )


def prepare_image_path(
    path: Path,
    *,
    target_dir: Optional[Path] = None,
    mime_type: str = "",
    prefix: str = "image",
    source: str = "",
) -> PreparedImage:
    raw = path.read_bytes()
    normalized, normalized_mime, normalized_ext = _normalize_image_bytes(
        raw,
        filename=path.name,
        mime_type=mime_type,
    )
    request_block = _build_request_block(normalized, normalized_mime)
    keep_existing = (
        path.exists()
        and normalized == raw
        and normalized_mime == _guess_mime_type(path.name, mime_type)
        and (target_dir is None or path.parent == target_dir)
    )
    if keep_existing:
        final_path = path
    else:
        final_dir = target_dir or path.parent
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / _make_filename(prefix, path.name, normalized_ext)
        final_path.write_bytes(normalized)
    return PreparedImage(
        path=final_path,
        mime_type=normalized_mime,
        size=len(normalized),
        request_block=request_block,
        original_filename=path.name,
        source=source,
    )


def _build_request_block(image_bytes: bytes, mime_type: str) -> dict:
    data_url = "data:{mime};base64,{payload}".format(
        mime=mime_type,
        payload=base64.b64encode(image_bytes).decode("ascii"),
    )
    return {
        "type": "input_image",
        "image_url": data_url,
        "detail": "auto",
    }


def _normalize_image_bytes(
    data: bytes,
    *,
    filename: str = "",
    mime_type: str = "",
) -> tuple[bytes, str, str]:
    resolved_mime = _guess_mime_type(filename, mime_type)
    if resolved_mime in {"image/png", "image/jpeg", "image/webp"}:
        return data, resolved_mime, OPENAI_SUPPORTED_IMAGE_MIME_TYPES[resolved_mime]

    if resolved_mime == "image/gif":
        if _is_animated_gif(data):
            png_bytes = _convert_to_png(data)
            return png_bytes, "image/png", ".png"
        return data, "image/gif", ".gif"

    png_bytes = _convert_to_png(data)
    return png_bytes, "image/png", ".png"


def _guess_mime_type(filename: str, mime_type: str) -> str:
    guessed = (mime_type or "").split(";", 1)[0].strip().lower()
    if guessed.startswith("image/"):
        if guessed == "image/jpg":
            return "image/jpeg"
        return guessed
    by_name, _ = mimetypes.guess_type(filename or "")
    if by_name and by_name.startswith("image/"):
        if by_name == "image/jpg":
            return "image/jpeg"
        return by_name
    return "application/octet-stream"


def _is_animated_gif(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return bool(getattr(image, "is_animated", False) and getattr(image, "n_frames", 1) > 1)
    except Exception:
        logger.debug("gif_animation_probe_failed", exc_info=True)
        return False


def _convert_to_png(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if getattr(image, "is_animated", False):
                image.seek(0)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                converted = image.convert("RGBA")
            else:
                converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG")
            return buffer.getvalue()
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported or unreadable image format") from exc


def _make_filename(prefix: str, filename: str, extension: str) -> str:
    stem = Path(filename or "image").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "image"
    safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix or "image").strip("._-") or "image"
    unique = uuid.uuid4().hex[:8]
    return f"{safe_prefix}_{int(time.time() * 1000)}_{unique}_{stem}{extension}"
