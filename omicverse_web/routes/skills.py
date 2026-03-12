"""
Skills Routes - Skill Store API Endpoints
=========================================
Expose the same skill catalog that ``ov.Agent`` discovers at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import omicverse
from flask import Blueprint, jsonify, request

from omicverse.utils.skill_registry import build_multi_path_skill_registry


bp = Blueprint("skills", __name__)


def _package_root() -> Path:
    return Path(omicverse.__file__).resolve().parent.parent


def _workspace_root() -> Path:
    return Path.cwd().resolve()


def _builtin_skill_root() -> Path:
    return (_package_root() / ".claude" / "skills").resolve()


def _workspace_skill_root() -> Path:
    return (_workspace_root() / ".claude" / "skills").resolve()


def _allowed_roots() -> List[Tuple[str, Path]]:
    return [
        ("Built-in", _builtin_skill_root()),
        ("Workspace", _workspace_skill_root()),
    ]


def _resolve_skill_path(raw_path: str) -> Path:
    candidate = Path(raw_path or "").expanduser().resolve()
    for _, root in _allowed_roots():
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise ValueError("Invalid skill path")


def _parse_frontmatter(text: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    lines = (text or "").splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return payload
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip().lower()] = value.strip().strip('"').strip("'")
    return payload


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip()).strip("-_.")
    return text.lower() or "custom-skill"


def _read_text_if_exists(path: Path | None) -> str:
    if not path or not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _reference_file(skill_dir: Path) -> Path:
    return skill_dir / "reference.md"


def _is_builtin_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(_builtin_skill_root())
        return True
    except ValueError:
        return False


def _is_workspace_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(_workspace_skill_root())
        return True
    except ValueError:
        return False


def _is_editable_path(path: Path) -> bool:
    resolved = path.resolve()
    for _, root in _allowed_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _markdown_excerpt(text: str, limit: int = 260) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^[#>*\-\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _default_skill_template(name: str, description: str) -> str:
    skill_name = (name or "Custom Skill").strip() or "Custom Skill"
    skill_desc = (description or "Describe what this skill does.").strip() or "Describe what this skill does."
    return (
        "---\n"
        f'name: {skill_name}\n'
        f'description: "{skill_desc}"\n'
        "---\n\n"
        f"# {skill_name}\n\n"
        "## Overview\n\n"
        "Describe when to use this skill.\n\n"
        "## Workflow\n\n"
        "1. Define the task.\n"
        "2. Explain the constraints.\n"
        "3. Execute the specialized steps.\n"
    )


def _skill_entry_from_metadata(slug: str, metadata) -> Dict[str, object]:
    skill_dir = Path(metadata.path).resolve()
    skill_file = skill_dir / "SKILL.md"
    reference_file = _reference_file(skill_dir)
    reference_content = _read_text_if_exists(reference_file)
    root_label = "Workspace"
    root_path = _workspace_skill_root()
    editable = _is_editable_path(skill_file)
    try:
        skill_dir.relative_to(_builtin_skill_root())
        root_label = "Built-in"
        root_path = _builtin_skill_root()
    except ValueError:
        pass

    try:
        rel_path = str(skill_file.relative_to(root_path))
    except ValueError:
        rel_path = skill_file.name

    return {
        "name": metadata.name,
        "slug": slug,
        "description": metadata.description,
        "version": "",
        "path": str(skill_file),
        "filename": skill_file.name,
        "directory": str(skill_dir),
        "relative_path": rel_path,
        "root_label": root_label,
        "editable": editable,
        "updated_at": int(skill_file.stat().st_mtime) if skill_file.exists() else 0,
        "reference_path": str(reference_file) if reference_file.exists() else "",
        "reference_relative_path": str(reference_file.relative_to(root_path)) if reference_file.exists() else "",
        "reference_excerpt": _markdown_excerpt(reference_content),
    }


@bp.route("/list", methods=["GET"])
def list_skills():
    registry = build_multi_path_skill_registry(_package_root(), _workspace_root())
    items = [
        _skill_entry_from_metadata(slug, metadata)
        for slug, metadata in sorted(registry.skill_metadata.items(), key=lambda item: item[0].lower())
    ]
    return jsonify(
        {
            "skills": items,
            "workspace_root": str(_workspace_skill_root()),
            "builtin_root": str(_builtin_skill_root()),
        }
    )


@bp.route("/open", methods=["POST"])
def open_skill():
    payload = request.get_json(silent=True) or {}
    raw_path = str(payload.get("path") or "")
    if not raw_path:
        return jsonify({"error": "Missing skill path"}), 400
    try:
        skill_file = _resolve_skill_path(raw_path)
    except ValueError:
        return jsonify({"error": "Invalid skill path"}), 400
    if not skill_file.exists() or not skill_file.is_file():
        return jsonify({"error": "Skill file not found"}), 404

    text = skill_file.read_text(encoding="utf-8", errors="ignore")
    meta = _parse_frontmatter(text)
    reference_file = _reference_file(skill_file.parent)
    reference_content = _read_text_if_exists(reference_file)
    editable = _is_editable_path(skill_file)

    return jsonify(
        {
            "name": meta.get("name") or skill_file.parent.name,
            "filename": skill_file.name,
            "path": str(skill_file),
            "content": text,
            "type": "skill",
            "editable": editable,
            "reference_path": str(reference_file) if reference_file.exists() else "",
            "reference_content": reference_content,
        }
    )


@bp.route("/open_reference", methods=["POST"])
def open_reference():
    payload = request.get_json(silent=True) or {}
    raw_path = str(payload.get("path") or "")
    if not raw_path:
        return jsonify({"error": "Missing skill path"}), 400
    try:
        source_path = _resolve_skill_path(raw_path)
    except ValueError:
        return jsonify({"error": "Invalid skill path"}), 400

    skill_dir = source_path.parent if source_path.suffix.lower() == ".md" else source_path
    reference_file = _reference_file(skill_dir)
    editable = _is_editable_path(reference_file)

    if not reference_file.exists() and editable:
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        reference_file.write_text("", encoding="utf-8")

    content = _read_text_if_exists(reference_file)
    return jsonify(
        {
            "name": "reference.md",
            "filename": "reference.md",
            "path": str(reference_file),
            "content": content,
            "type": "skill",
            "editable": editable,
        }
    )


@bp.route("/save", methods=["POST"])
def save_skill():
    payload = request.get_json(silent=True) or {}
    raw_path = str(payload.get("path") or "")
    content = payload.get("content")
    if not raw_path or not isinstance(content, str):
        return jsonify({"error": "Missing skill path or content"}), 400
    try:
        skill_file = _resolve_skill_path(raw_path)
    except ValueError:
        return jsonify({"error": "Invalid skill path"}), 400

    if not _is_editable_path(skill_file):
        return jsonify({"error": "Skill path is not editable"}), 403

    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(content, encoding="utf-8")
    return jsonify({"success": True})


@bp.route("/create", methods=["POST"])
def create_skill():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()
    slug = _slugify(str(payload.get("slug") or name))
    description = str(payload.get("description") or "").strip()
    content = payload.get("content")
    if not name:
        return jsonify({"error": "Missing skill name"}), 400
    if content is not None and not isinstance(content, str):
        return jsonify({"error": "Invalid skill content"}), 400

    root = _workspace_skill_root()
    root.mkdir(parents=True, exist_ok=True)
    skill_dir = (root / slug).resolve()
    try:
        skill_dir.relative_to(root)
    except ValueError:
        return jsonify({"error": "Invalid skill destination"}), 400
    if skill_dir.exists():
        return jsonify({"error": "Skill already exists"}), 400

    skill_dir.mkdir(parents=True, exist_ok=False)
    skill_file = skill_dir / "SKILL.md"
    reference_file = _reference_file(skill_dir)
    body = content if isinstance(content, str) and content.strip() else _default_skill_template(name, description)
    skill_file.write_text(body, encoding="utf-8")
    reference_file.write_text("", encoding="utf-8")
    return jsonify(
        {
            "success": True,
            "skill": {
                "name": name,
                "slug": slug,
                "description": description,
                "version": "",
                "path": str(skill_file),
                "filename": skill_file.name,
                "directory": str(skill_dir),
                "relative_path": str(skill_file.relative_to(root)),
                "root_label": "Workspace",
                "editable": True,
                "updated_at": int(skill_file.stat().st_mtime),
                "reference_path": str(reference_file),
                "reference_relative_path": str(reference_file.relative_to(root)),
                "reference_excerpt": "",
            },
        }
    )
