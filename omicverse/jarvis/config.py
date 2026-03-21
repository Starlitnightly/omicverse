"""
Persistent configuration helpers for Jarvis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def default_state_dir() -> Path:
    return Path(os.path.expanduser("~/.ovjarvis"))


def default_config_path() -> Path:
    return default_state_dir() / "config.json"


def default_auth_path() -> Path:
    return default_state_dir() / "auth.json"


def default_config() -> Dict[str, Any]:
    return {
        "channel": "telegram",
        "model": "claude-sonnet-4-6",
        "llm_provider": None,
        "auth_mode": "environment",
        "endpoint": None,
        "setup_language": "en",
        "session_dir": None,
        "max_prompts": 0,
        "telegram": {
            "token": None,
            "allowed_users": [],
        },
        "discord": {
            "token": None,
        },
        "feishu": {
            "app_id": None,
            "app_secret": None,
            "connection_mode": "websocket",
            "verification_token": None,
            "encrypt_key": None,
            "host": "0.0.0.0",
            "port": 8080,
            "path": "/feishu/events",
        },
        "imessage": {
            "cli_path": "imsg",
            "db_path": os.path.expanduser("~/Library/Messages/chat.db"),
            "include_attachments": False,
        },
        "qq": {
            "app_id": None,
            "client_secret": None,
            "image_host": None,
            "image_server_port": 8081,
            "markdown": False,
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    _ensure_parent(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = path or default_config_path()
    return _deep_merge(default_config(), _load_json(config_path))


def save_config(config: Dict[str, Any], path: Optional[Path] = None) -> Path:
    payload = _deep_merge(default_config(), config)
    return _write_json(path or default_config_path(), payload)


def config_exists(path: Optional[Path] = None) -> bool:
    return (path or default_config_path()).exists()


def load_auth(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_json(path or default_auth_path())


def save_auth(auth: Dict[str, Any], path: Optional[Path] = None) -> Path:
    return _write_json(path or default_auth_path(), auth)


def auth_exists(path: Optional[Path] = None) -> bool:
    return (path or default_auth_path()).exists()


def load_codex_auth() -> Dict[str, Any]:
    return _load_json(Path(os.path.expanduser("~/.codex/auth.json")))
