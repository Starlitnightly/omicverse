"""
Foundation Model conda-subprocess runner.

This script is intended to be executed *inside* a model-specific conda env
via::

    conda run -n fm-<model> python /path/to/_conda_runner.py --payload payload.json

It intentionally avoids importing the top-level ``omicverse`` package (which
may pull in unrelated heavy deps). Instead, it bootstraps minimal package
stubs so we can import fm adapter modules from the repo source tree.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any


def _stub_package(name: str, package_dir: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(package_dir)]
    module.__package__ = name
    module.__file__ = str(package_dir / "__init__.py")
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, is_package=True
    )
    sys.modules[name] = module


def _bootstrap_repo_imports(repo_root: Path) -> None:
    omicverse_dir = repo_root / "omicverse"
    fm_dir = omicverse_dir / "fm"
    adapters_dir = fm_dir / "adapters"

    _stub_package("omicverse", omicverse_dir)
    _stub_package("omicverse.fm", fm_dir)
    _stub_package("omicverse.fm.adapters", adapters_dir)


def _load_adapter(model_name: str, checkpoint_dir: str | None):
    model_name = model_name.lower()

    # Try registry-based resolution first (supports plugins)
    try:
        from omicverse.fm.registry import get_registry
        adapter_cls = get_registry().get_adapter_class(model_name)
        if adapter_cls is not None:
            return adapter_cls(checkpoint_dir)
    except Exception:
        pass  # Fall through to hardcoded mapping

    # Hardcoded fallback for isolated conda envs
    mapping: dict[str, tuple[str, str]] = {
        "uce": ("omicverse.fm.adapters._uce", "UCEAdapter"),
        "scgpt": ("omicverse.fm.adapters._scgpt", "ScGPTAdapter"),
        "geneformer": ("omicverse.fm.adapters._geneformer", "GeneformerAdapter"),
        "scfoundation": ("omicverse.fm.adapters._scfoundation", "ScFoundationAdapter"),
        "cellplm": ("omicverse.fm.adapters._cellplm", "CellPLMAdapter"),
    }

    if model_name not in mapping:
        raise ValueError(f"Unknown model adapter: {model_name}")

    module_name, class_name = mapping[model_name]
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(checkpoint_dir)


def _coerce_none(value: Any) -> Any:
    if value == "" or value == "null":
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path to JSON payload file")
    args = parser.parse_args()

    payload_path = Path(args.payload).expanduser()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    model_name = payload["model_name"]
    task = payload["task"]
    adata_path = payload["adata_path"]
    output_path = payload["output_path"]

    checkpoint_dir = _coerce_none(payload.get("checkpoint_dir"))
    batch_key = _coerce_none(payload.get("batch_key"))
    label_key = _coerce_none(payload.get("label_key"))
    device = _coerce_none(payload.get("device")) or "auto"
    batch_size = payload.get("batch_size")

    # Mark provenance backend
    os.environ.setdefault("OV_FM_BACKEND", "conda-subprocess")

    # Repo root: <repo>/omicverse/fm/_conda_runner.py -> parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    _bootstrap_repo_imports(repo_root)

    try:
        from omicverse.fm.registry import TaskType

        adapter = _load_adapter(model_name=model_name, checkpoint_dir=checkpoint_dir)
        result = adapter.run(
            task=TaskType(task),
            adata_path=adata_path,
            output_path=output_path,
            batch_key=batch_key,
            label_key=label_key,
            device=device,
            batch_size=batch_size or adapter.spec.hardware.default_batch_size,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e), "model": model_name, "task": task}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())