import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
OMICVERSE_ROOT = ROOT / "omicverse"
UTILS_ROOT = OMICVERSE_ROOT / "utils"
HARNESS_ROOT = UTILS_ROOT / "harness"
WEB_ROOT = ROOT / "omicverse_web"
SERVICES_ROOT = WEB_ROOT / "services"


def _skip_missing_submodule(reason: str) -> None:
    pytest.skip(
        f"{reason} Clone with --recurse-submodules or enable submodules in CI checkout.",
        allow_module_level=True,
    )


def ensure_web_checkout() -> Path:
    if not WEB_ROOT.exists():
        _skip_missing_submodule("omicverse_web submodule is not checked out.")
    init_path = WEB_ROOT / "__init__.py"
    if not init_path.exists():
        _skip_missing_submodule("omicverse_web package files are missing.")
    return WEB_ROOT


def ensure_service_path(filename: str) -> Path:
    ensure_web_checkout()
    path = SERVICES_ROOT / filename
    if not path.exists():
        _skip_missing_submodule(f"Required omicverse_web service file is missing: {filename}.")
    return path


def register_web_namespace_packages() -> None:
    ensure_web_checkout()
    omicverse_pkg = types.ModuleType("omicverse")
    omicverse_pkg.__path__ = [str(OMICVERSE_ROOT)]
    utils_pkg = types.ModuleType("omicverse.utils")
    utils_pkg.__path__ = [str(UTILS_ROOT)]
    sys.modules.setdefault("omicverse", omicverse_pkg)
    sys.modules.setdefault("omicverse.utils", utils_pkg)
    web_pkg = types.ModuleType("omicverse_web")
    web_pkg.__path__ = [str(WEB_ROOT)]
    services_pkg = types.ModuleType("omicverse_web.services")
    services_pkg.__path__ = [str(SERVICES_ROOT)]
    sys.modules.setdefault("omicverse_web", web_pkg)
    sys.modules.setdefault("omicverse_web.services", services_pkg)
    harness_path = HARNESS_ROOT / "__init__.py"
    if "omicverse.utils.harness" not in sys.modules and harness_path.exists():
        spec = importlib.util.spec_from_file_location(
            "omicverse.utils.harness",
            harness_path,
            submodule_search_locations=[str(HARNESS_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["omicverse.utils.harness"] = module
        spec.loader.exec_module(module)


def load_service_module(module_name: str, filename: str):
    register_web_namespace_packages()
    path = ensure_service_path(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_web_module(module_name: str):
    ensure_web_checkout()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "omicverse_web" or (exc.name and exc.name.startswith("omicverse_web.")):
            _skip_missing_submodule(f"Required module is unavailable: {module_name}.")
        raise
