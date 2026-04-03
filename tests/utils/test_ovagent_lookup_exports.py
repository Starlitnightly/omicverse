"""Tests for ``omicverse.utils._ovagent_lookup`` compatibility helpers."""

import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_MODULES = {
    name: mod
    for name, mod in list(sys.modules.items())
    if name == "omicverse" or name.startswith("omicverse.")
}

for _mod_name, _mod_path in [
    ("omicverse", PACKAGE_ROOT),
    ("omicverse.utils", PACKAGE_ROOT / "utils"),
    ("omicverse.utils.ovagent", PACKAGE_ROOT / "utils" / "ovagent"),
]:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        _stub.__path__ = [str(_mod_path)]
        _stub.__spec__ = importlib.machinery.ModuleSpec(
            _mod_name, loader=None, is_package=True
        )
        sys.modules[_mod_name] = _stub

if "omicverse" in sys.modules and "omicverse.utils" in sys.modules:
    sys.modules["omicverse"].utils = sys.modules["omicverse.utils"]
if "omicverse.utils" in sys.modules and "omicverse.utils.ovagent" in sys.modules:
    sys.modules["omicverse.utils"].ovagent = sys.modules["omicverse.utils.ovagent"]

_module_path = PACKAGE_ROOT / "utils" / "_ovagent_lookup.py"
_spec = importlib.util.spec_from_file_location(
    "omicverse.utils._ovagent_lookup", _module_path
)
_lookup_mod = importlib.util.module_from_spec(_spec)
sys.modules["omicverse.utils._ovagent_lookup"] = _lookup_mod
assert _spec.loader is not None
_spec.loader.exec_module(_lookup_mod)


def teardown_module(module):
    for name in list(sys.modules):
        if name == "omicverse" or name.startswith("omicverse."):
            sys.modules.pop(name, None)
    for name, mod in _ORIGINAL_MODULES.items():
        if mod is not None:
            sys.modules[name] = mod


def test_registry_lookup_delegates_with_scanner_context(monkeypatch):
    class _FakeScanner:
        def collect_static_entries(self, request, max_entries=8):
            return [{"full_name": "ov.pp.qc", "request": request, "limit": max_entries}]

    def _fake_create_registry_scanner():
        return _FakeScanner()

    def _fake_delegate(ctx, query):
        entries = ctx._collect_static_registry_entries(query, max_entries=3)
        return json.dumps({"query": query, "entries": entries})

    monkeypatch.setattr(_lookup_mod, "_create_registry_scanner", _fake_create_registry_scanner)
    monkeypatch.setattr(_lookup_mod, "_delegate_registry_lookup", _fake_delegate)

    result = _lookup_mod.registry_lookup("qc pipeline")
    payload = json.loads(result)

    assert payload["query"] == "qc pipeline"
    assert payload["entries"][0]["full_name"] == "ov.pp.qc"


def test_registry_lookup_returns_fallback_on_scanner_error(monkeypatch):
    def _boom():
        raise RuntimeError("scanner unavailable")

    monkeypatch.setattr(_lookup_mod, "_create_registry_scanner", _boom)

    result = _lookup_mod.registry_lookup("qc")

    assert "RegistryScanner not available" in result


def test_skill_lookup_uses_initialized_registry_and_max_chars(monkeypatch):
    class _FakeSkillDefinition:
        name = "Skill A"
        description = "desc"
        path = Path("/tmp/skill-a")
        metadata = {"x": "1"}

        def prompt_instructions(self, max_chars=4000, provider=None):
            return f"max_chars={max_chars};provider={provider}"

    class _FakeRegistry:
        skill_metadata = {"skill-a": object()}

        def load_full_skill(self, slug):
            assert slug == "skill-a"
            return _FakeSkillDefinition()

    def _fake_create_skill_registry():
        return _FakeRegistry()

    def _fake_delegate(ctx, query):
        return ctx._load_skill_guidance(query)

    monkeypatch.setattr(_lookup_mod, "_create_skill_registry", _fake_create_skill_registry)
    monkeypatch.setattr(_lookup_mod, "_delegate_skill_lookup", _fake_delegate)

    result = _lookup_mod.skill_lookup("skill-a", max_body_chars=123)
    payload = json.loads(result)

    assert payload["name"] == "Skill A"
    assert payload["instructions"] == "max_chars=123;provider=None"
