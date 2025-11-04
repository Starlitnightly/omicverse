import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "omicverse" / "utils" / "model_config.py"

spec = importlib.util.spec_from_file_location("omicverse.utils.model_config", MODULE_PATH)
model_config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model_config)


def test_normalize_known_alias():
    normalized = model_config.ModelConfig.normalize_model_id("Claude-Opus-4")
    assert normalized == "anthropic/claude-opus-4-20250514"


def test_supported_models_ascii_fallback(monkeypatch):
    monkeypatch.setattr(model_config, "_supports_unicode_output", lambda: False)
    listing = model_config.ModelConfig.list_supported_models()
    assert "Usage:" in listing
    assert "💡" not in listing
