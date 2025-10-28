from __future__ import annotations

import io
import json
import os
from pathlib import Path

import importlib
import sys
import types
import pytest


def _make_skill(tmp: Path, slug: str = "test-skill", title: str = "Test Skill", desc: str = "A test skill") -> Path:
    skill_dir = tmp / ".claude" / "skills" / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = (
        "---\n"
        f"name: {slug}\n"
        f"title: {title}\n"
        f"description: {desc}\n"
        "---\n\n"
        f"# {title}\n\nBody.\n"
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return skill_dir


@pytest.fixture(autouse=False)
def stub_omicverse_pkg():
    """Stub the 'omicverse' package to avoid running heavy __init__ on import.

    Provides package paths to load submodules directly from the source tree.
    """
    pkgs_root = Path(__file__).resolve().parents[1] / "omicverse"
    # Create lightweight package stubs
    ov = types.ModuleType("omicverse")
    ov.__path__ = [str(pkgs_root)]  # type: ignore[attr-defined]
    utils = types.ModuleType("omicverse.utils")
    utils.__path__ = [str(pkgs_root / "utils")]  # type: ignore[attr-defined]
    seeker = types.ModuleType("omicverse.ov_skill_seeker")
    seeker.__path__ = [str(pkgs_root / "ov_skill_seeker")]  # type: ignore[attr-defined]
    # Attach subpackages as attributes for dotted getattr resolution
    setattr(ov, "utils", utils)
    setattr(ov, "ov_skill_seeker", seeker)
    sys.modules.setdefault("omicverse", ov)
    sys.modules.setdefault("omicverse.utils", utils)
    sys.modules.setdefault("omicverse.ov_skill_seeker", seeker)
    try:
        yield
    finally:
        # Leave stubs in place for performance; they are harmless
        pass


def test_link_builder_writes_skill_and_references(stub_omicverse_pkg, monkeypatch, tmp_path: Path):
    link_builder = importlib.import_module("omicverse.ov_skill_seeker.link_builder")

    # Stub docs scraper to avoid network/deps
    def fake_scrape(url: str, max_pages: int = 30):
        return [("page-001.md", "# Hello\n\nWorld"), ("page-002.md", "Text")]

    monkeypatch.setattr(
        "omicverse.ov_skill_seeker.docs_scraper.scrape", fake_scrape, raising=True
    )

    out = link_builder.build_from_link(
        link="https://example.com/feature",
        output_root=tmp_path,
        name="New Analysis",
        description="Desc",
        max_pages=5,
    )

    assert out.exists()
    assert (out / "SKILL.md").exists()
    # References created from stub
    assert (out / "references" / "page-001.md").read_text(encoding="utf-8").startswith("# Hello")
    assert (out / "references" / "page-002.md").exists()

    # Frontmatter has expected keys regardless of PyYAML
    fm = (out / "SKILL.md").read_text(encoding="utf-8").split("---\n", 2)[1]
    assert "name: new-analysis" in fm
    assert "title: New Analysis" in fm
    assert "description: Desc" in fm


def test_link_builder_handles_scrape_error(stub_omicverse_pkg, monkeypatch, tmp_path: Path):
    link_builder = importlib.import_module("omicverse.ov_skill_seeker.link_builder")

    def boom(url: str, max_pages: int = 30):
        raise RuntimeError("no network")

    monkeypatch.setattr(
        "omicverse.ov_skill_seeker.docs_scraper.scrape", boom, raising=True
    )

    out = link_builder.build_from_link(
        link="https://example.com/feature",
        output_root=tmp_path,
        name="New Analysis",
        max_pages=1,
    )

    err = (out / "references" / "source-error.md").read_text(encoding="utf-8")
    assert "Error scraping" in err
    assert (out / "SKILL.md").exists()


def test_unified_builder_mixed_sources(stub_omicverse_pkg, monkeypatch, tmp_path: Path):
    unified_builder = importlib.import_module("omicverse.ov_skill_seeker.unified_builder")

    # Stub all extractors
    monkeypatch.setattr(
        "omicverse.ov_skill_seeker.docs_scraper.scrape",
        lambda url, max_pages=50: [("docs.md", "D")],
        raising=True,
    )
    monkeypatch.setattr(
        "omicverse.ov_skill_seeker.github_scraper.extract",
        lambda repo: [("github.md", "G")],
        raising=True,
    )
    monkeypatch.setattr(
        "omicverse.ov_skill_seeker.pdf_scraper.extract",
        lambda path: [("pdf.md", "P")],
        raising=True,
    )

    cfg = {
        "name": "My Skill",
        "description": "desc",
        "sources": [
            {"type": "documentation", "base_url": "https://a/", "max_pages": 2},
            {"type": "github", "repo": "owner/repo"},
            {"type": "pdf", "path": str(tmp_path / "dummy.pdf")},
        ],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    out = unified_builder.build_from_config(cfg_path, tmp_path)
    assert out.name == "my-skill"
    refs = {p.name for p in (out / "references").iterdir()}
    assert {"docs.md", "github.md", "pdf.md"} <= refs

    fm = (out / "SKILL.md").read_text(encoding="utf-8").split("---\n", 2)[1]
    assert "name: my-skill" in fm
    assert "title: My Skill" in fm


def test_cli_list_validate_and_package(stub_omicverse_pkg, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cli = importlib.import_module("omicverse.ov_skill_seeker.cli")

    project_root = tmp_path / "proj"
    project_root.mkdir()
    _make_skill(project_root)

    # --list
    rc = cli.main(["--project-root", str(project_root), "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Discovered" in out and "test-skill" in out

    # --validate
    rc = cli.main(["--project-root", str(project_root), "--validate"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Summary:" in out

    # --package single
    out_dir = project_root / "out"
    rc = cli.main([
        "--project-root",
        str(project_root),
        "--package",
        "test-skill",
        "--out-dir",
        str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "test-skill.zip").exists()


def test_cli_create_from_link_offline(stub_omicverse_pkg, monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cli = importlib.import_module("omicverse.ov_skill_seeker.cli")

    # Stub docs scraper
    monkeypatch.setattr(
        "omicverse.ov_skill_seeker.docs_scraper.scrape",
        lambda url, max_pages=30: [("one.md", "ONE")],
        raising=True,
    )

    project_root = tmp_path / "proj"
    project_root.mkdir()

    rc = cli.main([
        "--project-root",
        str(project_root),
        "--create-from-link",
        "https://example.com",
        "--name",
        "Demo",
        "--target",
        "output",
        "--out-dir",
        str(project_root / "output"),
        "--package-after",
    ])
    assert rc == 0
    skill_dir = project_root / "output" / "demo"
    assert skill_dir.exists()
    assert (project_root / "output" / "demo.zip").exists()



def test_config_validator_valid_and_invalid(stub_omicverse_pkg, tmp_path: Path):
    validate_config = importlib.import_module("omicverse.ov_skill_seeker.config_validator").validate_config

    valid = {
        "name": "X",
        "description": "D",
        "sources": [
            {"type": "documentation", "base_url": "https://a"},
            {"type": "github", "repo": "o/r"},
            {"type": "pdf", "path": "/p.pdf"},
        ],
    }
    validate_config(valid)  # does not raise

    # Missing fields
    with pytest.raises(ValueError):
        validate_config({})
    with pytest.raises(ValueError):
        validate_config({"name": "x", "description": "d", "sources": []})
    with pytest.raises(ValueError):
        validate_config({"name": "x", "description": "d", "sources": [{"type": "github"}]})


def test_deprecated_agent_seeker_forwards_to_new_api(stub_omicverse_pkg, tmp_path: Path, monkeypatch):
    """Test that ov.agent.seeker emits deprecation warning and forwards to Agent.seeker."""
    import warnings
    from unittest.mock import MagicMock, patch

    # Import the agent module
    agent_module = importlib.import_module("omicverse.agent")

    # Mock Agent.seeker to avoid actual implementation details
    mock_result = {
        'slug': 'test-skill',
        'skill_dir': str(tmp_path / 'test-skill'),
        'zip': str(tmp_path / 'test-skill.zip')
    }

    with patch('omicverse.utils.smart_agent.Agent.seeker', return_value=mock_result) as mock_agent_seeker:
        # Test that deprecation warning is emitted
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            result = agent_module.seeker("https://example.com", name="Test")

            # Verify deprecation warning was emitted
            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "ov.agent.seeker is deprecated; use ov.Agent.seeker instead" in str(warning_list[0].message)

            # Verify it forwards to Agent.seeker with correct arguments
            mock_agent_seeker.assert_called_once_with(
                "https://example.com",
                name="Test",
                description=None,
                max_pages=30,
                target="skills",
                out_dir=None,
                package=False,
                package_dir=None,
            )

            # Verify the result is returned correctly
            assert result == mock_result


def test_deprecated_agent_seeker_with_all_params(stub_omicverse_pkg, tmp_path: Path):
    """Test that ov.agent.seeker forwards all parameters correctly."""
    import warnings
    from unittest.mock import patch

    agent_module = importlib.import_module("omicverse.agent")

    mock_result = {'slug': 'test', 'skill_dir': 'dir'}

    with patch('omicverse.utils.smart_agent.Agent.seeker', return_value=mock_result) as mock_agent_seeker:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            result = agent_module.seeker(
                ["https://a.com", "https://b.com"],
                name="Multi",
                description="Multi-source skill",
                max_pages=50,
                target="output",
                out_dir=str(tmp_path),
                package=True,
                package_dir=str(tmp_path / "packages")
            )

            # Verify all parameters are forwarded correctly
            mock_agent_seeker.assert_called_once_with(
                ["https://a.com", "https://b.com"],
                name="Multi",
                description="Multi-source skill",
                max_pages=50,
                target="output",
                out_dir=str(tmp_path),
                package=True,
                package_dir=str(tmp_path / "packages"),
            )

            assert result == mock_result
