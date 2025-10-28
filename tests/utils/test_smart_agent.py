import asyncio
import importlib.machinery
import importlib.util
import sys
import threading
import types
import warnings
from pathlib import Path
from types import MethodType
import unittest.mock
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

smart_agent_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.smart_agent", PACKAGE_ROOT / "utils" / "smart_agent.py"
)
smart_agent_module = importlib.util.module_from_spec(smart_agent_spec)
sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
assert smart_agent_spec.loader is not None
smart_agent_spec.loader.exec_module(smart_agent_module)

OmicVerseAgent = smart_agent_module.OmicVerseAgent


def _build_agent(return_value):
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    async def _fake_run_async(self, request, adata):
        return {
            "request": request,
            "adata": adata,
            "value": return_value,
        }

    agent.run_async = MethodType(_fake_run_async, agent)
    return agent


def test_run_outside_event_loop(monkeypatch):
    agent = _build_agent(return_value="plain")
    original_asyncio_run = asyncio.run
    run_threads = []

    def _tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_asyncio_run(coro)

    monkeypatch.setattr(asyncio, "run", _tracking_run)

    sentinel = object()
    result = agent.run("plain request", sentinel)

    assert result["request"] == "plain request"
    assert result["adata"] is sentinel
    assert result["value"] == "plain"
    assert run_threads == [threading.current_thread().name]


def test_run_inside_running_loop(monkeypatch):
    agent = _build_agent(return_value="nested")
    original_asyncio_run = asyncio.run
    run_threads = []

    def _tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_asyncio_run(coro)

    async def _caller():
        monkeypatch.setattr(asyncio, "run", _tracking_run)
        sentinel = object()
        result = agent.run("loop request", sentinel)

        assert result["request"] == "loop request"
        assert result["adata"] is sentinel
        assert result["value"] == "nested"
        assert run_threads and run_threads[0] != threading.current_thread().name

    asyncio.run(_caller())


def test_agent_seeker_available():
    """Test that Agent.seeker is available as an attribute."""
    Agent = smart_agent_module.Agent
    assert hasattr(Agent, 'seeker'), "Agent should have seeker attribute"
    assert callable(Agent.seeker), "Agent.seeker should be callable"


def test_agent_seeker_with_mock_dependencies(monkeypatch, tmp_path):
    """Test Agent.seeker with mocked dependencies to avoid network calls."""
    Agent = smart_agent_module.Agent

    # Mock Path objects that builders would return
    mock_skill_dir_single = tmp_path / "test-skill"
    mock_skill_dir_multi = tmp_path / "multi-source-skill"

    # Mock the import dependencies
    def mock_import_side_effect(name, *args, **kwargs):
        if name in ['requests', 'bs4', 'github', 'fitz']:
            return unittest.mock.MagicMock()
        return original_import(name, *args, **kwargs)

    import builtins
    original_import = builtins.__import__

    # Mock the builders to return Path objects
    # build_from_link returns Path to skill_dir
    mock_build_from_link = unittest.mock.MagicMock(return_value=mock_skill_dir_single)
    # build_from_config returns Path to skill_dir
    mock_build_from_config = unittest.mock.MagicMock(return_value=mock_skill_dir_multi)
    # Mock _zip_dir
    mock_zip_dir = unittest.mock.MagicMock(return_value=tmp_path / "test-skill.zip")

    with unittest.mock.patch('builtins.__import__', side_effect=mock_import_side_effect):
        with unittest.mock.patch('omicverse.ov_skill_seeker.link_builder.build_from_link', mock_build_from_link):
            with unittest.mock.patch('omicverse.ov_skill_seeker.unified_builder.build_from_config', mock_build_from_config):
                with unittest.mock.patch('omicverse.agent._zip_dir', mock_zip_dir):
                    # Test single link without packaging
                    result = Agent.seeker("https://example.com", name="Test", package=False)
                    assert result['slug'] == 'test-skill'
                    assert result['skill_dir'] == str(mock_skill_dir_single)
                    assert 'zip' not in result
                    # Verify correct signature usage
                    mock_build_from_link.assert_called_once()
                    call_args = mock_build_from_link.call_args
                    assert call_args[0][0] == "https://example.com"  # link
                    assert 'name' in call_args[1]  # name kwarg
                    assert 'max_pages' in call_args[1]  # max_pages kwarg
                    # Verify no unsupported kwargs
                    assert 'target' not in call_args[1]
                    assert 'out_dir' not in call_args[1]
                    assert 'package' not in call_args[1]
                    assert 'package_dir' not in call_args[1]

                    # Test multiple links
                    mock_build_from_link.reset_mock()
                    mock_build_from_config.reset_mock()
                    result = Agent.seeker(["https://a.com", "https://b.com"], name="Multi")
                    assert result['slug'] == 'multi-source-skill'
                    assert 'zip' not in result
                    mock_build_from_config.assert_called_once()
                    # Verify correct signature: (config_path: Path, output_root: Path)
                    call_args = mock_build_from_config.call_args
                    assert len(call_args[0]) == 2  # Two positional args
                    assert isinstance(call_args[0][0], Path)  # config_path
                    assert isinstance(call_args[0][1], Path)  # output_root


def test_agent_seeker_missing_dependencies():
    """Test Agent.seeker raises appropriate error when optional dependencies are missing."""
    Agent = smart_agent_module.Agent

    # Mock missing dependency
    def mock_import_side_effect(name, *args, **kwargs):
        if name == 'bs4':
            raise ImportError("No module named 'bs4'")
        return original_import(name, *args, **kwargs)

    import builtins
    original_import = builtins.__import__

    with unittest.mock.patch('builtins.__import__', side_effect=mock_import_side_effect):
        with pytest.raises(ImportError, match="Install optional extras: pip install -e .\\[skillseeker\\]"):
            Agent.seeker("https://example.com")


def test_agent_seeker_integration_with_packaging(tmp_path, monkeypatch):
    """Integration test for Agent.seeker that exercises real packaging logic."""
    Agent = smart_agent_module.Agent

    # Create a fake skill directory structure
    skill_name = "test-integration-skill"
    skill_dir = tmp_path / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Test Skill\n\nThis is a test.", encoding="utf-8")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "ref1.md").write_text("Reference content", encoding="utf-8")

    # Mock the import dependencies
    def mock_import_side_effect(name, *args, **kwargs):
        if name in ['requests', 'bs4', 'github', 'fitz']:
            return unittest.mock.MagicMock()
        return original_import(name, *args, **kwargs)

    import builtins
    original_import = builtins.__import__

    # Mock the builders to return our fake skill_dir
    mock_build_from_link = unittest.mock.MagicMock(return_value=skill_dir)

    with unittest.mock.patch('builtins.__import__', side_effect=mock_import_side_effect):
        with unittest.mock.patch('omicverse.ov_skill_seeker.link_builder.build_from_link', mock_build_from_link):
            # Test with packaging enabled
            output_dir = tmp_path / "output"
            result = Agent.seeker(
                "https://example.com/docs",
                name="Test Integration Skill",
                package=True,
                package_dir=str(output_dir)
            )

            # Verify result structure
            assert 'slug' in result
            assert 'skill_dir' in result
            assert 'zip' in result

            # Verify the zip file was created
            zip_path = Path(result['zip'])
            assert zip_path.exists(), f"Zip file should exist at {zip_path}"
            assert zip_path.suffix == '.zip'
            assert zip_path.parent == output_dir

            # Verify zip contents
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                assert 'SKILL.md' in names
                assert 'references/ref1.md' in names
                # Verify content
                assert zf.read('SKILL.md').decode('utf-8') == "# Test Skill\n\nThis is a test."


def test_agent_seeker_integration_without_packaging(tmp_path, monkeypatch):
    """Integration test for Agent.seeker without packaging."""
    Agent = smart_agent_module.Agent

    # Create a fake skill directory
    skill_name = "test-no-package"
    skill_dir = tmp_path / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Test\n", encoding="utf-8")

    # Mock dependencies and builders
    def mock_import_side_effect(name, *args, **kwargs):
        if name in ['requests', 'bs4', 'github', 'fitz']:
            return unittest.mock.MagicMock()
        return original_import(name, *args, **kwargs)

    import builtins
    original_import = builtins.__import__
    mock_build_from_link = unittest.mock.MagicMock(return_value=skill_dir)

    with unittest.mock.patch('builtins.__import__', side_effect=mock_import_side_effect):
        with unittest.mock.patch('omicverse.ov_skill_seeker.link_builder.build_from_link', mock_build_from_link):
            result = Agent.seeker("https://example.com/docs", name="Test", package=False)

            # Verify result structure without zip
            assert 'slug' in result
            assert result['slug'] == skill_name
            assert 'skill_dir' in result
            assert 'zip' not in result
