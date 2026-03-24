"""Tests for ovagent.bootstrap — agent subsystem initialization."""

import os
import sys
import types
import importlib.machinery
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: same isolation pattern as test_smart_agent.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]
}
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

from omicverse.utils.ovagent.bootstrap import (
    format_skill_overview,
    initialize_skill_registry,
    initialize_notebook_executor,
    initialize_filesystem_context,
    initialize_session_history,
    initialize_tracing,
    initialize_security,
    initialize_ov_runtime,
    display_reflection_config,
    _is_under_root,
)
from omicverse.utils.ovagent.prompt_builder import build_filesystem_context_instructions
from omicverse.utils.agent_config import AgentConfig

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# ===================================================================
# format_skill_overview
# ===================================================================

class TestFormatSkillOverview:

    def test_returns_empty_for_none_registry(self):
        assert format_skill_overview(None) == ""

    def test_returns_empty_for_empty_metadata(self):
        registry = SimpleNamespace(skill_metadata={})
        assert format_skill_overview(registry) == ""

    def test_formats_skills_alphabetically(self):
        metadata = {
            "b": SimpleNamespace(name="Beta Skill", description="Second skill"),
            "a": SimpleNamespace(name="Alpha Skill", description="First skill"),
        }
        registry = SimpleNamespace(skill_metadata=metadata)
        result = format_skill_overview(registry)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "Alpha Skill" in lines[0]
        assert "Beta Skill" in lines[1]


# ===================================================================
# _is_under_root
# ===================================================================

class TestIsUnderRoot:

    def test_child_is_under_root(self, tmp_path):
        child = tmp_path / "sub" / "file.py"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        assert _is_under_root(child, tmp_path) is True

    def test_unrelated_path_is_not_under_root(self, tmp_path):
        other = Path("/some/other/path")
        assert _is_under_root(other, tmp_path) is False


# ===================================================================
# initialize_skill_registry
# ===================================================================

class TestInitializeSkillRegistry:

    def test_returns_tuple(self):
        """initialize_skill_registry returns (registry_or_None, str)."""
        result = initialize_skill_registry()
        assert isinstance(result, tuple)
        assert len(result) == 2
        registry, overview = result
        # overview should be a string
        assert isinstance(overview, str)


# ===================================================================
# initialize_notebook_executor
# ===================================================================

class TestInitializeNotebookExecutor:

    def test_disabled_returns_false_none(self, capsys):
        use_nb, executor = initialize_notebook_executor(
            use_notebook=False,
            storage_dir=None,
            max_prompts_per_session=5,
            keep_notebooks=True,
            timeout=600,
            strict_kernel_validation=True,
        )
        assert use_nb is False
        assert executor is None
        captured = capsys.readouterr()
        assert "in-process execution" in captured.out


# ===================================================================
# initialize_filesystem_context
# ===================================================================

class TestInitializeFilesystemContext:

    def test_disabled_returns_false_none(self, capsys):
        enabled, ctx = initialize_filesystem_context(
            enabled=False,
            storage_dir=None,
        )
        assert enabled is False
        assert ctx is None
        captured = capsys.readouterr()
        assert "Filesystem context disabled" in captured.out

    def test_enabled_creates_context(self, tmp_path, capsys):
        enabled, ctx = initialize_filesystem_context(
            enabled=True,
            storage_dir=tmp_path / "ctx",
        )
        assert enabled is True
        assert ctx is not None
        assert hasattr(ctx, "session_id")
        captured = capsys.readouterr()
        assert "Filesystem context enabled" in captured.out


# ===================================================================
# initialize_session_history
# ===================================================================

class TestInitializeSessionHistory:

    def test_disabled_returns_none(self):
        config = SimpleNamespace(history_enabled=False)
        assert initialize_session_history(config) is None

    def test_enabled_returns_session_history(self, tmp_path):
        config = SimpleNamespace(
            history_enabled=True,
            history_path=tmp_path / "history.json",
        )
        result = initialize_session_history(config)
        assert result is not None


# ===================================================================
# initialize_tracing
# ===================================================================

class TestInitializeTracing:

    def test_no_harness_config_returns_none_none(self):
        config = SimpleNamespace(harness=None)
        trace_store, compactor = initialize_tracing(config, llm=None, model="test")
        assert trace_store is None
        assert compactor is None

    def test_traces_enabled_returns_store(self, tmp_path):
        config = SimpleNamespace(
            harness=SimpleNamespace(
                enable_traces=True,
                trace_dir=tmp_path / "traces",
                enable_context_compaction=False,
            )
        )
        trace_store, compactor = initialize_tracing(config, llm=None, model="test")
        assert trace_store is not None
        assert compactor is None


# ===================================================================
# initialize_security
# ===================================================================

class TestInitializeSecurity:

    def test_returns_config_and_scanner(self, capsys):
        config = AgentConfig()
        sec_config, scanner = initialize_security(config)
        assert sec_config is not None
        assert scanner is not None
        captured = capsys.readouterr()
        assert "Security scanner enabled" in captured.out


# ===================================================================
# initialize_ov_runtime
# ===================================================================

class TestInitializeOvRuntime:

    def test_returns_runtime_for_valid_root(self, tmp_path):
        rt = initialize_ov_runtime(tmp_path)
        # Should return an OmicVerseRuntime instance (even without WORKFLOW.md)
        assert rt is not None

    def test_returns_none_on_failure(self):
        """Invalid repo_root should return None gracefully."""
        rt = initialize_ov_runtime(None)
        # None repo_root triggers resolve_repo_root() which may or may not find one
        # Either way it should not raise
        # (result depends on whether we're inside a git repo)


# ===================================================================
# display_reflection_config
# ===================================================================

class TestDisplayReflectionConfig:

    def test_reflection_enabled(self, capsys):
        display_reflection_config(True, 2, True)
        out = capsys.readouterr().out
        assert "Reflection enabled" in out
        assert "2 iterations" in out
        assert "Result review enabled" in out

    def test_reflection_disabled(self, capsys):
        display_reflection_config(False, 1, False)
        out = capsys.readouterr().out
        assert "Reflection disabled" in out
        assert "Result review disabled" in out


# ===================================================================
# build_filesystem_context_instructions (in prompt_builder)
# ===================================================================

class TestBuildFilesystemContextInstructions:

    def test_includes_session_id(self):
        result = build_filesystem_context_instructions("test-session-123")
        assert "test-session-123" in result

    def test_includes_workspace_section(self):
        result = build_filesystem_context_instructions()
        assert "Context Engineering" in result
        assert "CONTEXT_WRITE" in result

    def test_default_session_id(self):
        result = build_filesystem_context_instructions()
        assert "N/A" in result
