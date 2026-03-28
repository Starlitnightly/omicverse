"""Tests for subagent override plumbing through runtime bootstrap (task-016).

Covers:
  - SubagentController.run_subagent uses AgentConfig.get_subagent_config
    (not raw SUBAGENT_CONFIGS) so overrides actually reach the runtime.
  - Default explore/plan/execute behavior is unchanged when no overrides.
  - Permission policy and SubagentRuntime honor overridden allowed_tools,
    max_turns, and can_mutate_adata.
  - Tool restriction regressions are caught for both default and override paths.
"""

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: require harness env-var
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Subagent override plumbing tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SAVED = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils"]
}
for name in ["omicverse", "omicverse.utils"]:
    sys.modules.pop(name, None)

_ov_pkg = types.ModuleType("omicverse")
_ov_pkg.__path__ = [str(PACKAGE_ROOT)]
_ov_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = _ov_pkg

_utils_pkg = types.ModuleType("omicverse.utils")
_utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
_utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = _utils_pkg
_ov_pkg.utils = _utils_pkg

from omicverse.utils.agent_config import (
    AgentConfig,
    SUBAGENT_CONFIGS,
    SubagentConfig,
)
from omicverse.utils.ovagent import (
    SubagentController,
    SubagentRuntime,
    ToolRuntime,
    create_subagent_policy,
)
from omicverse.utils.ovagent.prompt_builder import PromptBuilder
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS
from omicverse.utils.ovagent.context_budget import create_subagent_budget_manager

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

class _DummyCtx:
    """Minimal AgentContext stub with configurable AgentConfig."""

    LEGACY_AGENT_TOOLS = LEGACY_AGENT_TOOLS

    def __init__(self, agent_config=None):
        self.model = "test-model"
        self.provider = "openai"
        self.endpoint = "https://api.example.com"
        self.api_key = None
        self._llm = None
        self._config = agent_config or AgentConfig()
        self._security_config = SimpleNamespace()
        self._security_scanner = SimpleNamespace()
        self._filesystem_context = None
        self.skill_registry = None
        self._notebook_executor = None
        self._ov_runtime = None
        self._trace_store = None
        self._session_history = None
        self._context_compactor = None
        self._approval_handler = None
        self._reporter = MagicMock()
        self.last_usage = None
        self.last_usage_breakdown = {}
        self._last_run_trace = None
        self._active_run_id = ""
        self._web_session_id = "test-session"
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False

    def _get_runtime_session_id(self):
        return self._web_session_id

    def _emit(self, level, message, category=""):
        pass

    def _get_harness_session_id(self):
        return self._web_session_id

    def _get_visible_agent_tools(self, *, allowed_names=None):
        return []

    def _get_loaded_tool_names(self):
        return []

    def _refresh_runtime_working_directory(self):
        return "."

    def _tool_blocked_in_plan_mode(self, tool_name):
        return False

    def _detect_repo_root(self, cwd=None):
        return None

    def _resolve_local_path(self, file_path, *, allow_relative=False):
        raise NotImplementedError

    def _ensure_server_tool_mode(self, tool_name):
        pass

    def _request_interaction(self, payload):
        raise NotImplementedError

    def _request_tool_approval(self, tool_name, *, reason, payload):
        raise NotImplementedError

    def _load_skill_guidance(self, slug):
        return ""

    def _extract_python_code(self, text):
        return text

    def _normalize_registry_entry_for_codegen(self, entry):
        return entry

    def _build_agentic_system_prompt(self):
        return "test"

    async def _run_agentic_loop(self, request, adata, **kwargs):
        return adata

    from contextlib import contextmanager

    @contextmanager
    def _temporary_api_keys(self):
        yield


class _DummyExecutor:
    pass


def _make_controller(agent_config=None):
    """Build a SubagentController with a given AgentConfig."""
    ctx = _DummyCtx(agent_config=agent_config)
    pb = PromptBuilder(ctx)
    rt = ToolRuntime(ctx, _DummyExecutor())
    sc = SubagentController(ctx, pb, rt)
    return sc, ctx


# ===================================================================
# 1. Default behaviour unchanged when no overrides
# ===================================================================

class TestDefaultBehaviourUnchanged:

    def test_default_explore_runtime_matches_subagent_configs(self):
        sc, _ = _make_controller()
        rt = sc._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(SUBAGENT_CONFIGS["explore"].allowed_tools),
            max_turns=SUBAGENT_CONFIGS["explore"].max_turns,
            can_mutate_adata=SUBAGENT_CONFIGS["explore"].can_mutate_adata,
        )
        assert rt.agent_type == "explore"
        assert rt.max_turns == 5
        assert rt.can_mutate_adata is False

    def test_default_plan_runtime_matches_subagent_configs(self):
        sc, _ = _make_controller()
        rt = sc._create_subagent_runtime(
            agent_type="plan",
            allowed_tools=frozenset(SUBAGENT_CONFIGS["plan"].allowed_tools),
            max_turns=SUBAGENT_CONFIGS["plan"].max_turns,
            can_mutate_adata=SUBAGENT_CONFIGS["plan"].can_mutate_adata,
        )
        assert rt.max_turns == 8
        assert rt.can_mutate_adata is False

    def test_default_execute_runtime_matches_subagent_configs(self):
        sc, _ = _make_controller()
        rt = sc._create_subagent_runtime(
            agent_type="execute",
            allowed_tools=frozenset(SUBAGENT_CONFIGS["execute"].allowed_tools),
            max_turns=SUBAGENT_CONFIGS["execute"].max_turns,
            can_mutate_adata=SUBAGENT_CONFIGS["execute"].can_mutate_adata,
        )
        assert rt.max_turns == 10
        assert rt.can_mutate_adata is True

    def test_no_override_config_resolves_to_defaults_for_all_types(self):
        cfg = AgentConfig()
        for agent_type, default in SUBAGENT_CONFIGS.items():
            resolved = cfg.get_subagent_config(agent_type)
            assert resolved.max_turns == default.max_turns
            assert resolved.allowed_tools == default.allowed_tools
            assert resolved.can_mutate_adata == default.can_mutate_adata
            assert resolved.temperature == default.temperature


# ===================================================================
# 2. Override plumbing reaches SubagentController._create_subagent_runtime
# ===================================================================

class TestOverridePlumbing:

    def test_overridden_max_turns_reaches_runtime(self):
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 20}})
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("explore")
        rt = sc._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert rt.max_turns == 20

    def test_overridden_can_mutate_adata_reaches_runtime(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"can_mutate_adata": True}
        })
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("explore")
        rt = sc._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert rt.can_mutate_adata is True

    def test_overridden_allowed_tools_restrict_permission_policy(self):
        """When allowed_tools is overridden, the permission policy reflects it."""
        custom_tools = ["inspect_data", "finish"]
        cfg = AgentConfig(subagent_overrides={
            "execute": {"allowed_tools": custom_tools}
        })
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("execute")
        rt = sc._create_subagent_runtime(
            agent_type="execute",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        # Allowed tools should pass
        assert not rt.check_tool_permission("inspect_data").is_denied
        assert not rt.check_tool_permission("finish").is_denied
        # Tools NOT in the override list should be denied
        assert rt.check_tool_permission("execute_code").is_denied
        assert rt.check_tool_permission("web_fetch").is_denied


# ===================================================================
# 3. run_subagent uses get_subagent_config (not raw SUBAGENT_CONFIGS)
# ===================================================================

class TestRunSubagentResolvesConfig:
    """Verify run_subagent reads the effective profile, not the global default.

    We can't easily run the full async loop without an LLM, so we
    monkeypatch _create_subagent_runtime to capture the arguments it
    receives from run_subagent and verify they match the overridden config.
    """

    def test_run_subagent_passes_overridden_values_to_runtime_factory(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 42, "can_mutate_adata": True}
        })
        sc, ctx = _make_controller(agent_config=cfg)

        captured = {}
        original_create = sc._create_subagent_runtime

        def spy_create(*, agent_type, allowed_tools, max_turns, can_mutate_adata=False):
            captured["agent_type"] = agent_type
            captured["allowed_tools"] = allowed_tools
            captured["max_turns"] = max_turns
            captured["can_mutate_adata"] = can_mutate_adata
            return original_create(
                agent_type=agent_type,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                can_mutate_adata=can_mutate_adata,
            )

        sc._create_subagent_runtime = spy_create

        # We need a mock LLM that returns a response with no tool_calls
        mock_response = SimpleNamespace(
            content="done",
            tool_calls=None,
            raw_message=None,
            usage=None,
        )
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=mock_response)
        # Make chat a coroutine
        import asyncio

        async def mock_chat(*args, **kwargs):
            return mock_response

        mock_llm.chat = mock_chat
        ctx._llm = mock_llm

        result = asyncio.run(sc.run_subagent("explore", "test task", None))

        assert captured["max_turns"] == 42
        assert captured["can_mutate_adata"] is True
        assert captured["agent_type"] == "explore"

    def test_run_subagent_default_config_passes_default_values(self):
        """With no overrides, run_subagent passes the SUBAGENT_CONFIGS defaults."""
        cfg = AgentConfig()  # no overrides
        sc, ctx = _make_controller(agent_config=cfg)

        captured = {}
        original_create = sc._create_subagent_runtime

        def spy_create(*, agent_type, allowed_tools, max_turns, can_mutate_adata=False):
            captured["agent_type"] = agent_type
            captured["allowed_tools"] = allowed_tools
            captured["max_turns"] = max_turns
            captured["can_mutate_adata"] = can_mutate_adata
            return original_create(
                agent_type=agent_type,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                can_mutate_adata=can_mutate_adata,
            )

        sc._create_subagent_runtime = spy_create

        mock_response = SimpleNamespace(
            content="done", tool_calls=None, raw_message=None, usage=None,
        )

        import asyncio

        async def mock_chat(*args, **kwargs):
            return mock_response

        mock_llm = MagicMock()
        mock_llm.chat = mock_chat
        ctx._llm = mock_llm

        asyncio.run(sc.run_subagent("plan", "test task", None))

        default = SUBAGENT_CONFIGS["plan"]
        assert captured["max_turns"] == default.max_turns
        assert captured["can_mutate_adata"] == default.can_mutate_adata
        assert captured["allowed_tools"] == frozenset(default.allowed_tools)

    def test_run_subagent_overridden_allowed_tools_flow_to_runtime(self):
        """Overridden allowed_tools reach the runtime factory."""
        custom = ["inspect_data", "finish"]
        cfg = AgentConfig(subagent_overrides={
            "execute": {"allowed_tools": custom}
        })
        sc, ctx = _make_controller(agent_config=cfg)

        captured = {}
        original_create = sc._create_subagent_runtime

        def spy_create(*, agent_type, allowed_tools, max_turns, can_mutate_adata=False):
            captured["allowed_tools"] = allowed_tools
            return original_create(
                agent_type=agent_type,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                can_mutate_adata=can_mutate_adata,
            )

        sc._create_subagent_runtime = spy_create

        mock_response = SimpleNamespace(
            content="done", tool_calls=None, raw_message=None, usage=None,
        )

        import asyncio

        async def mock_chat(*args, **kwargs):
            return mock_response

        mock_llm = MagicMock()
        mock_llm.chat = mock_chat
        ctx._llm = mock_llm

        asyncio.run(sc.run_subagent("execute", "test task", None))

        assert captured["allowed_tools"] == frozenset(custom)


# ===================================================================
# 4. Permission policy covers both default and override paths
# ===================================================================

class TestPermissionPolicyDefaultVsOverride:

    def test_default_explore_denies_execute_code(self):
        """Default explore subagent does NOT have execute_code."""
        cfg = AgentConfig()
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("explore")
        rt = sc._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert rt.check_tool_permission("execute_code").is_denied

    def test_default_execute_allows_execute_code(self):
        """Default execute subagent DOES have execute_code."""
        cfg = AgentConfig()
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("execute")
        rt = sc._create_subagent_runtime(
            agent_type="execute",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert not rt.check_tool_permission("execute_code").is_denied

    def test_overriding_explore_to_add_execute_code(self):
        """If explore overrides gain execute_code, policy allows it."""
        explore_tools = list(SUBAGENT_CONFIGS["explore"].allowed_tools) + ["execute_code"]
        cfg = AgentConfig(subagent_overrides={
            "explore": {"allowed_tools": explore_tools}
        })
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("explore")
        rt = sc._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert not rt.check_tool_permission("execute_code").is_denied

    def test_overriding_execute_to_remove_execute_code(self):
        """If execute overrides drop execute_code, policy denies it."""
        restricted_tools = ["inspect_data", "run_snippet", "finish"]
        cfg = AgentConfig(subagent_overrides={
            "execute": {"allowed_tools": restricted_tools}
        })
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("execute")
        rt = sc._create_subagent_runtime(
            agent_type="execute",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert rt.check_tool_permission("execute_code").is_denied
        assert not rt.check_tool_permission("inspect_data").is_denied


# ===================================================================
# 5. Runtime isolation properties preserved under overrides
# ===================================================================

class TestRuntimeIsolationWithOverrides:

    def test_overridden_runtime_does_not_share_parent_usage(self):
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 30}})
        ctx = _DummyCtx(agent_config=cfg)
        ctx.last_usage = {"prompt_tokens": 999}

        resolved = cfg.get_subagent_config("explore")
        policy = create_subagent_policy(
            ToolRuntime(ctx, _DummyExecutor()).registry,
            allowed_tools=frozenset(resolved.allowed_tools),
        )
        rt = SubagentRuntime(
            agent_type="explore",
            max_turns=resolved.max_turns,
            permission_policy=policy,
            budget_manager=create_subagent_budget_manager(model="test"),
        )
        rt.record_usage({"prompt_tokens": 42})

        assert ctx.last_usage == {"prompt_tokens": 999}
        assert rt.last_usage == {"prompt_tokens": 42}
        assert rt.max_turns == 30

    def test_overridden_runtime_tool_schemas_are_snapshot(self):
        cfg = AgentConfig(subagent_overrides={"plan": {"max_turns": 15}})
        sc, _ = _make_controller(agent_config=cfg)
        resolved = cfg.get_subagent_config("plan")
        rt = sc._create_subagent_runtime(
            agent_type="plan",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        original_len = len(rt.tool_schemas)
        rt.tool_schemas.append({"name": "injected"})
        # A new runtime gets a fresh list
        rt2 = sc._create_subagent_runtime(
            agent_type="plan",
            allowed_tools=frozenset(resolved.allowed_tools),
            max_turns=resolved.max_turns,
            can_mutate_adata=resolved.can_mutate_adata,
        )
        assert len(rt2.tool_schemas) == original_len


# ===================================================================
# 6. No public regression in subagent tool restrictions
# ===================================================================

class TestToolRestrictionRegression:
    """Ensure default tool sets are exactly as specified in SUBAGENT_CONFIGS."""

    def test_explore_default_tool_set(self):
        cfg = AgentConfig()
        resolved = cfg.get_subagent_config("explore")
        expected = {
            "inspect_data", "run_snippet", "search_functions",
            "web_fetch", "web_search", "finish",
        }
        assert set(resolved.allowed_tools) == expected

    def test_plan_default_tool_set(self):
        cfg = AgentConfig()
        resolved = cfg.get_subagent_config("plan")
        expected = {
            "inspect_data", "run_snippet", "search_functions",
            "search_skills", "web_fetch", "web_search", "finish",
        }
        assert set(resolved.allowed_tools) == expected

    def test_execute_default_tool_set(self):
        cfg = AgentConfig()
        resolved = cfg.get_subagent_config("execute")
        expected = {
            "inspect_data", "execute_code", "run_snippet",
            "search_functions", "web_fetch", "web_search",
            "web_download", "finish",
        }
        assert set(resolved.allowed_tools) == expected

    def test_explore_default_cannot_mutate(self):
        cfg = AgentConfig()
        assert cfg.get_subagent_config("explore").can_mutate_adata is False

    def test_plan_default_cannot_mutate(self):
        cfg = AgentConfig()
        assert cfg.get_subagent_config("plan").can_mutate_adata is False

    def test_execute_default_can_mutate(self):
        cfg = AgentConfig()
        assert cfg.get_subagent_config("execute").can_mutate_adata is True

    def test_global_defaults_not_mutated_after_override_resolution(self):
        """Resolving overrides must not contaminate module-level SUBAGENT_CONFIGS."""
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 999, "can_mutate_adata": True},
            "plan": {"allowed_tools": ["finish"]},
            "execute": {"temperature": 0.99},
        })
        for agent_type in SUBAGENT_CONFIGS:
            cfg.get_subagent_config(agent_type)

        assert SUBAGENT_CONFIGS["explore"].max_turns == 5
        assert SUBAGENT_CONFIGS["explore"].can_mutate_adata is False
        assert len(SUBAGENT_CONFIGS["plan"].allowed_tools) > 1
        assert SUBAGENT_CONFIGS["execute"].temperature == 0.1
