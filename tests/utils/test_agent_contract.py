"""
Contract regression tests for smart_agent.py public API.

These tests freeze the public interface and behavioral contracts that must
survive any refactor.  They are intentionally lightweight: no real LLM calls,
no heavy imports.  Each test targets one row in docs/harness/contract-matrix.md.

Guarded contracts:
  - Agent factory
  - OmicVerseAgent.run_async / stream_async
  - OmicVerseAgent.restart_session / get_session_history
  - Workflow injection via OmicVerseRuntime.compose_system_prompt
  - Claw downstream integration seam
  - list_supported_models module function
"""

import asyncio
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load smart_agent module in isolation (same pattern as
# test_smart_agent.py) to avoid pulling heavy omicverse dependencies.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

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

Agent = smart_agent_module.Agent
OmicVerseAgent = smart_agent_module.OmicVerseAgent
list_supported_models = smart_agent_module.list_supported_models

from omicverse.utils.harness import build_stream_event
from omicverse.utils.ovagent.runtime import OmicVerseRuntime
from omicverse.utils.ovagent.workflow import (
    WorkflowConfig,
    WorkflowDocument,
    load_workflow_document,
)
from omicverse.utils.ovagent.prompt_templates import (
    PromptTemplateEngine,
    build_skill_layer,
    IDENTITY_BLOCK,
    TOOL_CATALOG_BLOCK,
    CODE_QUALITY_BLOCK,
    DELEGATION_BLOCK,
    CLAW_CODE_ONLY_BLOCK,
    EXPLORE_IDENTITY_BLOCK,
    EXPLORE_REPORT_BLOCK,
)
from omicverse.utils.ovagent.prompt_builder import PromptBuilder, CODE_QUALITY_RULES
from omicverse.utils.ovagent.contracts import (
    PromptLayer,
    PromptLayerKind,
    PromptComposer,
)

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_agent(**overrides):
    """Create an OmicVerseAgent without running __init__."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    defaults = {
        "model": "test-model",
        "provider": "openai",
        "use_notebook_execution": False,
        "_notebook_executor": None,
        "_last_run_trace": None,
        "_session_history": None,
        "_trace_store": None,
        "_context_compactor": None,
        "_approval_handler": None,
        "_code_only_mode": False,
        "last_usage": None,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(agent, key, value)
    return agent


# ===================================================================
# 1. Agent Factory
# ===================================================================

class TestAgentFactory:

    def test_agent_factory_returns_omicverseagent(self):
        """Agent() is a thin wrapper that returns an OmicVerseAgent instance."""
        assert callable(Agent)
        sig = inspect.signature(Agent)
        # Must accept 'model' as the first parameter
        params = list(sig.parameters.keys())
        assert params[0] == "model"
        # Return annotation must be OmicVerseAgent
        assert sig.return_annotation is OmicVerseAgent

    def test_agent_factory_signature_matches_init(self):
        """Agent() and OmicVerseAgent.__init__ must accept identical parameters."""
        factory_sig = inspect.signature(Agent)
        init_sig = inspect.signature(OmicVerseAgent.__init__)
        # __init__ has 'self' as the first param; Agent does not
        init_params = {
            k: v for k, v in init_sig.parameters.items() if k != "self"
        }
        factory_params = dict(factory_sig.parameters)
        assert set(factory_params.keys()) == set(init_params.keys()), (
            f"Mismatch: factory has {set(factory_params) - set(init_params)}, "
            f"init has {set(init_params) - set(factory_params)}"
        )

    def test_agent_is_in_module_all(self):
        """Agent, OmicVerseAgent, list_supported_models are in __all__."""
        all_exports = smart_agent_module.__all__
        assert "Agent" in all_exports
        assert "OmicVerseAgent" in all_exports
        assert "list_supported_models" in all_exports


# ===================================================================
# 2. run_async
# ===================================================================

class TestRunAsync:

    def test_run_async_delegates_to_agentic_loop(self):
        """run_async enters _run_agentic_mode for non-Python requests."""
        agent = _bare_agent()

        loop_called = {}

        async def _fake_agentic_mode(self, request, adata):
            loop_called["request"] = request
            loop_called["adata"] = adata
            return adata

        agent._run_agentic_mode = MethodType(_fake_agentic_mode, agent)
        agent._detect_direct_python_request = MethodType(
            lambda self, req: None, agent
        )

        sentinel = object()
        result = asyncio.run(agent.run_async("cluster analysis", sentinel))

        assert result is sentinel
        assert loop_called["request"] == "cluster analysis"
        assert loop_called["adata"] is sentinel

    def test_run_async_direct_python_bypasses_llm(self):
        """run_async executes direct Python without entering the agentic loop."""
        agent = _bare_agent()
        agent.last_usage_breakdown = {
            "generation": None,
            "reflection": [],
            "review": [],
            "total": None,
        }

        exec_called = {}

        def _fake_detect(self, request):
            return "x = 42"

        def _fake_execute(self, code, adata):
            exec_called["code"] = code
            return "executed"

        agent._detect_direct_python_request = MethodType(_fake_detect, agent)
        agent._execute_generated_code = MethodType(_fake_execute, agent)

        result = asyncio.run(agent.run_async("import numpy", None))

        assert result == "executed"
        assert exec_called["code"] == "x = 42"

    def test_run_async_rejects_python_provider_without_code(self):
        """Python provider raises ValueError for non-executable requests."""
        agent = _bare_agent(provider="python")
        agent._detect_direct_python_request = MethodType(
            lambda self, req: None, agent
        )

        with pytest.raises(ValueError, match="Python provider"):
            asyncio.run(agent.run_async("do analysis", None))


# ===================================================================
# 3. stream_async
# ===================================================================

class TestStreamAsync:

    def test_stream_async_yields_typed_events(self):
        """stream_async yields dicts with a 'type' key."""
        agent = _bare_agent()
        agent._get_harness_session_id = MethodType(
            lambda self: "sess-test", agent
        )

        async def _fake_loop(self, request, adata, event_callback=None,
                             cancel_event=None, history=None,
                             approval_handler=None):
            await event_callback(build_stream_event(
                "tool_call",
                {"name": "inspect_data", "arguments": {}},
                turn_id="t1", trace_id="tr1", session_id="sess-test",
                category="tool",
            ))
            await event_callback(build_stream_event(
                "done", "completed",
                turn_id="t1", trace_id="tr1", session_id="sess-test",
                category="lifecycle",
            ))

        agent._run_agentic_loop = MethodType(_fake_loop, agent)

        async def _collect():
            events = []
            async for event in agent.stream_async("inspect", object()):
                events.append(event)
            return events

        events = asyncio.run(_collect())

        assert len(events) == 2
        assert all("type" in e for e in events)
        assert events[0]["type"] == "tool_call"
        assert events[1]["type"] == "done"

    def test_stream_async_error_emits_error_then_done(self):
        """On exception, stream_async emits error then done (defense-in-depth)."""
        agent = _bare_agent()
        agent._get_harness_session_id = MethodType(
            lambda self: "sess-err", agent
        )

        async def _exploding_loop(self, request, adata, event_callback=None,
                                  cancel_event=None, history=None,
                                  approval_handler=None):
            raise RuntimeError("boom")

        agent._run_agentic_loop = MethodType(_exploding_loop, agent)

        async def _collect():
            events = []
            async for event in agent.stream_async("fail", object()):
                events.append(event)
            return events

        events = asyncio.run(_collect())

        types_seen = [e["type"] for e in events]
        assert "error" in types_seen
        assert "done" in types_seen
        # done must come after error
        assert types_seen.index("error") < types_seen.index("done")


# ===================================================================
# 4. restart_session / get_session_history
# ===================================================================

class TestSessionManagement:

    def test_restart_session_clears_executor_state(self):
        """restart_session resets the notebook executor's session."""
        agent = _bare_agent(use_notebook_execution=True)

        archived = []

        executor = SimpleNamespace(
            current_session={"session_id": "s1"},
            session_prompt_count=3,
            _archive_current_session=lambda: archived.append(True),
        )
        agent._notebook_executor = executor

        agent.restart_session()

        assert executor.current_session is None
        assert executor.session_prompt_count == 0
        assert len(archived) == 1

    def test_restart_session_noop_without_notebook(self):
        """restart_session prints warning and does not raise when notebooks disabled."""
        agent = _bare_agent(use_notebook_execution=False)
        # Should not raise
        agent.restart_session()

    def test_restart_session_noop_no_active_session(self):
        """restart_session is safe to call when no session is active."""
        agent = _bare_agent(use_notebook_execution=True)
        agent._notebook_executor = SimpleNamespace(current_session=None)
        # Should not raise
        agent.restart_session()

    def test_get_session_history_returns_executor_history(self):
        """get_session_history returns the executor's session_history list."""
        agent = _bare_agent(use_notebook_execution=True)

        history_data = [
            {"session_id": "s1", "prompt_count": 3},
            {"session_id": "s2", "prompt_count": 5},
        ]
        agent._notebook_executor = SimpleNamespace(session_history=history_data)

        result = agent.get_session_history()
        assert result is history_data
        assert len(result) == 2

    def test_get_session_history_empty_without_notebook(self):
        """get_session_history returns [] when notebook execution is disabled."""
        agent = _bare_agent(use_notebook_execution=False)

        result = agent.get_session_history()
        assert result == []


# ===================================================================
# 5. Workflow Injection
# ===================================================================

class TestWorkflowInjection:

    def test_compose_system_prompt_injects_workflow(self, tmp_path):
        """compose_system_prompt appends workflow block to base prompt."""
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text(
            "---\n"
            "domain: bioinformatics\n"
            "default_tools:\n"
            "  - inspect_data\n"
            "  - execute_code\n"
            "---\n"
            "\n"
            "Always validate outputs.\n",
            encoding="utf-8",
        )

        rt = OmicVerseRuntime(
            repo_root=tmp_path,
            workflow_path=wf_path,
        )
        base = "You are a helpful agent."
        result = rt.compose_system_prompt(base)

        assert result.startswith(base.rstrip())
        assert "REPOSITORY WORKFLOW POLICY" in result
        assert "bioinformatics" in result
        assert "Always validate outputs." in result

    def test_compose_system_prompt_noop_without_body(self, tmp_path):
        """compose_system_prompt returns base unchanged for empty workflow."""
        # No WORKFLOW.md exists → load_workflow_document returns a bare doc
        rt = OmicVerseRuntime(repo_root=tmp_path)
        base = "You are a helpful agent."
        result = rt.compose_system_prompt(base)

        assert result == base


# ===================================================================
# 6. Downstream Integration: Claw
# ===================================================================

class TestClawIntegration:

    def test_claw_load_agent_factory_returns_callable(self):
        """claw._load_agent_factory() returns the Agent callable."""
        claw_path = PACKAGE_ROOT / "claw.py"
        claw_spec = importlib.util.spec_from_file_location(
            "omicverse.claw", claw_path,
            submodule_search_locations=[],
        )
        claw_module = importlib.util.module_from_spec(claw_spec)

        # Pre-seed omicverse.utils.smart_agent in sys.modules so the
        # relative import inside _load_agent_factory succeeds.
        saved = sys.modules.get("omicverse.utils.smart_agent")
        sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
        try:
            assert claw_spec.loader is not None
            claw_spec.loader.exec_module(claw_module)
            factory = claw_module._load_agent_factory()
            assert factory is Agent
        finally:
            if saved is None:
                sys.modules.pop("omicverse.utils.smart_agent", None)
            else:
                sys.modules["omicverse.utils.smart_agent"] = saved

    def test_claw_build_agent_passes_expected_kwargs(self):
        """claw._build_agent passes the documented parameter subset to Agent."""
        claw_path = PACKAGE_ROOT / "claw.py"
        claw_spec = importlib.util.spec_from_file_location(
            "omicverse.claw", claw_path,
            submodule_search_locations=[],
        )
        claw_module = importlib.util.module_from_spec(claw_spec)

        saved = sys.modules.get("omicverse.utils.smart_agent")
        sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
        try:
            assert claw_spec.loader is not None
            claw_spec.loader.exec_module(claw_module)

            # Capture kwargs passed to Agent
            captured = {}

            def _capturing_agent(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace()

            claw_module._load_agent_factory = lambda: _capturing_agent

            args = SimpleNamespace(
                model="gpt-5.2",
                api_key="test-key",
                endpoint=None,
                no_reflection=False,
            )
            claw_module._build_agent(args)

            # Contract: Claw must pass these specific overrides
            assert captured["model"] == "gpt-5.2"
            assert captured["api_key"] == "test-key"
            assert captured["enable_reflection"] is True  # not args.no_reflection
            assert captured["enable_result_review"] is False
            assert captured["use_notebook_execution"] is False
            assert captured["enable_filesystem_context"] is False
            assert captured["verbose"] is False
        finally:
            if saved is None:
                sys.modules.pop("omicverse.utils.smart_agent", None)
            else:
                sys.modules["omicverse.utils.smart_agent"] = saved


# ===================================================================
# 7. list_supported_models
# ===================================================================

class TestListSupportedModels:

    def test_list_supported_models_returns_string(self):
        """list_supported_models() returns a non-empty string."""
        result = list_supported_models()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_supported_models_show_all_flag(self):
        """list_supported_models(show_all=True) returns at least as much as default."""
        default = list_supported_models(show_all=False)
        full = list_supported_models(show_all=True)
        assert len(full) >= len(default)


# ===================================================================
# 8. Prompt Template Engine
# ===================================================================

class TestPromptTemplateEngine:
    """Verify that PromptTemplateEngine satisfies the PromptComposer protocol
    and composes layers correctly."""

    def test_satisfies_prompt_composer_protocol(self):
        """PromptTemplateEngine is a valid PromptComposer."""
        engine = PromptTemplateEngine()
        assert isinstance(engine, PromptComposer)

    def test_compose_joins_layers_by_priority(self):
        """Layers are composed in ascending priority order with \\n\\n separators."""
        engine = PromptTemplateEngine()
        engine.add(PromptLayerKind.BASE_SYSTEM, "First", priority=0)
        engine.add(PromptLayerKind.WORKFLOW, "Third", priority=20)
        engine.add(PromptLayerKind.SKILL, "Second", priority=10)
        result = engine.compose()
        assert result == "First\n\nSecond\n\nThird"

    def test_compose_skips_empty_layers(self):
        """Empty or whitespace-only layers are not added."""
        engine = PromptTemplateEngine()
        engine.add(PromptLayerKind.BASE_SYSTEM, "Content", priority=0)
        engine.add(PromptLayerKind.SKILL, "", priority=10)
        engine.add(PromptLayerKind.SKILL, "   ", priority=20)
        assert engine.layer_count() == 1
        assert engine.compose() == "Content"

    def test_total_tokens_estimates(self):
        """total_tokens returns a positive estimate."""
        engine = PromptTemplateEngine()
        engine.add(PromptLayerKind.BASE_SYSTEM, "a" * 400, priority=0)
        assert engine.total_tokens() >= 100  # ~400/4

    def test_remove_kind(self):
        """remove_kind removes all layers of a given kind."""
        engine = PromptTemplateEngine()
        engine.add(PromptLayerKind.BASE_SYSTEM, "Base", priority=0)
        engine.add(PromptLayerKind.WORKFLOW, "WF1", priority=10)
        engine.add(PromptLayerKind.WORKFLOW, "WF2", priority=20)
        removed = engine.remove_kind(PromptLayerKind.WORKFLOW)
        assert removed == 2
        assert not engine.has_kind(PromptLayerKind.WORKFLOW)
        assert engine.compose() == "Base"

    def test_has_kind(self):
        """has_kind correctly detects layer presence."""
        engine = PromptTemplateEngine()
        assert not engine.has_kind(PromptLayerKind.SKILL)
        engine.add(PromptLayerKind.SKILL, "Skills", priority=0)
        assert engine.has_kind(PromptLayerKind.SKILL)

    def test_clear(self):
        """clear removes all layers."""
        engine = PromptTemplateEngine()
        engine.add(PromptLayerKind.BASE_SYSTEM, "A", priority=0)
        engine.add(PromptLayerKind.WORKFLOW, "B", priority=10)
        engine.clear()
        assert engine.layer_count() == 0
        assert engine.compose() == ""

    def test_layers_returns_sorted_sequence(self):
        """layers() returns layers sorted by priority."""
        engine = PromptTemplateEngine()
        engine.add(PromptLayerKind.SKILL, "C", priority=30)
        engine.add(PromptLayerKind.BASE_SYSTEM, "A", priority=10)
        engine.add(PromptLayerKind.WORKFLOW, "B", priority=20)
        result = engine.layers()
        assert [l.content for l in result] == ["A", "B", "C"]


# ===================================================================
# 9. Block-based Prompt Assembly
# ===================================================================

class TestBlockBasedPromptAssembly:
    """Verify that PromptBuilder uses the template engine and that
    overlays (provider, workflow, skill) are explicit layers."""

    def _mock_ctx(self, **overrides):
        """Create a minimal mock AgentContext for PromptBuilder."""
        defaults = {
            "skill_registry": None,
            "_ov_runtime": None,
            "_code_only_mode": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_agentic_engine_has_base_system_layers(self):
        """build_agentic_engine returns an engine with BASE_SYSTEM layers."""
        ctx = self._mock_ctx()
        builder = PromptBuilder(ctx)
        engine = builder.build_agentic_engine()
        base_layers = [l for l in engine.layers()
                       if l.kind == PromptLayerKind.BASE_SYSTEM]
        assert len(base_layers) >= 8  # identity through delegation
        # Content check: key blocks are present
        composed = engine.compose()
        assert "OmicVerse Agent" in composed
        assert "MANDATORY CODE QUALITY RULES" in composed
        assert "DELEGATION STRATEGY" in composed
        assert "CODING / FILESYSTEM" in composed
        assert "WEB ACCESS" in composed

    def test_agentic_prompt_materially_smaller_monolith(self):
        """The monolithic string constant is gone; content comes from blocks."""
        # Each block constant is at most a few hundred chars, not thousands
        assert len(IDENTITY_BLOCK) < 200
        assert len(DELEGATION_BLOCK) < 600
        # But composed together they form a substantial prompt
        ctx = self._mock_ctx()
        builder = PromptBuilder(ctx)
        prompt = builder.build_agentic_system_prompt()
        assert len(prompt) > 2000

    def test_claw_overlay_is_provider_layer(self):
        """Code-only mode adds a PROVIDER layer."""
        ctx = self._mock_ctx(_code_only_mode=True)
        builder = PromptBuilder(ctx)
        engine = builder.build_agentic_engine()
        assert engine.has_kind(PromptLayerKind.PROVIDER)
        provider_layers = [l for l in engine.layers()
                           if l.kind == PromptLayerKind.PROVIDER]
        assert len(provider_layers) == 1
        assert "CLAW CODE-ONLY MODE" in provider_layers[0].content

    def test_no_claw_overlay_by_default(self):
        """Without code-only mode, no PROVIDER layer exists."""
        ctx = self._mock_ctx()
        builder = PromptBuilder(ctx)
        engine = builder.build_agentic_engine()
        assert not engine.has_kind(PromptLayerKind.PROVIDER)

    def test_skill_overlay_is_explicit_layer(self):
        """Skills are added as a SKILL layer when registry has metadata."""
        mock_meta = SimpleNamespace(slug="qc", description="Quality control workflow")
        mock_registry = SimpleNamespace(skill_metadata={"qc": mock_meta})
        ctx = self._mock_ctx(skill_registry=mock_registry)
        builder = PromptBuilder(ctx)
        engine = builder.build_agentic_engine()
        assert engine.has_kind(PromptLayerKind.SKILL)
        skill_layers = [l for l in engine.layers()
                        if l.kind == PromptLayerKind.SKILL]
        assert len(skill_layers) == 1
        assert "qc" in skill_layers[0].content

    def test_workflow_overlay_is_explicit_layer(self, tmp_path):
        """Workflow config is added as a WORKFLOW layer."""
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text(
            "---\n"
            "domain: bioinformatics\n"
            "---\n"
            "\nAlways check outputs.\n",
            encoding="utf-8",
        )
        rt = OmicVerseRuntime(repo_root=tmp_path, workflow_path=wf_path)
        ctx = self._mock_ctx(_ov_runtime=rt)
        builder = PromptBuilder(ctx)
        engine = builder.build_agentic_engine()
        assert engine.has_kind(PromptLayerKind.WORKFLOW)
        wf_layers = [l for l in engine.layers()
                     if l.kind == PromptLayerKind.WORKFLOW]
        assert len(wf_layers) == 1
        assert "Always check outputs." in wf_layers[0].content

    def test_no_workflow_layer_without_document(self, tmp_path):
        """Without a WORKFLOW.md, no WORKFLOW layer is present."""
        rt = OmicVerseRuntime(repo_root=tmp_path)
        ctx = self._mock_ctx(_ov_runtime=rt)
        builder = PromptBuilder(ctx)
        engine = builder.build_agentic_engine()
        assert not engine.has_kind(PromptLayerKind.WORKFLOW)

    def test_subagent_explore_prompt_contains_expected_content(self):
        """Explore subagent prompt includes identity and report blocks."""
        ctx = self._mock_ctx()
        builder = PromptBuilder(ctx)
        prompt = builder.build_explore_prompt("")
        assert "bioinformatics data inspector" in prompt
        assert "Dataset dimensions" in prompt
        assert "finish()" in prompt

    def test_subagent_plan_prompt_includes_code_quality(self):
        """Plan subagent prompt includes code quality rules."""
        ctx = self._mock_ctx()
        builder = PromptBuilder(ctx)
        prompt = builder.build_plan_prompt("")
        assert "MANDATORY CODE QUALITY RULES" in prompt
        assert "finish()" in prompt

    def test_subagent_context_appears_in_prompt(self):
        """Parent context is injected into subagent prompts."""
        ctx = self._mock_ctx()
        builder = PromptBuilder(ctx)
        prompt = builder.build_explore_prompt("look at mito genes")
        assert "look at mito genes" in prompt

    def test_code_quality_rules_backward_compat(self):
        """CODE_QUALITY_RULES constant matches CODE_QUALITY_BLOCK content."""
        assert "MANDATORY CODE QUALITY RULES" in CODE_QUALITY_RULES
        assert CODE_QUALITY_RULES.startswith(CODE_QUALITY_BLOCK)

    def test_build_skill_layer_returns_none_for_empty(self):
        """build_skill_layer returns None when no metadata."""
        assert build_skill_layer({}) is None

    def test_build_skill_layer_creates_valid_layer(self):
        """build_skill_layer produces a SKILL PromptLayer."""
        meta = SimpleNamespace(slug="clustering", description="Cell clustering")
        layer = build_skill_layer({"clustering": meta})
        assert layer is not None
        assert layer.kind == PromptLayerKind.SKILL
        assert "clustering" in layer.content

    def test_runtime_build_workflow_layer(self, tmp_path):
        """OmicVerseRuntime.build_workflow_layer returns a PromptLayer."""
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text(
            "---\ndomain: bioinformatics\n---\nValidate always.\n",
            encoding="utf-8",
        )
        rt = OmicVerseRuntime(repo_root=tmp_path, workflow_path=wf_path)
        layer = rt.build_workflow_layer()
        assert layer is not None
        assert layer.kind == PromptLayerKind.WORKFLOW
        assert "Validate always." in layer.content

    def test_runtime_build_workflow_layer_none_without_doc(self, tmp_path):
        """build_workflow_layer returns None when no WORKFLOW.md exists."""
        rt = OmicVerseRuntime(repo_root=tmp_path)
        assert rt.build_workflow_layer() is None
