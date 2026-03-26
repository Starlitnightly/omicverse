"""Runtime contract tests for the OVAgent subsystem surface.

Verifies that the live checked-out OVAgent modules satisfy the
runtime contracts that the review baseline depends on:

  - ``ovagent/__init__.py`` exports resolve to real objects
  - ``AgentContext`` protocol is runtime-checkable and lists expected members
  - Module composition seams (ToolRuntime, TurnController, PromptBuilder,
    SessionService, ContextService, SubagentController, RegistryScanner)
    can be instantiated against a duck-typed context double
  - ``OmicVerseRuntime`` workflow composition contract
  - ``FollowUpGate`` classification heuristics
  - ``WorkflowDocument`` / ``WorkflowConfig`` round-trip
  - ``AnalysisRun`` / ``RunStore`` lifecycle
  - ``ProactiveCodeTransformer`` is importable and callable

These tests are intentionally lightweight: no real LLM calls, no heavy
optional dependencies.  They run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import importlib
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: these tests require the harness env-var
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Runtime contract tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs to avoid heavy omicverse.__init__
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

# Now import the ovagent package
import omicverse.utils.ovagent as ovagent_pkg
from omicverse.utils.ovagent import (
    AgentContext,
    AnalysisExecutor,
    AnalysisRun,
    CODE_QUALITY_RULES,
    ContextService,
    FollowUpGate,
    OmicVerseRuntime,
    ProactiveCodeTransformer,
    PromptBuilder,
    RegistryScanner,
    ResolvedBackend,
    RunStore,
    SessionService,
    SubagentController,
    ToolRuntime,
    TurnController,
    WorkflowConfig,
    WorkflowDocument,
    build_filesystem_context_instructions,
    collect_api_key_env,
    create_llm_backend,
    display_backend_info,
    display_reflection_config,
    format_skill_overview,
    initialize_filesystem_context,
    initialize_notebook_executor,
    initialize_ov_runtime,
    initialize_security,
    initialize_session_history,
    initialize_skill_registry,
    initialize_tracing,
    load_workflow_document,
    resolve_model_and_provider,
    temporary_api_keys,
)
from omicverse.utils.ovagent.protocol import AgentContext as AgentContextProtocol
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS
from omicverse.utils.harness.runtime_state import runtime_state

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Minimal duck-typed context double (matches AgentContext protocol)
# ---------------------------------------------------------------------------

class _MinimalCtx:
    """Lightweight AgentContext double for composition tests."""

    LEGACY_AGENT_TOOLS = LEGACY_AGENT_TOOLS

    def __init__(self, session_id: str = "test-contract-session"):
        self.model = "test-model"
        self.provider = "openai"
        self.endpoint = "https://api.example.com"
        self.api_key = None
        self._llm = None
        self._config = SimpleNamespace()
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
        self._web_session_id = session_id
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False

    def _emit(self, level, message, category=""):
        pass

    def _get_harness_session_id(self):
        return self._web_session_id

    def _get_runtime_session_id(self):
        return self._web_session_id or "default"

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

    def _extract_python_code_strict(self, text):
        return text

    def _gather_code_candidates(self, text):
        return [text]

    def _normalize_code_candidate(self, code):
        return code

    def _collect_static_registry_entries(self, query, max_entries=20):
        return []

    def _collect_runtime_registry_entries(self, query, max_entries=20):
        return []

    def _review_generated_code_lightweight(self, request, code, entries):
        return code

    def _contains_forbidden_scanpy_usage(self, code):
        return False

    def _rewrite_scanpy_calls_with_registry(self, code, entries):
        return code

    def _normalize_registry_entry_for_codegen(self, entry):
        return entry

    def _build_agentic_system_prompt(self):
        return "You are a test agent."

    async def _run_agentic_loop(self, request, adata, event_callback=None,
                                cancel_event=None, history=None,
                                approval_handler=None, request_content=None):
        return adata

    @contextmanager
    def _temporary_api_keys(self):
        yield


class _DummyExecutor:
    pass


# ===================================================================
# 1. __init__.py export contract
# ===================================================================

class TestOvagentExports:
    """Every name in ovagent.__all__ resolves to a real object."""

    def test_all_exports_are_importable(self):
        all_names = ovagent_pkg.__all__
        assert len(all_names) > 0
        for name in all_names:
            obj = getattr(ovagent_pkg, name, None)
            assert obj is not None, f"ovagent.__all__ lists '{name}' but it is None"

    def test_key_classes_in_all(self):
        expected = {
            "AgentContext", "ToolRuntime", "TurnController", "FollowUpGate",
            "PromptBuilder", "SessionService", "ContextService",
            "SubagentController", "RegistryScanner", "OmicVerseRuntime",
            "AnalysisExecutor", "ProactiveCodeTransformer",
            "WorkflowConfig", "WorkflowDocument", "AnalysisRun", "RunStore",
        }
        actual = set(ovagent_pkg.__all__)
        missing = expected - actual
        assert not missing, f"Expected exports missing from __all__: {missing}"

    def test_key_functions_in_all(self):
        expected = {
            "load_workflow_document", "build_filesystem_context_instructions",
            "resolve_model_and_provider", "collect_api_key_env",
            "temporary_api_keys", "create_llm_backend",
        }
        actual = set(ovagent_pkg.__all__)
        missing = expected - actual
        assert not missing, f"Expected function exports missing: {missing}"


# ===================================================================
# 2. AgentContext protocol contract
# ===================================================================

class TestAgentContextProtocol:
    """AgentContext is runtime-checkable and defines the expected surface."""

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(AgentContextProtocol, "__protocol_attrs__") or hasattr(
            AgentContextProtocol, "__abstractmethods__"
        ) or isinstance(AgentContextProtocol, type)

    def test_minimal_ctx_satisfies_protocol(self):
        ctx = _MinimalCtx()
        assert isinstance(ctx, AgentContextProtocol)

    def test_protocol_requires_core_attributes(self):
        """Protocol must declare model, provider, endpoint at minimum."""
        hints = {}
        for cls in AgentContextProtocol.__mro__:
            hints.update(getattr(cls, "__annotations__", {}))
        assert "model" in hints
        assert "provider" in hints
        assert "endpoint" in hints

    def test_protocol_requires_subsystem_refs(self):
        hints = {}
        for cls in AgentContextProtocol.__mro__:
            hints.update(getattr(cls, "__annotations__", {}))
        for attr in ["_llm", "_config", "_reporter", "_notebook_executor",
                      "_trace_store", "_session_history"]:
            assert attr in hints, f"Protocol missing subsystem ref '{attr}'"


# ===================================================================
# 3. Module composition seams
# ===================================================================

class TestModuleComposition:
    """Key OVAgent modules can be instantiated with a duck-typed context."""

    def test_tool_runtime_construction(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        assert rt is not None

    def test_tool_runtime_has_dispatch_tool(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        assert hasattr(rt, "dispatch_tool")
        assert callable(rt.dispatch_tool)

    def test_tool_runtime_get_visible_agent_tools(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        tools = rt.get_visible_agent_tools()
        assert isinstance(tools, list)
        names = {t["name"] for t in tools}
        # Must include both catalog and legacy tools
        assert "inspect_data" in names
        assert "execute_code" in names

    def test_prompt_builder_construction(self):
        ctx = _MinimalCtx()
        pb = PromptBuilder(ctx)
        assert pb is not None

    def test_prompt_builder_has_explore_prompt(self):
        ctx = _MinimalCtx()
        pb = PromptBuilder(ctx)
        result = pb.build_explore_prompt("test context")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_session_service_construction(self):
        ctx = _MinimalCtx()
        svc = SessionService(ctx)
        assert svc is not None

    def test_session_service_session_id(self):
        ctx = _MinimalCtx(session_id="contract-test-id")
        svc = SessionService(ctx)
        sid = svc.get_harness_session_id()
        assert sid == "contract-test-id"

    def test_context_service_construction(self):
        ctx = _MinimalCtx()
        svc = ContextService(ctx)
        assert svc is not None

    def test_registry_scanner_construction(self):
        scanner = RegistryScanner()
        assert scanner is not None
        assert scanner._static_registry_entries_cache is None

    def test_subagent_controller_construction(self):
        ctx = _MinimalCtx()
        pb = PromptBuilder(ctx)
        rt = ToolRuntime(ctx, _DummyExecutor())
        sc = SubagentController(ctx, pb, rt)
        assert sc is not None


# ===================================================================
# 4. FollowUpGate heuristics
# ===================================================================

class TestFollowUpGateContract:
    """FollowUpGate classification patterns are stable."""

    def test_url_request_requires_tool_action(self):
        assert FollowUpGate.request_requires_tool_action(
            "fetch https://example.com/data.csv", None
        ) is True

    def test_adata_present_requires_tool_action(self):
        assert FollowUpGate.request_requires_tool_action(
            "analyze this", object()
        ) is True

    def test_empty_request_no_tool_action(self):
        assert FollowUpGate.request_requires_tool_action("", None) is False

    def test_action_verb_requires_tool_action(self):
        for verb in ["analyze", "download", "inspect", "load", "execute"]:
            result = FollowUpGate.request_requires_tool_action(
                f"{verb} the dataset", None
            )
            assert result is True, f"Expected True for action verb '{verb}'"

    def test_gate_has_expected_patterns(self):
        """FollowUpGate exposes the documented regex patterns."""
        assert hasattr(FollowUpGate, "URL_PATTERN")
        assert hasattr(FollowUpGate, "ACTION_REQUEST_PATTERN")
        assert hasattr(FollowUpGate, "PROMISSORY_PATTERN")
        assert hasattr(FollowUpGate, "BLOCKER_PATTERN")
        assert hasattr(FollowUpGate, "RESULT_PATTERN")


# ===================================================================
# 5. OmicVerseRuntime workflow composition
# ===================================================================

class TestOmicVerseRuntimeContract:
    """OmicVerseRuntime composes workflow documents into prompts."""

    def test_runtime_construction_without_workflow(self, tmp_path):
        rt = OmicVerseRuntime(repo_root=tmp_path)
        assert rt.repo_root == tmp_path
        assert rt.workflow is not None

    def test_compose_system_prompt_noop_without_workflow(self, tmp_path):
        rt = OmicVerseRuntime(repo_root=tmp_path)
        base = "Base system prompt."
        result = rt.compose_system_prompt(base)
        assert result == base

    def test_compose_system_prompt_injects_workflow(self, tmp_path):
        wf = tmp_path / "WORKFLOW.md"
        wf.write_text(
            "---\n"
            "domain: bioinformatics\n"
            "default_tools:\n"
            "  - inspect_data\n"
            "---\n"
            "\n"
            "Run QC first.\n",
            encoding="utf-8",
        )
        rt = OmicVerseRuntime(repo_root=tmp_path, workflow_path=wf)
        result = rt.compose_system_prompt("Base prompt.")
        assert "REPOSITORY WORKFLOW POLICY" in result
        assert "bioinformatics" in result
        assert "Run QC first." in result

    def test_reload_workflow_returns_document(self, tmp_path):
        rt = OmicVerseRuntime(repo_root=tmp_path)
        doc = rt.reload_workflow()
        assert isinstance(doc, WorkflowDocument)


# ===================================================================
# 6. WorkflowDocument / WorkflowConfig
# ===================================================================

class TestWorkflowContractTypes:

    def test_workflow_config_has_expected_fields(self):
        cfg = WorkflowConfig()
        assert hasattr(cfg, "domain")
        assert hasattr(cfg, "default_tools")

    def test_load_workflow_document_returns_document(self, tmp_path):
        doc = load_workflow_document(tmp_path)
        assert isinstance(doc, WorkflowDocument)

    def test_workflow_document_has_config_and_body(self, tmp_path):
        doc = load_workflow_document(tmp_path)
        assert hasattr(doc, "config")
        assert hasattr(doc, "body")
        assert isinstance(doc.config, WorkflowConfig)


# ===================================================================
# 7. RunStore / AnalysisRun lifecycle
# ===================================================================

class TestRunStoreContract:

    def test_run_store_construction(self, tmp_path):
        store = RunStore(root=tmp_path / "runs")
        assert store is not None

    def test_analysis_run_has_expected_fields(self):
        run = AnalysisRun(
            run_id="test-run-001",
            request="run qc",
            model="test-model",
            provider="openai",
        )
        assert run.run_id == "test-run-001"
        assert run.status == "started"
        assert isinstance(run.trace_ids, list)
        assert isinstance(run.artifacts, list)

    def test_start_and_finish_run(self, tmp_path):
        store = RunStore(root=tmp_path / "runs")
        doc = load_workflow_document(tmp_path)
        run = store.start_run(
            request="test analysis",
            model="test-model",
            provider="openai",
            session_id="sess-001",
            workflow=doc,
        )
        assert run.status == "started"
        finished = store.finish_run(run.run_id, status="success", summary="done")
        assert finished.status == "success"


# ===================================================================
# 8. ProactiveCodeTransformer
# ===================================================================

class TestProactiveCodeTransformerContract:

    def test_transformer_is_importable(self):
        assert ProactiveCodeTransformer is not None

    def test_transformer_has_inplace_functions(self):
        assert hasattr(ProactiveCodeTransformer, "INPLACE_FUNCTIONS")
        funcs = ProactiveCodeTransformer.INPLACE_FUNCTIONS
        assert isinstance(funcs, set)
        assert "pca" in funcs
        assert "scale" in funcs

    def test_transformer_instantiation(self):
        t = ProactiveCodeTransformer()
        assert t is not None


# ===================================================================
# 9. LEGACY_AGENT_TOOLS contract
# ===================================================================

class TestLegacyToolsContract:

    def test_legacy_tools_is_list(self):
        assert isinstance(LEGACY_AGENT_TOOLS, list)
        assert len(LEGACY_AGENT_TOOLS) > 0

    def test_legacy_tools_have_required_fields(self):
        for tool in LEGACY_AGENT_TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "parameters" in tool, f"Tool missing 'parameters': {tool}"
            assert tool["parameters"]["type"] == "object"

    def test_expected_legacy_tool_names(self):
        names = {t["name"] for t in LEGACY_AGENT_TOOLS}
        expected = {
            "inspect_data", "execute_code", "run_snippet",
            "search_functions", "search_skills", "delegate",
            "web_fetch", "web_search", "web_download", "finish",
        }
        assert expected == names


# ===================================================================
# 10. CODE_QUALITY_RULES constant
# ===================================================================

class TestCodeQualityRulesContract:

    def test_code_quality_rules_is_string(self):
        assert isinstance(CODE_QUALITY_RULES, str)
        assert len(CODE_QUALITY_RULES) > 0

    def test_code_quality_rules_mentions_mandatory(self):
        assert "MANDATORY" in CODE_QUALITY_RULES


# ===================================================================
# 11. Auth module surface
# ===================================================================

class TestAuthSurfaceContract:

    def test_resolve_model_and_provider_callable(self):
        assert callable(resolve_model_and_provider)

    def test_collect_api_key_env_callable(self):
        assert callable(collect_api_key_env)

    def test_temporary_api_keys_callable(self):
        assert callable(temporary_api_keys)

    def test_resolved_backend_is_type(self):
        assert isinstance(ResolvedBackend, type)


# ===================================================================
# 12. Bootstrap function surface
# ===================================================================

class TestBootstrapSurfaceContract:

    def test_bootstrap_functions_are_callable(self):
        funcs = [
            format_skill_overview,
            initialize_skill_registry,
            initialize_notebook_executor,
            initialize_filesystem_context,
            initialize_session_history,
            initialize_tracing,
            initialize_security,
            initialize_ov_runtime,
            create_llm_backend,
            display_reflection_config,
        ]
        for fn in funcs:
            assert callable(fn), f"{fn} is not callable"
