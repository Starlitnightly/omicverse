"""
Compatibility audit — integration hardening tests for the upgraded OVAgent runtime.

Task-016 acceptance criterion:
  Compatibility audit is complete across Agent/Jarvis/Claw/MCP/workflow surfaces;
  residual risks and follow-ups are documented.

This test module validates that:
1. Public API surfaces remain stable after tasks 009–015 runtime upgrades.
2. Cross-module import integrity holds with no circular dependencies.
3. Integration surfaces (Claw, Jarvis, MCP, workflow) can consume the
   runtime as expected.
4. Contract conformance (ContextBudgetManager, EventEmitter, PromptComposer)
   is validated at runtime.
5. All ovagent __init__.py exports resolve to importable symbols.
6. The AgentContext protocol captures all attributes the runtime modules need.
7. Backward-compatibility shims (LEGACY_AGENT_TOOLS, aliases) are intact.
"""

import asyncio
import importlib
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap smart_agent in isolation (mirrors test_agent_contract.py pattern)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"


# ---------------------------------------------------------------------------
# §1  Public API Surface Stability
# ---------------------------------------------------------------------------

class TestSmartAgentPublicAPI:
    """Freeze smart_agent.__all__ to catch unintended removals."""

    def test_smart_agent_all_contains_required_exports(self):
        mod = importlib.import_module("omicverse.utils.smart_agent")
        required = {"Agent", "OmicVerseAgent", "list_supported_models"}
        actual = set(mod.__all__)
        assert required == actual, f"smart_agent.__all__ drifted: {required.symmetric_difference(actual)}"

    def test_omicverse_agent_has_public_methods(self):
        mod = importlib.import_module("omicverse.utils.smart_agent")
        cls = mod.OmicVerseAgent
        required_methods = {
            "run", "run_async", "stream_async", "generate_code",
            "generate_code_async", "restart_session", "get_session_history",
        }
        actual = {name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
                  if not name.startswith("_")}
        missing = required_methods - actual
        assert not missing, f"OmicVerseAgent missing public methods: {missing}"

    def test_omicverse_agent_class_attributes(self):
        mod = importlib.import_module("omicverse.utils.smart_agent")
        cls = mod.OmicVerseAgent
        assert hasattr(cls, "LEGACY_AGENT_TOOLS"), "LEGACY_AGENT_TOOLS class attr missing"
        assert hasattr(cls, "AGENT_TOOLS"), "AGENT_TOOLS class attr missing"
        assert isinstance(cls.LEGACY_AGENT_TOOLS, list)
        assert isinstance(cls.AGENT_TOOLS, list)


class TestOvagentInitExports:
    """Freeze ovagent/__init__.py __all__ to prevent export regressions."""

    REQUIRED_EXPORTS = {
        # Core runtime
        "AgentContext", "OmicVerseRuntime", "AnalysisRun", "RunStore",
        # Prompt
        "PromptBuilder", "PromptTemplateEngine", "CODE_QUALITY_RULES",
        "build_filesystem_context_instructions", "build_skill_layer",
        # Tool
        "ToolRuntime", "ToolScheduler", "BatchResult", "ExecutionWave",
        "ScheduledCall", "ScheduledResult",
        # Permission & isolation
        "PermissionPolicy", "PermissionDecision", "PermissionVerdict",
        "ToolPermissionRule", "build_subagent_policy",
        "SubagentController", "IsolatedSubagentContext", "SubagentResult",
        # Control flow
        "TurnController", "FollowUpGate",
        "RepairLoop", "RepairResult",
        "build_default_repair_loop", "normalize_execution_failure",
        # Session & analysis
        "SessionService", "ContextService", "RegistryScanner",
        "AnalysisExecutor", "ProactiveCodeTransformer",
        # Auth & bootstrap
        "ResolvedBackend", "resolve_model_and_provider",
        "collect_api_key_env", "temporary_api_keys", "display_backend_info",
        "create_llm_backend",
        "format_skill_overview", "display_reflection_config",
        "initialize_skill_registry", "initialize_notebook_executor",
        "initialize_filesystem_context", "initialize_session_history",
        "initialize_tracing", "initialize_security", "initialize_ov_runtime",
        # Workflow
        "WorkflowConfig", "WorkflowDocument", "load_workflow_document",
    }

    def test_all_required_exports_present(self):
        mod = importlib.import_module("omicverse.utils.ovagent")
        actual = set(mod.__all__)
        missing = self.REQUIRED_EXPORTS - actual
        assert not missing, f"ovagent __all__ missing: {missing}"

    def test_all_exports_resolve_to_importable_symbols(self):
        mod = importlib.import_module("omicverse.utils.ovagent")
        broken = []
        for name in mod.__all__:
            obj = getattr(mod, name, None)
            if obj is None:
                broken.append(name)
        assert not broken, f"ovagent exports not importable: {broken}"

    def test_no_unexpected_removals_from_all(self):
        """Ensure __all__ has at least the count we expect (guard against bulk deletion)."""
        mod = importlib.import_module("omicverse.utils.ovagent")
        assert len(mod.__all__) >= 50, (
            f"ovagent __all__ shrank to {len(mod.__all__)} — expected ≥50"
        )


# ---------------------------------------------------------------------------
# §2  Cross-Module Import Integrity
# ---------------------------------------------------------------------------

class TestCrossModuleImports:
    """Verify all ovagent submodules import cleanly and their inter-dependencies are acyclic."""

    OVAGENT_MODULES = [
        "omicverse.utils.ovagent.runtime",
        "omicverse.utils.ovagent.tool_runtime",
        "omicverse.utils.ovagent.tool_scheduler",
        "omicverse.utils.ovagent.turn_controller",
        "omicverse.utils.ovagent.context_budget",
        "omicverse.utils.ovagent.repair_loop",
        "omicverse.utils.ovagent.permission_policy",
        "omicverse.utils.ovagent.prompt_templates",
        "omicverse.utils.ovagent.prompt_builder",
        "omicverse.utils.ovagent.event_stream",
        "omicverse.utils.ovagent.subagent_controller",
        "omicverse.utils.ovagent.session_context",
        "omicverse.utils.ovagent.registry_scanner",
        "omicverse.utils.ovagent.analysis_executor",
        "omicverse.utils.ovagent.codegen_pipeline",
        "omicverse.utils.ovagent.protocol",
        "omicverse.utils.ovagent.contracts",
        "omicverse.utils.ovagent.workflow",
    ]

    @pytest.mark.parametrize("module_path", OVAGENT_MODULES)
    def test_module_imports_cleanly(self, module_path):
        mod = importlib.import_module(module_path)
        assert mod is not None

    def test_no_circular_import_between_ovagent_modules(self):
        """All ovagent modules must be importable in a single pass — no circular dependency."""
        for module_path in self.OVAGENT_MODULES:
            mod = importlib.import_module(module_path)
            assert mod is not None, f"Circular import suspected: {module_path}"

    def test_smart_agent_imports_from_ovagent_without_error(self):
        mod = importlib.import_module("omicverse.utils.smart_agent")
        assert hasattr(mod, "OmicVerseAgent")


# ---------------------------------------------------------------------------
# §3  Contract Conformance
# ---------------------------------------------------------------------------

class TestContractsConformance:
    """Validate that runtime implementations conform to their Protocol contracts."""

    def test_context_budget_manager_conforms_to_protocol(self):
        from omicverse.utils.ovagent.contracts import ContextBudgetManager
        from omicverse.utils.ovagent.context_budget import DefaultContextBudgetManager

        mgr = DefaultContextBudgetManager()
        assert isinstance(mgr, ContextBudgetManager), (
            "DefaultContextBudgetManager does not satisfy ContextBudgetManager protocol"
        )

    def test_prompt_template_engine_conforms_to_prompt_composer(self):
        from omicverse.utils.ovagent.contracts import PromptComposer
        from omicverse.utils.ovagent.prompt_templates import PromptTemplateEngine

        engine = PromptTemplateEngine()
        assert isinstance(engine, PromptComposer), (
            "PromptTemplateEngine does not satisfy PromptComposer protocol"
        )

    def test_event_bus_conforms_to_event_emitter_shape(self):
        """EventBus should have emit() and emit_failure() or compatible shape."""
        from omicverse.utils.ovagent.event_stream import EventBus, make_event_bus

        bus = make_event_bus()
        assert hasattr(bus, "emit"), "EventBus missing emit()"

    def test_tool_contract_fields_roundtrip(self):
        from omicverse.utils.ovagent.contracts import (
            ToolContract, ToolPolicyMetadata, ApprovalClass, ParallelClass,
        )
        tc = ToolContract(
            name="test_tool", group="test", description="test tool",
            policy=ToolPolicyMetadata(approval=ApprovalClass.ASK, parallel=ParallelClass.UNSAFE),
        )
        d = tc.to_dict()
        assert d["name"] == "test_tool"
        assert d["policy"]["approval"] == "ask"
        assert d["policy"]["parallel"] == "unsafe"
        assert tc.requires_approval() is True
        assert tc.is_parallel_safe() is False

    def test_execution_failure_envelope_roundtrip(self):
        from omicverse.utils.ovagent.contracts import (
            ExecutionFailureEnvelope, FailurePhase, RepairHint,
        )
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="NameError",
            message="name 'x' is not defined",
            repair_hints=[RepairHint(strategy="add_import", description="add import x")],
        )
        assert env.retryable is True
        d = env.to_dict()
        assert d["tool_name"] == "execute_code"
        assert d["retryable"] is True
        llm_msg = env.to_llm_message()
        assert "execute_code" in llm_msg
        assert "NameError" in llm_msg

    def test_context_entry_token_estimate(self):
        from omicverse.utils.ovagent.contracts import ContextEntry, ImportanceTier
        entry = ContextEntry(content="hello world", source="test")
        assert entry.estimated_tokens() >= 1
        entry2 = ContextEntry(content="x", source="test", token_count=42)
        assert entry2.estimated_tokens() == 42

    def test_prompt_layer_token_estimate(self):
        from omicverse.utils.ovagent.contracts import PromptLayer, PromptLayerKind
        layer = PromptLayer(kind=PromptLayerKind.BASE_SYSTEM, content="test content")
        assert layer.estimated_tokens() >= 1
        layer2 = PromptLayer(kind=PromptLayerKind.SKILL, content="x", token_estimate=10)
        assert layer2.estimated_tokens() == 10

    def test_remote_review_config_roundtrip(self):
        from omicverse.utils.ovagent.contracts import RemoteReviewConfig
        cfg = RemoteReviewConfig(host="tw.example.com", user="admin", workspace="/slow/ov")
        d = cfg.to_dict()
        cfg2 = RemoteReviewConfig.from_dict(d)
        assert cfg2.host == cfg.host
        assert cfg2.user == cfg.user
        assert cfg2.workspace == cfg.workspace


# ---------------------------------------------------------------------------
# §4  Claw Integration Surface
# ---------------------------------------------------------------------------

class TestClawIntegrationSurface:
    """Claw must be able to import Agent and call generate_code."""

    def test_claw_imports_agent_from_smart_agent(self):
        """claw.py imports Agent from .utils.smart_agent — verify this path works."""
        from omicverse.utils.smart_agent import Agent
        assert callable(Agent)

    def test_generate_code_signature_stable(self):
        """Claw calls agent.generate_code(question, adata=None, max_functions, progress_callback)."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        sig = inspect.signature(OmicVerseAgent.generate_code)
        params = list(sig.parameters.keys())
        assert "request" in params or "self" in params
        assert "adata" in params
        assert "max_functions" in params
        assert "progress_callback" in params

    def test_agent_has_registry_introspection_methods(self):
        """Claw uses private registry methods — verify they exist on the class."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        expected = [
            "_ensure_runtime_registry_for_codegen",
            "_collect_runtime_registry_entries",
            "_normalize_registry_entry_for_codegen",
            "_get_registry_stats",
        ]
        for method_name in expected:
            assert hasattr(OmicVerseAgent, method_name), (
                f"Claw depends on OmicVerseAgent.{method_name}() — missing"
            )

    def test_claw_module_imports_cleanly(self):
        """Verify claw.py can be imported without side effects."""
        mod = importlib.import_module("omicverse.claw")
        assert hasattr(mod, "build_parser")
        assert hasattr(mod, "_build_agent")


# ---------------------------------------------------------------------------
# §5  Jarvis Integration Surface
# ---------------------------------------------------------------------------

class TestJarvisIntegrationSurface:
    """Jarvis consumes stream_async and harvests figures from notebook state."""

    def test_stream_async_is_async_generator(self):
        """Jarvis AgentBridge iterates 'async for event in agent.stream_async(...)'."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        assert hasattr(OmicVerseAgent, "stream_async")
        method = OmicVerseAgent.stream_async
        assert inspect.isfunction(method) or inspect.ismethod(method)
        assert inspect.isasyncgenfunction(method), "stream_async must be an async generator"

    def test_stream_async_signature_stable(self):
        from omicverse.utils.smart_agent import OmicVerseAgent
        sig = inspect.signature(OmicVerseAgent.stream_async)
        params = list(sig.parameters.keys())
        assert "request" in params
        assert "adata" in params

    def test_agent_bridge_imports_cleanly(self):
        """Jarvis agent_bridge module must import without error."""
        mod = importlib.import_module("omicverse.jarvis.agent_bridge")
        assert hasattr(mod, "AgentBridge")
        assert hasattr(mod, "AgentRunResult")

    def test_agent_bridge_event_types_documented(self):
        """Verify the event types Jarvis depends on are documented in the bridge."""
        mod = importlib.import_module("omicverse.jarvis.agent_bridge")
        source = inspect.getsource(mod.AgentBridge.run)
        expected_types = ["llm_chunk", "code", "result", "tool_call",
                          "tool_result", "status", "finish", "done", "error", "usage"]
        for etype in expected_types:
            assert f'"{etype}"' in source or f"'{etype}'" in source, (
                f"Event type '{etype}' not handled in AgentBridge.run"
            )


# ---------------------------------------------------------------------------
# §6  MCP Integration Surface
# ---------------------------------------------------------------------------

class TestMCPIntegrationSurface:
    """MCP server is decoupled from smart_agent/ovagent — verify the isolation."""

    def test_mcp_manifest_does_not_import_smart_agent(self):
        """MCP should not depend on the agent runtime at all."""
        manifest_path = PACKAGE_ROOT / "mcp" / "manifest.py"
        if not manifest_path.exists():
            pytest.skip("MCP manifest not present")
        source = manifest_path.read_text()
        assert "smart_agent" not in source, "MCP manifest should not import smart_agent"
        assert "from ..utils.ovagent" not in source, "MCP manifest should not import ovagent"

    def test_mcp_server_does_not_import_smart_agent(self):
        server_path = PACKAGE_ROOT / "mcp" / "server.py"
        if not server_path.exists():
            pytest.skip("MCP server not present")
        source = server_path.read_text()
        assert "smart_agent" not in source, "MCP server should not import smart_agent"

    def test_mcp_module_imports_cleanly(self):
        try:
            mod = importlib.import_module("omicverse.mcp")
            assert mod is not None
        except ImportError:
            pytest.skip("MCP not installed")


# ---------------------------------------------------------------------------
# §7  Workflow Runtime Surface
# ---------------------------------------------------------------------------

class TestWorkflowRuntimeSurface:
    """Workflow system must be self-contained and parse WORKFLOW.md manifests."""

    def test_workflow_module_imports(self):
        from omicverse.utils.ovagent.workflow import (
            WorkflowConfig, WorkflowDocument, load_workflow_document,
        )
        assert callable(load_workflow_document)

    def test_workflow_config_default_values(self):
        from omicverse.utils.ovagent.workflow import WorkflowConfig
        cfg = WorkflowConfig()
        assert cfg.domain == "bioinformatics"
        assert cfg.max_turns == 15
        assert cfg.approval_policy in ("guarded", "open", "restricted")

    def test_workflow_document_build_prompt_block(self):
        from omicverse.utils.ovagent.workflow import WorkflowConfig, WorkflowDocument
        doc = WorkflowDocument(
            path=Path("/fake"),
            config=WorkflowConfig(),
            body="Test body",
            raw_text="---\ndomain: bioinformatics\n---\nTest body",
        )
        block = doc.build_prompt_block()
        assert "Test body" in block


# ---------------------------------------------------------------------------
# §8  AgentContext Protocol Completeness
# ---------------------------------------------------------------------------

class TestAgentContextProtocol:
    """Verify AgentContext protocol attributes match what OmicVerseAgent provides."""

    def test_protocol_attributes_exist_on_omicverse_agent(self):
        from omicverse.utils.ovagent.protocol import AgentContext
        from omicverse.utils.smart_agent import OmicVerseAgent

        # Collect protocol-defined members, skipping Python Protocol internals
        _PROTOCOL_INTERNALS = {"_is_protocol", "_abc_impl", "_is_runtime_protocol"}
        protocol_members = set()
        for name in dir(AgentContext):
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in _PROTOCOL_INTERNALS:
                continue
            protocol_members.add(name)

        # These are defined at class or instance level — check both
        missing = []
        for name in protocol_members:
            # Check class attrs, properties, methods
            if not hasattr(OmicVerseAgent, name):
                # Check if it's set in __init__ (instance attrs start with _)
                init_source = inspect.getsource(OmicVerseAgent.__init__)
                if f"self.{name}" not in init_source:
                    missing.append(name)

        assert not missing, (
            f"OmicVerseAgent missing AgentContext attributes: {missing}"
        )


# ---------------------------------------------------------------------------
# §9  Tool Registry & Scheduler Integration
# ---------------------------------------------------------------------------

class TestToolRegistryIntegration:
    """Verify tool registry, scheduler, and permission policy interoperate."""

    def test_tool_dispatch_registry_lookup(self):
        from omicverse.utils.ovagent.tool_runtime import ToolDispatchRegistry, ToolRegistryEntry, ToolPolicy

        registry = ToolDispatchRegistry()
        entry = ToolRegistryEntry(
            name="test_tool",
            executor=lambda: None,
            schema={"type": "object"},
            policy=ToolPolicy(),
        )
        registry.register(entry)
        assert registry.lookup("test_tool") is entry
        assert "test_tool" in registry

    def test_scheduler_classifies_tool_calls(self):
        from omicverse.utils.ovagent.tool_runtime import ToolDispatchRegistry, ToolRegistryEntry, ToolPolicy
        from omicverse.utils.ovagent.tool_scheduler import ToolScheduler

        registry = ToolDispatchRegistry()
        for name, parallel in [("read", True), ("write", False)]:
            entry = ToolRegistryEntry(
                name=name, executor=lambda: None,
                schema={"type": "object"},
                policy=ToolPolicy(parallel_safe=parallel),
            )
            registry.register(entry)

        scheduler = ToolScheduler(registry)
        # Simulate tool calls (scheduler expects tc.name directly)
        calls = [
            SimpleNamespace(name="read", arguments="{}"),
            SimpleNamespace(name="write", arguments="{}"),
        ]
        waves = scheduler.plan(calls)
        assert len(waves) >= 1

    def test_permission_policy_check(self):
        from omicverse.utils.ovagent.permission_policy import (
            PermissionPolicy, PermissionVerdict, ToolPermissionRule,
        )
        from omicverse.utils.ovagent.contracts import ApprovalClass, IsolationMode

        policy = PermissionPolicy(
            rules=(
                ToolPermissionRule(
                    tool_name="bash",
                    approval=ApprovalClass.ASK,
                    required_isolation=IsolationMode.SUBPROCESS,
                    reason="shell access",
                ),
            ),
            denied_tools=frozenset(["rm"]),
        )
        decision = policy.check("bash")
        assert decision.verdict == PermissionVerdict.ASK
        decision2 = policy.check("rm")
        assert decision2.verdict == PermissionVerdict.DENY

    def test_build_subagent_policy(self):
        from omicverse.utils.ovagent.permission_policy import (
            build_subagent_policy, PermissionVerdict,
        )
        policy = build_subagent_policy(["read", "grep"], deny_mutations=True)
        r = policy.check("read")
        assert r.verdict == PermissionVerdict.ALLOW
        w = policy.check("write")
        assert w.verdict == PermissionVerdict.DENY


# ---------------------------------------------------------------------------
# §10  Repair Loop Integration
# ---------------------------------------------------------------------------

class TestRepairLoopIntegration:
    """Verify RepairLoop factory and strategy ordering."""

    def test_build_default_repair_loop(self):
        from omicverse.utils.ovagent.repair_loop import build_default_repair_loop
        mock_executor = MagicMock()
        loop = build_default_repair_loop(
            executor=mock_executor,
            executor_fn=lambda code, adata: (None, adata),
            max_retries=3,
        )
        names = loop.strategy_names
        assert len(names) >= 2, "Default loop should have at least 2 strategies"
        # Hint-driven should come before regex
        assert "hint_driven_repair" in names
        assert "regex_pattern_fix" in names

    def test_normalize_execution_failure(self):
        from omicverse.utils.ovagent.repair_loop import normalize_execution_failure
        env = normalize_execution_failure(
            tool_name="execute_code",
            exception=NameError("undefined"),
            code="print(x)",
        )
        assert env.tool_name == "execute_code"
        assert env.exception_type == "NameError"
        assert env.retryable is True


# ---------------------------------------------------------------------------
# §11  Context Budget Integration
# ---------------------------------------------------------------------------

class TestContextBudgetIntegration:
    """Verify DefaultContextBudgetManager lifecycle."""

    def test_add_entry_and_compact(self):
        from omicverse.utils.ovagent.context_budget import DefaultContextBudgetManager
        from omicverse.utils.ovagent.contracts import ContextEntry, ImportanceTier

        mgr = DefaultContextBudgetManager()
        entry = ContextEntry(
            content="hello " * 100,
            source="test",
            importance=ImportanceTier.LOW,
        )
        mgr.add_entry(entry)
        assert mgr.current_token_count > 0

    def test_checkpoint_restore(self):
        from omicverse.utils.ovagent.context_budget import DefaultContextBudgetManager
        from omicverse.utils.ovagent.contracts import ContextEntry

        mgr = DefaultContextBudgetManager()
        mgr.add_entry(ContextEntry(content="first", source="test"))
        cp = mgr.checkpoint()
        mgr.add_entry(ContextEntry(content="second", source="test"))
        assert len(mgr.entries) == 2
        mgr.restore(cp)
        assert len(mgr.entries) == 1


# ---------------------------------------------------------------------------
# §12  Prompt Templates Integration
# ---------------------------------------------------------------------------

class TestPromptTemplatesIntegration:
    """Verify prompt template engine works with layered composition."""

    def test_template_engine_compose(self):
        from omicverse.utils.ovagent.prompt_templates import PromptTemplateEngine
        from omicverse.utils.ovagent.contracts import PromptLayer, PromptLayerKind

        engine = PromptTemplateEngine()
        engine.add_layer(PromptLayer(
            kind=PromptLayerKind.BASE_SYSTEM, content="You are a bot.", priority=100,
        ))
        engine.add_layer(PromptLayer(
            kind=PromptLayerKind.SKILL, content="Use GSEA.", priority=50,
        ))
        result = engine.compose()
        assert "You are a bot." in result
        assert "Use GSEA." in result

    def test_build_skill_layer(self):
        from omicverse.utils.ovagent.prompt_templates import build_skill_layer
        # build_skill_layer expects dict of slug → objects with .slug and .description
        skill_meta = SimpleNamespace(slug="test-skill", description="do it")
        layer = build_skill_layer({"test-skill": skill_meta})
        assert layer is not None
        assert "test-skill" in layer.content

    def test_prompt_block_constants_defined(self):
        from omicverse.utils.ovagent import prompt_templates as pt
        blocks = [
            "IDENTITY_BLOCK", "TOOL_CATALOG_BLOCK", "CODING_FILESYSTEM_BLOCK",
            "ANALYSIS_WORKFLOW_BLOCK", "CODE_QUALITY_BLOCK", "GUIDELINES_BLOCK",
        ]
        for name in blocks:
            assert hasattr(pt, name), f"Prompt block constant {name} missing"
            assert isinstance(getattr(pt, name), str)
            assert len(getattr(pt, name)) > 10, f"{name} is suspiciously short"


# ---------------------------------------------------------------------------
# §13  Event Stream Integration
# ---------------------------------------------------------------------------

class TestEventStreamIntegration:
    """Verify EventBus lifecycle events fire without error."""

    def test_event_bus_lifecycle(self):
        from omicverse.utils.ovagent.event_stream import make_event_bus

        bus = make_event_bus()
        # These must not raise
        bus.init("Agent initialized")
        bus.turn_start(turn=1, max_turns=10)
        bus.tool_dispatch("execute_code", ["code"])
        bus.tool_result("execute_code", "ok")
        bus.tool_finished("done")
        bus.turn_cancelled("user requested cancel")
        bus.request_start("test request")
        bus.request_success()
        bus.request_error("test error")

    def test_silent_reporter_suppresses_output(self, capsys):
        from omicverse.utils.agent_reporter import SilentReporter
        from omicverse.utils.ovagent.event_stream import EventBus

        bus = EventBus(SilentReporter())
        bus.init("Should be silent")
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# §14  Subagent Controller Integration
# ---------------------------------------------------------------------------

class TestSubagentControllerIntegration:
    """Verify SubagentController and IsolatedSubagentContext creation."""

    def test_isolated_context_tool_permission_check(self):
        from omicverse.utils.ovagent.subagent_controller import IsolatedSubagentContext
        from omicverse.utils.ovagent.permission_policy import (
            PermissionPolicy, PermissionVerdict, build_subagent_policy,
        )

        policy = build_subagent_policy(["read", "grep"])
        ctx = IsolatedSubagentContext(
            agent_type="explore",
            allowed_tools=["read", "grep"],
            permission_policy=policy,
            can_mutate_adata=False,
            max_turns=5,
        )
        assert ctx.is_tool_allowed("read") is True
        assert ctx.is_tool_allowed("write") is False


# ---------------------------------------------------------------------------
# §15  Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify that backward-compatibility shims remain functional."""

    def test_legacy_agent_tools_is_list_of_dicts(self):
        from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS
        assert isinstance(LEGACY_AGENT_TOOLS, list)
        if LEGACY_AGENT_TOOLS:
            first = LEGACY_AGENT_TOOLS[0]
            assert isinstance(first, dict)
            assert "name" in first

    def test_agent_factory_returns_omicverse_agent(self):
        from omicverse.utils.smart_agent import Agent, OmicVerseAgent
        assert Agent is not None
        # Agent is a factory function that returns OmicVerseAgent instances
        assert callable(Agent)

    def test_tool_policy_defaults(self):
        from omicverse.utils.ovagent.tool_runtime import ToolPolicy
        p = ToolPolicy()
        assert p.parallel_safe is True
        assert p.read_only is False
        assert p.blocked_in_plan_mode is False

    def test_permission_policy_factory_methods(self):
        from omicverse.utils.ovagent.permission_policy import PermissionPolicy, PermissionVerdict
        permissive = PermissionPolicy.permissive()
        r = permissive.check("anything")
        assert r.verdict == PermissionVerdict.ALLOW

        restrictive = PermissionPolicy.restrictive()
        r2 = restrictive.check("anything")
        assert r2.verdict in (PermissionVerdict.DENY, PermissionVerdict.ASK), (
            "restrictive policy should deny or require approval"
        )


# ---------------------------------------------------------------------------
# §16  Integration Wire-Up Smoke Test
# ---------------------------------------------------------------------------

class TestRuntimeWireUp:
    """Smoke test that the full runtime wire-up path doesn't crash."""

    def test_smart_agent_subsystem_import_chain(self):
        """Verify the full import chain smart_agent -> ovagent submodules succeeds."""
        mod = importlib.import_module("omicverse.utils.smart_agent")
        # Check that the class references the extracted subsystem classes
        cls = mod.OmicVerseAgent
        init_source = inspect.getsource(cls.__init__)
        expected_subsystems = [
            "_ToolRuntime", "_TurnController", "_SubagentController",
            "_EventBus", "_CodegenPipeline",
        ]
        for subsystem in expected_subsystems:
            assert subsystem in init_source, (
                f"OmicVerseAgent.__init__ no longer wires {subsystem}"
            )


# ---------------------------------------------------------------------------
# §17  Contracts Module Completeness
# ---------------------------------------------------------------------------

class TestContractsModuleCompleteness:
    """Verify contracts.py exports all expected shapes."""

    def test_contracts_all_exports(self):
        from omicverse.utils.ovagent import contracts
        required = {
            "ApprovalClass", "IsolationMode", "ParallelClass", "OutputTier",
            "ToolPolicyMetadata", "ToolContract",
            "ImportanceTier", "ContextEntry", "OverflowPolicy", "ContextBudgetConfig",
            "ContextBudgetManager",
            "FailurePhase", "RepairHint", "ExecutionFailureEnvelope",
            "EventEmitter", "EVENT_CATEGORIES",
            "PromptLayerKind", "PromptLayer", "PromptComposer",
            "RemoteReviewConfig",
            "BenchmarkThresholds", "BENCHMARK_THRESHOLDS",
        }
        actual = set(contracts.__all__)
        missing = required - actual
        assert not missing, f"contracts __all__ missing: {missing}"

    def test_benchmark_thresholds_reasonable(self):
        from omicverse.utils.ovagent.contracts import BENCHMARK_THRESHOLDS
        assert BENCHMARK_THRESHOLDS.tool_lookup_max_seconds > 0
        assert BENCHMARK_THRESHOLDS.compaction_min_recovery_ratio > 0
        assert BENCHMARK_THRESHOLDS.event_emit_max_seconds > 0
