"""Tests for ovagent.prompt_templates — prompt composition contract.

Verifies:
  - PromptTemplateEngine renders deterministically (sorted by priority, name)
  - PromptOverlay is immutable (frozen dataclass)
  - Predefined template constants contain expected content
  - build_agentic_engine and build_subagent_engine produce valid engines
  - PromptBuilder delegates to the template engine correctly
  - Workflow/skill/provider overlay behaviour
  - Export contract in __init__.py
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any
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
    reason="Prompt template contract tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
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

# Now import the modules under test
from omicverse.utils.ovagent.prompt_templates import (
    BASE_IDENTITY,
    CODE_ONLY_MODE,
    DELEGATION_STRATEGY,
    GUIDELINES,
    SUBAGENT_BASES,
    TOOL_INSTRUCTIONS,
    WEB_ACCESS,
    WORKFLOW_STEPS,
    PromptOverlay,
    PromptTemplateEngine,
    build_agentic_engine,
    build_subagent_engine,
)
from omicverse.utils.ovagent.prompt_builder import (
    CODE_QUALITY_RULES,
    PromptBuilder,
)

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Helpers — minimal AgentContext stub
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    code_only: bool = False,
    skills: dict | None = None,
    ov_runtime: Any = None,
) -> MagicMock:
    """Return a minimal mock satisfying the AgentContext protocol."""
    ctx = MagicMock()
    ctx._code_only_mode = code_only
    ctx._ov_runtime = ov_runtime
    if skills is not None:
        reg = MagicMock()
        reg.skill_metadata = skills
        ctx.skill_registry = reg
    else:
        ctx.skill_registry = None
    return ctx


class _FakeSkillMeta:
    def __init__(self, slug: str, description: str) -> None:
        self.slug = slug
        self.description = description


# ===========================================================================
# 1. PromptOverlay unit tests
# ===========================================================================


class TestPromptOverlay:
    def test_fields(self):
        o = PromptOverlay("name", "content", priority=42)
        assert o.name == "name"
        assert o.content == "content"
        assert o.priority == 42

    def test_default_priority(self):
        o = PromptOverlay("x", "y")
        assert o.priority == 100

    def test_frozen(self):
        o = PromptOverlay("x", "y")
        with pytest.raises(AttributeError):
            o.name = "z"  # type: ignore[misc]


# ===========================================================================
# 2. PromptTemplateEngine unit tests
# ===========================================================================


class TestPromptTemplateEngine:
    def test_empty_engine(self):
        engine = PromptTemplateEngine()
        assert engine.render() == ""

    def test_base_only(self):
        engine = PromptTemplateEngine()
        engine.set_base("Hello")
        assert engine.render() == "Hello"

    def test_overlays_sorted_by_priority(self):
        engine = PromptTemplateEngine()
        engine.set_base("base")
        engine.add_overlay(PromptOverlay("c", "third", priority=30))
        engine.add_overlay(PromptOverlay("a", "first", priority=10))
        engine.add_overlay(PromptOverlay("b", "second", priority=20))
        assert engine.render() == "base\n\nfirst\n\nsecond\n\nthird"

    def test_overlays_sorted_by_name_on_tie(self):
        engine = PromptTemplateEngine()
        engine.set_base("base")
        engine.add_overlay(PromptOverlay("beta", "B", priority=10))
        engine.add_overlay(PromptOverlay("alpha", "A", priority=10))
        assert engine.render() == "base\n\nA\n\nB"

    def test_deterministic_render(self):
        """Same overlays always produce the same output."""
        engine = PromptTemplateEngine()
        engine.set_base("base")
        engine.add_overlay(PromptOverlay("x", "X", priority=50))
        engine.add_overlay(PromptOverlay("y", "Y", priority=10))
        first = engine.render()
        second = engine.render()
        assert first == second

    def test_replace_overlay(self):
        engine = PromptTemplateEngine()
        engine.add_overlay(PromptOverlay("x", "old"))
        engine.add_overlay(PromptOverlay("x", "new"))
        assert "new" in engine.render()
        assert "old" not in engine.render()

    def test_remove_overlay(self):
        engine = PromptTemplateEngine()
        engine.add_overlay(PromptOverlay("x", "content"))
        engine.remove_overlay("x")
        assert engine.render() == ""

    def test_remove_nonexistent_is_noop(self):
        engine = PromptTemplateEngine()
        engine.remove_overlay("missing")  # should not raise

    def test_has_overlay(self):
        engine = PromptTemplateEngine()
        assert not engine.has_overlay("x")
        engine.add_overlay(PromptOverlay("x", "y"))
        assert engine.has_overlay("x")

    def test_get_overlay(self):
        engine = PromptTemplateEngine()
        assert engine.get_overlay("x") is None
        o = PromptOverlay("x", "y", priority=5)
        engine.add_overlay(o)
        assert engine.get_overlay("x") == o

    def test_overlay_names_in_order(self):
        engine = PromptTemplateEngine()
        engine.add_overlay(PromptOverlay("c", "C", priority=30))
        engine.add_overlay(PromptOverlay("a", "A", priority=10))
        engine.add_overlay(PromptOverlay("b", "B", priority=20))
        assert engine.overlay_names == ["a", "b", "c"]

    def test_empty_content_skipped(self):
        engine = PromptTemplateEngine()
        engine.set_base("base")
        engine.add_overlay(PromptOverlay("empty", "", priority=10))
        engine.add_overlay(PromptOverlay("real", "content", priority=20))
        assert engine.render() == "base\n\ncontent"

    def test_base_property(self):
        engine = PromptTemplateEngine()
        assert engine.base == ""
        engine.set_base("hello")
        assert engine.base == "hello"


# ===========================================================================
# 3. Predefined template constants
# ===========================================================================


class TestPredefinedTemplates:
    def test_base_identity_mentions_omicverse(self):
        assert "OmicVerse Agent" in BASE_IDENTITY

    def test_tool_instructions_mentions_toolsearch(self):
        assert "ToolSearch" in TOOL_INSTRUCTIONS

    def test_web_access_mentions_webfetch(self):
        assert "WebFetch" in WEB_ACCESS

    def test_workflow_steps_has_numbered_steps(self):
        assert "1." in WORKFLOW_STEPS
        assert "6." in WORKFLOW_STEPS

    def test_guidelines_mentions_inspect(self):
        assert "inspect data" in GUIDELINES

    def test_delegation_mentions_explore(self):
        assert "explore" in DELEGATION_STRATEGY

    def test_code_only_mode_mentions_claw(self):
        assert "CLAW" in CODE_ONLY_MODE

    def test_subagent_bases_has_all_types(self):
        assert set(SUBAGENT_BASES.keys()) == {"explore", "plan", "execute"}


# ===========================================================================
# 4. Factory helpers
# ===========================================================================


class TestBuildAgenticEngine:
    def test_returns_engine(self):
        engine = build_agentic_engine()
        assert isinstance(engine, PromptTemplateEngine)

    def test_base_is_identity(self):
        engine = build_agentic_engine()
        assert engine.base == BASE_IDENTITY

    def test_has_workflow_overlays(self):
        engine = build_agentic_engine()
        expected = {"tool_instructions", "web_access", "workflow", "code_quality",
                    "guidelines", "delegation"}
        assert expected.issubset(set(engine.overlay_names))

    def test_render_contains_key_sections(self):
        rendered = build_agentic_engine().render()
        assert "OmicVerse Agent" in rendered
        assert "CODING / FILESYSTEM" in rendered
        assert "WEB ACCESS" in rendered
        assert "WORKFLOW" in rendered
        assert "MANDATORY CODE QUALITY" in rendered
        assert "DELEGATION STRATEGY" in rendered

    def test_overlay_order_is_deterministic(self):
        names_a = build_agentic_engine().overlay_names
        names_b = build_agentic_engine().overlay_names
        assert names_a == names_b

    def test_no_code_only_by_default(self):
        engine = build_agentic_engine()
        assert not engine.has_overlay("code_only_mode")


class TestBuildSubagentEngine:
    @pytest.mark.parametrize("agent_type", ["explore", "plan", "execute"])
    def test_returns_engine(self, agent_type):
        engine = build_subagent_engine(agent_type)
        assert isinstance(engine, PromptTemplateEngine)

    def test_explore_has_no_code_quality(self):
        engine = build_subagent_engine("explore")
        assert not engine.has_overlay("code_quality")

    def test_plan_has_code_quality(self):
        engine = build_subagent_engine("plan")
        assert engine.has_overlay("code_quality")

    def test_execute_has_code_quality_and_packages(self):
        engine = build_subagent_engine("execute")
        assert engine.has_overlay("code_quality")
        assert engine.has_overlay("packages")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown subagent type"):
            build_subagent_engine("invalid")


# ===========================================================================
# 5. PromptBuilder integration — engine-backed composition
# ===========================================================================


class TestPromptBuilderAgenticEngine:
    def test_basic_prompt_contains_identity(self):
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        prompt = builder.build_agentic_system_prompt()
        assert "OmicVerse Agent" in prompt

    def test_code_only_mode_overlay(self):
        ctx = _make_ctx(code_only=True)
        builder = PromptBuilder(ctx)
        prompt = builder.build_agentic_system_prompt()
        assert "CLAW CODE-ONLY MODE" in prompt

    def test_skill_overlay_added(self):
        skills = {
            "clustering": _FakeSkillMeta("clustering", "Run Leiden clustering"),
            "deg": _FakeSkillMeta("deg", "Differential expression analysis"),
        }
        ctx = _make_ctx(skills=skills)
        builder = PromptBuilder(ctx)
        prompt = builder.build_agentic_system_prompt()
        assert "clustering" in prompt
        assert "deg" in prompt

    def test_runtime_compose_called(self):
        runtime = MagicMock()
        runtime.compose_system_prompt.return_value = "COMPOSED"
        ctx = _make_ctx(ov_runtime=runtime)
        builder = PromptBuilder(ctx)
        result = builder.build_agentic_system_prompt()
        runtime.compose_system_prompt.assert_called_once()
        assert result == "COMPOSED"

    def test_no_runtime_no_compose(self):
        ctx = _make_ctx(ov_runtime=None)
        builder = PromptBuilder(ctx)
        prompt = builder.build_agentic_system_prompt()
        assert "OmicVerse Agent" in prompt


class TestPromptBuilderSubagentEngine:
    @pytest.mark.parametrize("agent_type", ["explore", "plan", "execute"])
    def test_builds_without_error(self, agent_type):
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        prompt = builder.build_subagent_system_prompt(agent_type)
        assert len(prompt) > 0

    def test_explore_contains_inspector_identity(self):
        ctx = _make_ctx()
        prompt = PromptBuilder(ctx).build_explore_prompt("")
        assert "data inspector" in prompt

    def test_plan_contains_architect_identity(self):
        ctx = _make_ctx()
        prompt = PromptBuilder(ctx).build_plan_prompt("")
        assert "workflow architect" in prompt

    def test_execute_contains_executor_identity(self):
        ctx = _make_ctx()
        prompt = PromptBuilder(ctx).build_execute_prompt("")
        assert "code executor" in prompt

    def test_context_appended_explore(self):
        ctx = _make_ctx()
        prompt = PromptBuilder(ctx).build_explore_prompt("extra info")
        assert "extra info" in prompt

    def test_context_appended_plan(self):
        ctx = _make_ctx()
        prompt = PromptBuilder(ctx).build_plan_prompt("extra info")
        assert "extra info" in prompt

    def test_context_appended_execute(self):
        ctx = _make_ctx()
        prompt = PromptBuilder(ctx).build_execute_prompt("extra info")
        assert "extra info" in prompt

    def test_plan_skill_overlay(self):
        skills = {
            "spatial": _FakeSkillMeta("spatial", "Spatial transcriptomics"),
        }
        ctx = _make_ctx(skills=skills)
        prompt = PromptBuilder(ctx).build_plan_prompt("")
        assert "spatial" in prompt

    def test_unknown_type_raises(self):
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="Unknown subagent type"):
            PromptBuilder(ctx).build_subagent_system_prompt("unknown")

    def test_user_message_basic(self):
        ctx = _make_ctx()
        msg = PromptBuilder(ctx).build_subagent_user_message("do analysis", None)
        assert "do analysis" in msg

    def test_user_message_with_adata(self):
        ctx = _make_ctx()
        adata = MagicMock()
        adata.shape = (1000, 2000)
        msg = PromptBuilder(ctx).build_subagent_user_message("do analysis", adata)
        assert "1000" in msg
        assert "2000" in msg


# ===========================================================================
# 6. Composition determinism — the core contract
# ===========================================================================


class TestCompositionDeterminism:
    """Verify the prompt composition contract: base + workflow + skill +
    runtime/provider overlays render deterministically."""

    def test_agentic_prompt_deterministic(self):
        skills = {
            "z_last": _FakeSkillMeta("z_last", "Last skill"),
            "a_first": _FakeSkillMeta("a_first", "First skill"),
        }
        ctx = _make_ctx(code_only=True, skills=skills)
        builder = PromptBuilder(ctx)
        p1 = builder.build_agentic_system_prompt()
        p2 = builder.build_agentic_system_prompt()
        assert p1 == p2

    def test_subagent_prompt_deterministic(self):
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        for agent_type in ("explore", "plan", "execute"):
            p1 = builder.build_subagent_system_prompt(agent_type, "ctx")
            p2 = builder.build_subagent_system_prompt(agent_type, "ctx")
            assert p1 == p2, "Subagent prompt for " + agent_type + " not deterministic"

    def test_overlay_order_preserved_across_engines(self):
        """Two independently constructed engines with the same overlays
        must render identically."""
        def _build():
            engine = PromptTemplateEngine()
            engine.set_base("base")
            engine.add_overlay(PromptOverlay("z", "Z", priority=10))
            engine.add_overlay(PromptOverlay("a", "A", priority=10))
            engine.add_overlay(PromptOverlay("m", "M", priority=5))
            return engine.render()

        assert _build() == _build()


# ===========================================================================
# 7. Export contract — prompt_templates symbols in __init__.py
# ===========================================================================


class TestExportContract:
    def test_prompt_overlay_exported(self):
        import omicverse.utils.ovagent as pkg
        assert hasattr(pkg, "PromptOverlay")
        assert pkg.PromptOverlay is PromptOverlay

    def test_prompt_template_engine_exported(self):
        import omicverse.utils.ovagent as pkg
        assert hasattr(pkg, "PromptTemplateEngine")
        assert pkg.PromptTemplateEngine is PromptTemplateEngine

    def test_build_agentic_engine_exported(self):
        import omicverse.utils.ovagent as pkg
        assert hasattr(pkg, "build_agentic_engine")
        assert pkg.build_agentic_engine is build_agentic_engine

    def test_build_subagent_engine_exported(self):
        import omicverse.utils.ovagent as pkg
        assert hasattr(pkg, "build_subagent_engine")
        assert pkg.build_subagent_engine is build_subagent_engine

    def test_all_contains_new_symbols(self):
        import omicverse.utils.ovagent as pkg
        for name in ("PromptOverlay", "PromptTemplateEngine",
                     "build_agentic_engine", "build_subagent_engine"):
            assert name in pkg.__all__, name + " missing from __all__"
