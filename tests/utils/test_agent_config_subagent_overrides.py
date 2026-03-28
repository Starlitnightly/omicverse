"""Tests for configurable subagent profile overrides in AgentConfig (task-015).

Covers:
  - AgentConfig has subagent_overrides as an optional field
  - SUBAGENT_CONFIGS remain the default when no overrides are provided
  - Partial overrides merge cleanly with defaults
  - Invalid override keys are rejected
  - Unknown agent types are rejected
  - from_flat_kwargs preserves subagent_overrides passthrough
  - Default AgentConfig construction is unchanged
"""

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard: require harness env-var
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Subagent override tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
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

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ===================================================================
# 1. subagent_overrides field exists and defaults to empty
# ===================================================================

class TestSubagentOverridesField:

    def test_field_exists_on_agent_config(self):
        cfg = AgentConfig()
        assert hasattr(cfg, "subagent_overrides")

    def test_default_is_empty_dict(self):
        cfg = AgentConfig()
        assert cfg.subagent_overrides == {}

    def test_field_accepts_dict(self):
        overrides = {"explore": {"max_turns": 10}}
        cfg = AgentConfig(subagent_overrides=overrides)
        assert cfg.subagent_overrides == overrides


# ===================================================================
# 2. SUBAGENT_CONFIGS remain the default source of truth
# ===================================================================

class TestDefaultSubagentConfigs:

    def test_get_subagent_config_returns_default_when_no_overrides(self):
        cfg = AgentConfig()
        for agent_type in SUBAGENT_CONFIGS:
            result = cfg.get_subagent_config(agent_type)
            default = SUBAGENT_CONFIGS[agent_type]
            assert result.agent_type == default.agent_type
            assert result.allowed_tools == default.allowed_tools
            assert result.max_turns == default.max_turns
            assert result.can_mutate_adata == default.can_mutate_adata
            assert result.temperature == default.temperature

    def test_get_subagent_config_does_not_mutate_global_defaults(self):
        overrides = {"explore": {"max_turns": 99}}
        cfg = AgentConfig(subagent_overrides=overrides)
        cfg.get_subagent_config("explore")
        # Global default must be unchanged
        assert SUBAGENT_CONFIGS["explore"].max_turns == 5

    def test_all_known_agent_types_are_accessible(self):
        cfg = AgentConfig()
        for agent_type in ("explore", "plan", "execute"):
            result = cfg.get_subagent_config(agent_type)
            assert isinstance(result, SubagentConfig)


# ===================================================================
# 3. Partial overrides merge cleanly with defaults
# ===================================================================

class TestPartialOverrideMerge:

    def test_single_field_override(self):
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 20}})
        result = cfg.get_subagent_config("explore")
        assert result.max_turns == 20
        # Other fields keep defaults
        assert result.temperature == SUBAGENT_CONFIGS["explore"].temperature
        assert result.allowed_tools == SUBAGENT_CONFIGS["explore"].allowed_tools
        assert result.can_mutate_adata == SUBAGENT_CONFIGS["explore"].can_mutate_adata

    def test_multiple_field_override(self):
        cfg = AgentConfig(subagent_overrides={
            "plan": {"max_turns": 15, "temperature": 0.7}
        })
        result = cfg.get_subagent_config("plan")
        assert result.max_turns == 15
        assert result.temperature == 0.7
        # Unchanged fields
        assert result.allowed_tools == SUBAGENT_CONFIGS["plan"].allowed_tools

    def test_override_allowed_tools(self):
        custom_tools = ["inspect_data", "finish"]
        cfg = AgentConfig(subagent_overrides={
            "execute": {"allowed_tools": custom_tools}
        })
        result = cfg.get_subagent_config("execute")
        assert result.allowed_tools == custom_tools
        # Other fields unchanged
        assert result.max_turns == SUBAGENT_CONFIGS["execute"].max_turns

    def test_override_can_mutate_adata(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"can_mutate_adata": True}
        })
        result = cfg.get_subagent_config("explore")
        assert result.can_mutate_adata is True

    def test_multiple_profiles_overridden_independently(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 3},
            "execute": {"temperature": 0.5},
        })
        explore = cfg.get_subagent_config("explore")
        execute = cfg.get_subagent_config("execute")
        assert explore.max_turns == 3
        assert explore.temperature == SUBAGENT_CONFIGS["explore"].temperature
        assert execute.temperature == 0.5
        assert execute.max_turns == SUBAGENT_CONFIGS["execute"].max_turns

    def test_non_overridden_profile_returns_default(self):
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 99}})
        plan = cfg.get_subagent_config("plan")
        assert plan.max_turns == SUBAGENT_CONFIGS["plan"].max_turns


# ===================================================================
# 4. Error handling
# ===================================================================

class TestSubagentOverrideErrors:

    def test_unknown_agent_type_raises_key_error(self):
        cfg = AgentConfig()
        with pytest.raises(KeyError, match="Unknown subagent type"):
            cfg.get_subagent_config("nonexistent")

    def test_invalid_override_field_raises_value_error(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"bogus_field": 42}
        })
        with pytest.raises(ValueError, match="Unknown SubagentConfig field"):
            cfg.get_subagent_config("explore")

    def test_error_message_lists_valid_fields(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"not_a_field": True}
        })
        with pytest.raises(ValueError, match="max_turns"):
            cfg.get_subagent_config("explore")


# ===================================================================
# 5. from_flat_kwargs compatibility
# ===================================================================

class TestFlatKwargsCompatibility:

    def test_from_flat_kwargs_without_overrides(self):
        cfg = AgentConfig.from_flat_kwargs(model="test-model")
        assert cfg.subagent_overrides == {}
        assert cfg.llm.model == "test-model"

    def test_from_flat_kwargs_with_overrides(self):
        overrides = {"explore": {"max_turns": 42}}
        cfg = AgentConfig.from_flat_kwargs(
            model="test-model",
            subagent_overrides=overrides,
        )
        assert cfg.subagent_overrides == overrides
        result = cfg.get_subagent_config("explore")
        assert result.max_turns == 42

    def test_from_flat_kwargs_preserves_all_other_fields(self):
        """Ensure adding subagent_overrides doesn't break existing kwargs."""
        cfg = AgentConfig.from_flat_kwargs(
            model="gpt-4",
            enable_reflection=False,
            notebook_timeout=300,
            subagent_overrides={"plan": {"temperature": 0.9}},
        )
        assert cfg.llm.model == "gpt-4"
        assert cfg.reflection.enabled is False
        assert cfg.execution.timeout == 300
        assert cfg.subagent_overrides == {"plan": {"temperature": 0.9}}

    def test_default_construction_unchanged(self):
        """AgentConfig() with no args still works identically."""
        cfg = AgentConfig()
        assert cfg.llm.model == "gemini-2.5-flash"
        assert cfg.reflection.enabled is True
        assert cfg.subagent_overrides == {}
        # get_subagent_config returns defaults
        for name in SUBAGENT_CONFIGS:
            result = cfg.get_subagent_config(name)
            assert result.max_turns == SUBAGENT_CONFIGS[name].max_turns


# ===================================================================
# 6. Returned config is a copy, not a reference to the global
# ===================================================================

class TestReturnedConfigIsolation:

    def test_mutating_returned_config_does_not_affect_global(self):
        cfg = AgentConfig()
        result = cfg.get_subagent_config("explore")
        result.max_turns = 999
        assert SUBAGENT_CONFIGS["explore"].max_turns == 5

    def test_successive_calls_return_independent_copies(self):
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 7}})
        a = cfg.get_subagent_config("explore")
        b = cfg.get_subagent_config("explore")
        assert a.max_turns == 7
        a.max_turns = 100
        assert b.max_turns == 7
