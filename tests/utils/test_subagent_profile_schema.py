"""Tests for configurable subagent profile overrides in AgentConfig (task-015).

Covers:
  - AgentConfig.subagent_overrides is an optional schema field
  - SUBAGENT_CONFIGS remain the default when no overrides are provided
  - Partial overrides merge cleanly with defaults
  - Flat kwargs compatibility is preserved
  - Invalid overrides are rejected
  - Overrides do not mutate the global SUBAGENT_CONFIGS defaults
"""

import os
import sys
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
    reason="Subagent profile tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omicverse.utils.agent_config import (
    AgentConfig,
    SUBAGENT_CONFIGS,
    SubagentConfig,
)


# ===================================================================
# 1. Schema field existence and defaults
# ===================================================================

class TestSubagentOverridesField:

    def test_field_exists_on_agent_config(self):
        cfg = AgentConfig()
        assert hasattr(cfg, "subagent_overrides")

    def test_default_is_empty_dict(self):
        cfg = AgentConfig()
        assert cfg.subagent_overrides == {}

    def test_field_accepts_dict(self):
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 10}})
        assert cfg.subagent_overrides == {"explore": {"max_turns": 10}}


# ===================================================================
# 2. Defaults preserved when no overrides
# ===================================================================

class TestDefaultsPreserved:

    def test_get_subagent_config_returns_defaults(self):
        cfg = AgentConfig()
        for name, expected in SUBAGENT_CONFIGS.items():
            result = cfg.get_subagent_config(name)
            assert result.agent_type == expected.agent_type
            assert result.allowed_tools == expected.allowed_tools
            assert result.max_turns == expected.max_turns
            assert result.can_mutate_adata == expected.can_mutate_adata
            assert result.temperature == expected.temperature

    def test_all_known_profiles_accessible(self):
        cfg = AgentConfig()
        for name in ["explore", "plan", "execute"]:
            result = cfg.get_subagent_config(name)
            assert isinstance(result, SubagentConfig)
            assert result.agent_type == name


# ===================================================================
# 3. Partial overrides merge cleanly
# ===================================================================

class TestPartialOverrideMerge:

    def test_single_field_override(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 20},
        })
        result = cfg.get_subagent_config("explore")
        # Overridden field
        assert result.max_turns == 20
        # Non-overridden fields match defaults
        default = SUBAGENT_CONFIGS["explore"]
        assert result.allowed_tools == default.allowed_tools
        assert result.can_mutate_adata == default.can_mutate_adata
        assert result.temperature == default.temperature

    def test_multiple_field_override(self):
        cfg = AgentConfig(subagent_overrides={
            "plan": {"max_turns": 15, "temperature": 0.5},
        })
        result = cfg.get_subagent_config("plan")
        assert result.max_turns == 15
        assert result.temperature == 0.5
        # Remaining fields unchanged
        default = SUBAGENT_CONFIGS["plan"]
        assert result.allowed_tools == default.allowed_tools
        assert result.can_mutate_adata == default.can_mutate_adata

    def test_override_allowed_tools(self):
        custom_tools = ["inspect_data", "finish"]
        cfg = AgentConfig(subagent_overrides={
            "execute": {"allowed_tools": custom_tools},
        })
        result = cfg.get_subagent_config("execute")
        assert result.allowed_tools == custom_tools
        # Other fields unchanged
        default = SUBAGENT_CONFIGS["execute"]
        assert result.max_turns == default.max_turns
        assert result.can_mutate_adata == default.can_mutate_adata

    def test_override_can_mutate_adata(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"can_mutate_adata": True},
        })
        result = cfg.get_subagent_config("explore")
        assert result.can_mutate_adata is True

    def test_multiple_profiles_overridden(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 3},
            "execute": {"temperature": 0.9},
        })
        explore = cfg.get_subagent_config("explore")
        execute = cfg.get_subagent_config("execute")
        plan = cfg.get_subagent_config("plan")

        assert explore.max_turns == 3
        assert execute.temperature == 0.9
        # Plan unaffected
        assert plan.max_turns == SUBAGENT_CONFIGS["plan"].max_turns

    def test_non_overridden_profile_returns_defaults(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 99},
        })
        plan = cfg.get_subagent_config("plan")
        default = SUBAGENT_CONFIGS["plan"]
        assert plan.max_turns == default.max_turns
        assert plan.temperature == default.temperature


# ===================================================================
# 4. Global SUBAGENT_CONFIGS not mutated
# ===================================================================

class TestNoGlobalMutation:

    def test_override_does_not_mutate_global(self):
        original_max_turns = SUBAGENT_CONFIGS["explore"].max_turns
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 999},
        })
        result = cfg.get_subagent_config("explore")
        assert result.max_turns == 999
        # Global default must be unchanged
        assert SUBAGENT_CONFIGS["explore"].max_turns == original_max_turns

    def test_repeated_calls_do_not_accumulate(self):
        cfg = AgentConfig(subagent_overrides={
            "plan": {"max_turns": 50},
        })
        r1 = cfg.get_subagent_config("plan")
        r2 = cfg.get_subagent_config("plan")
        assert r1.max_turns == 50
        assert r2.max_turns == 50
        # Each call returns a fresh copy
        assert r1 is not r2


# ===================================================================
# 5. Error handling
# ===================================================================

class TestOverrideErrors:

    def test_unknown_agent_type_raises_key_error(self):
        cfg = AgentConfig()
        with pytest.raises(KeyError, match="nonexistent"):
            cfg.get_subagent_config("nonexistent")

    def test_invalid_field_name_raises_value_error(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"bogus_field": 42},
        })
        with pytest.raises(ValueError, match="bogus_field"):
            cfg.get_subagent_config("explore")

    def test_valid_error_message_lists_valid_fields(self):
        cfg = AgentConfig(subagent_overrides={
            "explore": {"nope": 1},
        })
        with pytest.raises(ValueError, match="max_turns"):
            cfg.get_subagent_config("explore")


# ===================================================================
# 6. Flat kwargs compatibility
# ===================================================================

class TestFlatKwargsCompatibility:

    def test_from_flat_kwargs_without_overrides(self):
        cfg = AgentConfig.from_flat_kwargs(model="gpt-4")
        assert cfg.subagent_overrides == {}
        # All default profiles still accessible
        for name in SUBAGENT_CONFIGS:
            result = cfg.get_subagent_config(name)
            assert isinstance(result, SubagentConfig)

    def test_from_flat_kwargs_with_overrides(self):
        overrides = {"explore": {"max_turns": 12, "temperature": 0.7}}
        cfg = AgentConfig.from_flat_kwargs(
            model="gpt-4",
            subagent_overrides=overrides,
        )
        assert cfg.subagent_overrides == overrides
        result = cfg.get_subagent_config("explore")
        assert result.max_turns == 12
        assert result.temperature == 0.7

    def test_from_flat_kwargs_preserves_other_fields(self):
        cfg = AgentConfig.from_flat_kwargs(
            model="claude-3-opus",
            subagent_overrides={"plan": {"max_turns": 4}},
        )
        assert cfg.llm.model == "claude-3-opus"
        assert cfg.subagent_overrides == {"plan": {"max_turns": 4}}


# ===================================================================
# 7. Return type contract
# ===================================================================

class TestReturnTypeContract:

    def test_get_subagent_config_returns_subagent_config(self):
        cfg = AgentConfig()
        for name in SUBAGENT_CONFIGS:
            result = cfg.get_subagent_config(name)
            assert isinstance(result, SubagentConfig)

    def test_overridden_result_is_subagent_config(self):
        cfg = AgentConfig(subagent_overrides={
            "execute": {"max_turns": 25},
        })
        result = cfg.get_subagent_config("execute")
        assert isinstance(result, SubagentConfig)
        assert result.agent_type == "execute"
