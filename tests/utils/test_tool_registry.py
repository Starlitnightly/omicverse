"""Contract tests for the tool metadata model and registry seam.

Validates that ToolMetadata, ToolRegistry, and build_default_registry()
satisfy the acceptance criteria for task-024:

- A concrete tool metadata model exists for the live codebase
- The model reuses existing tool_catalog definitions where possible
- Handler registration, alias normalization, policy attributes, and
  migration constraints are explicit
- Follow-on dispatch and scheduler tasks have a stable contract
"""

import pytest

from omicverse.utils.ovagent.tool_registry import (
    ApprovalClass,
    IsolationMode,
    OutputTier,
    ParallelClass,
    ToolMetadata,
    ToolRegistry,
    build_default_registry,
    _CATALOG_TOOL_POLICIES,
    _LEGACY_TOOL_POLICIES,
    _LEGACY_CATALOG_ALIASES,
)
from omicverse.utils.harness.tool_catalog import (
    CLAUDE_CODE_TOOLS,
    ToolCatalog,
    ToolDefinition,
    get_default_tool_catalog,
    normalize_tool_name,
)
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS


# ===================================================================
# 1. ToolMetadata model
# ===================================================================


class TestToolMetadataModel:
    """ToolMetadata is a concrete, frozen data model with all required fields."""

    def test_construction_with_catalog_definition(self):
        catalog = get_default_tool_catalog()
        read_def = catalog.get("Read")
        assert read_def is not None

        meta = ToolMetadata(
            canonical_name="Read",
            handler_key="read",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.standard,
            isolation_mode=IsolationMode.none,
            definition=read_def,
        )
        assert meta.canonical_name == "Read"
        assert meta.handler_key == "read"
        assert meta.definition is read_def

    def test_construction_with_legacy_schema(self):
        schema = {"name": "inspect_data", "description": "Inspect", "parameters": {}}
        meta = ToolMetadata(
            canonical_name="inspect_data",
            handler_key="inspect_data",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.standard,
            isolation_mode=IsolationMode.none,
            legacy_schema=schema,
        )
        assert meta.legacy_schema is not None
        assert meta.definition is None

    def test_frozen_immutability(self):
        meta = ToolMetadata(
            canonical_name="test",
            handler_key="test",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        with pytest.raises(AttributeError):
            meta.canonical_name = "changed"  # type: ignore[misc]

    def test_schema_property_catalog_tool(self):
        catalog = get_default_tool_catalog()
        bash_def = catalog.get("Bash")
        meta = ToolMetadata(
            canonical_name="Bash",
            handler_key="bash",
            approval_class=ApprovalClass.ask,
            parallel_class=ParallelClass.stateful,
            output_tier=OutputTier.verbose,
            isolation_mode=IsolationMode.sandbox,
            definition=bash_def,
        )
        schema = meta.schema
        assert schema["name"] == "Bash"
        assert "parameters" in schema

    def test_schema_property_legacy_tool(self):
        legacy = {"name": "finish", "description": "Done", "parameters": {"type": "object"}}
        meta = ToolMetadata(
            canonical_name="finish",
            handler_key="finish",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.exclusive,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
            legacy_schema=legacy,
        )
        schema = meta.schema
        assert schema["name"] == "finish"

    def test_schema_property_fallback(self):
        meta = ToolMetadata(
            canonical_name="synthetic",
            handler_key="synthetic",
            approval_class=ApprovalClass.deny,
            parallel_class=ParallelClass.exclusive,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        schema = meta.schema
        assert schema["name"] == "synthetic"

    def test_aliases_catalog_tool(self):
        catalog = get_default_tool_catalog()
        agent_def = catalog.get("Agent")
        meta = ToolMetadata(
            canonical_name="Agent",
            handler_key="agent",
            approval_class=ApprovalClass.ask,
            parallel_class=ParallelClass.stateful,
            output_tier=OutputTier.verbose,
            isolation_mode=IsolationMode.none,
            definition=agent_def,
        )
        aliases = meta.aliases
        assert "Agent" in aliases
        assert "delegate" in aliases  # legacy alias

    def test_aliases_legacy_tool(self):
        meta = ToolMetadata(
            canonical_name="inspect_data",
            handler_key="inspect_data",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.standard,
            isolation_mode=IsolationMode.none,
        )
        assert meta.aliases == ("inspect_data",)

    def test_to_dict_roundtrip(self):
        meta = ToolMetadata(
            canonical_name="Read",
            handler_key="read",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.standard,
            isolation_mode=IsolationMode.none,
            migration_notes="test note",
        )
        d = meta.to_dict()
        assert d["canonical_name"] == "Read"
        assert d["approval_class"] == "allow"
        assert d["migration_notes"] == "test note"

    def test_all_required_fields_present(self):
        """ToolMetadata has every field that dispatch and scheduler need."""
        required_fields = {
            "canonical_name", "handler_key",
            "approval_class", "parallel_class",
            "output_tier", "isolation_mode",
            "is_async",
        }
        meta = ToolMetadata(
            canonical_name="t", handler_key="t",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        for field_name in required_fields:
            assert hasattr(meta, field_name), f"Missing field: {field_name}"

    def test_hook_fields_present(self):
        """Optional lifecycle hooks are part of the contract."""
        meta = ToolMetadata(
            canonical_name="t", handler_key="t",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
            pre_exec_hook="validate_input",
            post_exec_hook="log_result",
            normalize_result_hook="truncate",
        )
        assert meta.pre_exec_hook == "validate_input"
        assert meta.post_exec_hook == "log_result"
        assert meta.normalize_result_hook == "truncate"


# ===================================================================
# 2. Policy enums
# ===================================================================


class TestPolicyEnums:
    """Policy enums are string-valued and exhaustive for their domain."""

    def test_approval_class_values(self):
        assert set(ApprovalClass) == {
            ApprovalClass.allow, ApprovalClass.ask, ApprovalClass.deny,
        }
        assert ApprovalClass.allow.value == "allow"

    def test_parallel_class_values(self):
        assert set(ParallelClass) == {
            ParallelClass.readonly, ParallelClass.stateful, ParallelClass.exclusive,
        }

    def test_output_tier_values(self):
        assert set(OutputTier) == {
            OutputTier.minimal, OutputTier.standard, OutputTier.verbose,
        }

    def test_isolation_mode_values(self):
        assert set(IsolationMode) == {
            IsolationMode.none, IsolationMode.sandbox, IsolationMode.worktree,
        }

    def test_enums_are_string_subclass(self):
        assert isinstance(ApprovalClass.allow, str)
        assert isinstance(ParallelClass.readonly, str)


# ===================================================================
# 3. ToolRegistry
# ===================================================================


class TestToolRegistry:

    def test_register_and_get(self):
        registry = ToolRegistry()
        meta = ToolMetadata(
            canonical_name="TestTool",
            handler_key="test_tool",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        registry.register(meta)
        assert registry.get("TestTool") is meta

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        meta = ToolMetadata(
            canonical_name="Dup",
            handler_key="dup",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        registry.register(meta)
        with pytest.raises(ValueError, match="Duplicate"):
            registry.register(meta)

    def test_resolve_name_direct(self):
        registry = ToolRegistry()
        meta = ToolMetadata(
            canonical_name="Read",
            handler_key="read",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.standard,
            isolation_mode=IsolationMode.none,
            definition=get_default_tool_catalog().get("Read"),
        )
        registry.register(meta)
        assert registry.resolve_name("Read") == "Read"

    def test_resolve_name_via_catalog_alias(self):
        """Alias normalization delegates to the catalog."""
        registry = build_default_registry()
        # "delegate" is a legacy alias for "Agent" in the catalog
        assert registry.resolve_name("delegate") == "Agent"
        # "web_fetch" normalises to "WebFetch"
        assert registry.resolve_name("web_fetch") == "WebFetch"
        # snake_case normalization
        assert registry.resolve_name("tool_search") == "ToolSearch"

    def test_resolve_name_empty(self):
        registry = ToolRegistry()
        assert registry.resolve_name("") == ""
        assert registry.resolve_name("nonexistent") == ""

    def test_get_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get("unknown") is None

    def test_handler_registration_and_lookup(self):
        registry = ToolRegistry()
        meta = ToolMetadata(
            canonical_name="Tool",
            handler_key="my_handler",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        registry.register(meta)

        handler = lambda: "result"
        registry.register_handler("my_handler", handler)
        assert registry.get_handler("Tool") is handler

    def test_get_handler_unbound_returns_none(self):
        registry = ToolRegistry()
        meta = ToolMetadata(
            canonical_name="Tool",
            handler_key="unbound",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        registry.register(meta)
        assert registry.get_handler("Tool") is None

    def test_validate_handlers(self):
        registry = ToolRegistry()
        meta_a = ToolMetadata(
            canonical_name="A", handler_key="ha",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        meta_b = ToolMetadata(
            canonical_name="B", handler_key="hb",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        registry.register(meta_a)
        registry.register(meta_b)
        registry.register_handler("ha", lambda: None)
        # "hb" is still unbound
        unbound = registry.validate_handlers()
        assert unbound == ["hb"]

    def test_all_entries(self):
        registry = ToolRegistry()
        meta = ToolMetadata(
            canonical_name="X", handler_key="x",
            approval_class=ApprovalClass.allow,
            parallel_class=ParallelClass.readonly,
            output_tier=OutputTier.minimal,
            isolation_mode=IsolationMode.none,
        )
        registry.register(meta)
        entries = registry.all_entries()
        assert len(entries) == 1
        assert entries[0] is meta

    def test_handler_keys(self):
        registry = build_default_registry()
        keys = registry.handler_keys()
        assert "bash" in keys
        assert "read" in keys
        assert "inspect_data" in keys

    def test_policy_summary(self):
        registry = build_default_registry()
        summary = registry.policy_summary()
        assert "Bash" in summary
        assert summary["Bash"]["approval_class"] == "ask"
        assert summary["Read"]["parallel_class"] == "readonly"


# ===================================================================
# 4. Default registry — coverage and correctness
# ===================================================================


class TestBuildDefaultRegistry:
    """build_default_registry() must cover every dispatched tool."""

    @pytest.fixture()
    def registry(self):
        return build_default_registry()

    def test_every_catalog_tool_has_metadata(self, registry):
        """All catalog ToolDefinitions have a registry entry."""
        catalog = get_default_tool_catalog()
        for tool_def in catalog.all_tools():
            meta = registry.get(tool_def.name)
            assert meta is not None, f"Missing registry entry for catalog tool: {tool_def.name}"

    def test_every_non_alias_legacy_tool_has_metadata(self, registry):
        """Legacy-only tools (not catalog aliases) have registry entries."""
        for tool_schema in LEGACY_AGENT_TOOLS:
            name = tool_schema["name"]
            if name in _LEGACY_CATALOG_ALIASES:
                continue
            meta = registry.get(name)
            assert meta is not None, f"Missing registry entry for legacy tool: {name}"

    def test_legacy_catalog_aliases_resolve_to_catalog_entry(self, registry):
        """Legacy names that are catalog aliases resolve correctly."""
        expected = {
            "delegate": "Agent",
            "web_fetch": "WebFetch",
            "web_search": "WebSearch",
            "search_skills": "Skill",
        }
        for alias, canonical in expected.items():
            meta = registry.get(alias)
            assert meta is not None, f"Alias {alias!r} should resolve"
            assert meta.canonical_name == canonical

    def test_catalog_tools_reuse_tool_definition(self, registry):
        """Catalog-sourced entries carry their ToolDefinition."""
        catalog = get_default_tool_catalog()
        for tool_def in catalog.all_tools():
            meta = registry.get(tool_def.name)
            assert meta is not None
            assert meta.definition is tool_def, (
                f"{tool_def.name}: definition should be the exact catalog ToolDefinition"
            )

    def test_legacy_tools_carry_legacy_schema(self, registry):
        """Legacy-only entries carry their raw dict schema."""
        legacy_names = set(_LEGACY_TOOL_POLICIES.keys())
        legacy_by_name = {t["name"]: t for t in LEGACY_AGENT_TOOLS}
        for name in legacy_names:
            meta = registry.get(name)
            assert meta is not None
            if name in legacy_by_name:
                assert meta.legacy_schema is not None
                assert meta.legacy_schema["name"] == name

    def test_no_duplicate_handler_keys(self, registry):
        """Each entry has a unique handler key."""
        seen: dict[str, str] = {}
        for meta in registry.all_entries():
            if meta.handler_key in seen:
                assert False, (
                    f"Duplicate handler_key {meta.handler_key!r}: "
                    f"{seen[meta.handler_key]} and {meta.canonical_name}"
                )
            seen[meta.handler_key] = meta.canonical_name

    def test_all_handlers_initially_unbound(self, registry):
        """Default registry is a seam — no handlers bound yet."""
        unbound = registry.validate_handlers()
        assert len(unbound) > 0, "Default registry should have unbound handler keys"
        assert len(unbound) == len(registry.handler_keys())

    def test_total_entry_count(self, registry):
        """Registry covers exactly all catalog + legacy-only tools."""
        expected = len(CLAUDE_CODE_TOOLS) + len(_LEGACY_TOOL_POLICIES)
        assert len(registry.all_entries()) == expected


# ===================================================================
# 5. Policy attribute correctness
# ===================================================================


class TestPolicyAttributes:
    """Policy classifications are consistent with existing tool properties."""

    @pytest.fixture()
    def registry(self):
        return build_default_registry()

    def test_high_risk_catalog_tools_require_approval(self, registry):
        """Catalog tools with high_risk=True must have approval_class=ask."""
        catalog = get_default_tool_catalog()
        for tool_def in catalog.all_tools():
            if tool_def.high_risk:
                meta = registry.get(tool_def.name)
                assert meta is not None
                assert meta.approval_class == ApprovalClass.ask, (
                    f"{tool_def.name}: high_risk=True but approval_class={meta.approval_class}"
                )

    def test_only_agent_is_async(self, registry):
        """Only Agent requires an async handler."""
        for meta in registry.all_entries():
            if meta.canonical_name == "Agent":
                assert meta.is_async is True
            else:
                assert meta.is_async is False, (
                    f"{meta.canonical_name}: unexpected is_async=True"
                )

    def test_enter_worktree_has_worktree_isolation(self, registry):
        meta = registry.get("EnterWorktree")
        assert meta is not None
        assert meta.isolation_mode == IsolationMode.worktree

    def test_code_execution_tools_have_sandbox_isolation(self, registry):
        for name in ("Bash", "execute_code", "run_snippet"):
            meta = registry.get(name)
            assert meta is not None
            assert meta.isolation_mode == IsolationMode.sandbox, (
                f"{name}: expected sandbox isolation"
            )

    def test_finish_is_exclusive(self, registry):
        meta = registry.get("finish")
        assert meta is not None
        assert meta.parallel_class == ParallelClass.exclusive

    def test_read_only_tools_are_readonly_parallel(self, registry):
        readonly_tools = ["Read", "Glob", "Grep", "ToolSearch", "inspect_data", "search_functions"]
        for name in readonly_tools:
            meta = registry.get(name)
            assert meta is not None
            assert meta.parallel_class == ParallelClass.readonly, (
                f"{name}: expected readonly parallel class"
            )

    def test_mode_changing_tools_are_exclusive(self, registry):
        for name in ("EnterPlanMode", "ExitPlanMode", "AskUserQuestion"):
            meta = registry.get(name)
            assert meta is not None
            assert meta.parallel_class == ParallelClass.exclusive, (
                f"{name}: expected exclusive parallel class"
            )


# ===================================================================
# 6. Migration constraints
# ===================================================================


class TestMigrationConstraints:

    @pytest.fixture()
    def registry(self):
        return build_default_registry()

    def test_agent_has_migration_notes(self, registry):
        meta = registry.get("Agent")
        assert meta is not None
        assert "subagent_controller" in meta.migration_notes

    def test_adata_tools_have_migration_notes(self, registry):
        for name in ("inspect_data", "execute_code", "run_snippet"):
            meta = registry.get(name)
            assert meta is not None
            assert "adata" in meta.migration_notes.lower(), (
                f"{name}: migration notes should mention adata dependency"
            )

    def test_finish_has_migration_notes(self, registry):
        meta = registry.get("finish")
        assert meta is not None
        assert "terminal" in meta.migration_notes.lower()

    def test_aliased_tools_have_migration_notes(self, registry):
        """Catalog tools that absorb legacy aliases document the relationship."""
        for name in ("Skill", "WebFetch", "WebSearch"):
            meta = registry.get(name)
            assert meta is not None
            assert "legacy" in meta.migration_notes.lower(), (
                f"{name}: should document legacy alias migration"
            )


# ===================================================================
# 7. Stable contract for follow-on tasks
# ===================================================================


class TestSchedulerDispatchContract:
    """The registry contract is sufficient for dispatch and scheduling."""

    @pytest.fixture()
    def registry(self):
        return build_default_registry()

    def test_every_entry_has_handler_key(self, registry):
        for meta in registry.all_entries():
            assert meta.handler_key, f"{meta.canonical_name}: empty handler_key"

    def test_every_entry_has_valid_approval_class(self, registry):
        for meta in registry.all_entries():
            assert isinstance(meta.approval_class, ApprovalClass)

    def test_every_entry_has_valid_parallel_class(self, registry):
        for meta in registry.all_entries():
            assert isinstance(meta.parallel_class, ParallelClass)

    def test_every_entry_has_valid_output_tier(self, registry):
        for meta in registry.all_entries():
            assert isinstance(meta.output_tier, OutputTier)

    def test_every_entry_has_valid_isolation_mode(self, registry):
        for meta in registry.all_entries():
            assert isinstance(meta.isolation_mode, IsolationMode)

    def test_every_entry_has_schema(self, registry):
        """Scheduler needs the JSON schema to validate tool call arguments."""
        for meta in registry.all_entries():
            schema = meta.schema
            assert isinstance(schema, dict)
            assert "name" in schema

    def test_policy_summary_complete(self, registry):
        """policy_summary returns one row per entry with all policy fields."""
        summary = registry.policy_summary()
        assert len(summary) == len(registry.all_entries())
        for name, policies in summary.items():
            assert "approval_class" in policies
            assert "parallel_class" in policies
            assert "output_tier" in policies
            assert "isolation_mode" in policies

    def test_catalog_policy_table_covers_all_catalog_tools(self):
        """The policy table has an entry for every tool in CLAUDE_CODE_TOOLS."""
        catalog_names = {t.name for t in CLAUDE_CODE_TOOLS}
        policy_names = set(_CATALOG_TOOL_POLICIES.keys())
        assert catalog_names == policy_names, (
            f"Missing from policy table: {catalog_names - policy_names}; "
            f"Extra in policy table: {policy_names - catalog_names}"
        )
