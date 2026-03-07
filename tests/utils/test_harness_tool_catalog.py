import os

import pytest

from omicverse.utils.harness.tool_catalog import (
    CORE_TOOL_NAMES,
    get_default_loaded_tool_names,
    get_default_tool_catalog,
    get_visible_tool_schemas,
    resolve_tool_name,
)


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Harness tests are server-only and require OV_AGENT_RUN_HARNESS_TESTS=1.",
)


def test_claude_code_tool_catalog_has_unique_names():
    catalog = get_default_tool_catalog()
    tools = catalog.all_tools()
    names = [tool.name for tool in tools]

    assert len(tools) == 24
    assert len(names) == len(set(names))


def test_tool_catalog_entries_use_object_parameter_schemas():
    catalog = get_default_tool_catalog()

    for tool in catalog.all_tools():
        schema = tool.to_tool_schema()
        assert schema["name"] == tool.name
        assert schema["description"].strip()
        assert schema["parameters"]["type"] == "object"
        assert isinstance(schema["parameters"]["properties"], dict)


def test_tool_catalog_legacy_aliases_and_default_visibility_are_exposed():
    assert resolve_tool_name("delegate") == "Agent"
    assert resolve_tool_name("search_skills") == "Skill"
    assert resolve_tool_name("web_fetch") == "WebFetch"
    assert resolve_tool_name("web_search") == "WebSearch"

    loaded = set(get_default_loaded_tool_names())
    assert set(CORE_TOOL_NAMES).issubset(loaded)
    assert {"Agent", "AskUserQuestion", "Skill", "WebFetch", "WebSearch"}.issubset(loaded)

    visible = {tool["name"] for tool in get_visible_tool_schemas(loaded)}
    assert {"ToolSearch", "Bash", "Read", "Agent", "AskUserQuestion", "Skill", "WebFetch", "WebSearch"}.issubset(visible)
