from types import SimpleNamespace

from omicverse.utils.ovagent.tool_runtime import ToolRuntime


class _DummyExecutor:
    pass


class _DummyCtx:
    def _collect_static_registry_entries(self, query: str, max_entries: int = 20):
        if query != "dynamo":
            return []
        return [
            {
                "full_name": "omicverse.single.Velo.cal_velocity[method=dynamo]",
                "short_name": "dynamo",
                "signature": "cal_velocity(method)",
                "description": "Variant of velocity calculation when method='dynamo'.",
                "aliases": ["dynamo", "velo dynamo"],
                "examples": ["velo.cal_velocity(method='dynamo')"],
                "category": "trajectory",
                "branch_parameter": "method",
                "branch_value": "dynamo",
            }
        ]


def test_tool_search_functions_falls_back_to_static_registry(monkeypatch):
    runtime = ToolRuntime(_DummyCtx(), _DummyExecutor())
    fake_registry = SimpleNamespace(find=lambda query: [])

    monkeypatch.setattr(
        "omicverse.utils.ovagent.tool_runtime._global_registry",
        fake_registry,
    )

    result = runtime._tool_search_functions("dynamo")

    assert "omicverse.single.Velo.cal_velocity[method=dynamo]" in result
    assert "Branch: method='dynamo'" in result
