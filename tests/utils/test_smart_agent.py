import ast
import asyncio
import importlib.machinery
import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

OmicVerseAgent = smart_agent_module.OmicVerseAgent


def _build_agent(return_value):
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    async def _fake_run_async(self, request, adata):
        return {
            "request": request,
            "adata": adata,
            "value": return_value,
        }

    agent.run_async = MethodType(_fake_run_async, agent)
    return agent


def test_extract_python_code_includes_function_defs():
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    response_text = """
LLM response:
import numpy as np
def summarize_counts(adata):
    counts = []
    for value in np.sum(adata.X, axis=1):
        counts.append(float(value))
    return counts
for _ in range(1):
    totals = summarize_counts(adata)
    print(min(totals))
"""

    code = agent._extract_python_code(response_text)

    assert "def summarize_counts" in code
    assert "counts.append" in code
    ast.parse(code)


def test_run_outside_event_loop(monkeypatch):
    agent = _build_agent(return_value="plain")
    original_asyncio_run = asyncio.run
    run_threads = []

    def _tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_asyncio_run(coro)

    monkeypatch.setattr(asyncio, "run", _tracking_run)

    sentinel = object()
    result = agent.run("plain request", sentinel)

    assert result["request"] == "plain request"
    assert result["adata"] is sentinel
    assert result["value"] == "plain"
    assert run_threads == [threading.current_thread().name]


def test_run_inside_running_loop(monkeypatch):
    agent = _build_agent(return_value="nested")
    original_asyncio_run = asyncio.run
    run_threads = []

    def _tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_asyncio_run(coro)

    async def _caller():
        monkeypatch.setattr(asyncio, "run", _tracking_run)
        sentinel = object()
        result = agent.run("loop request", sentinel)

        assert result["request"] == "loop request"
        assert result["adata"] is sentinel
        assert result["value"] == "nested"
        assert run_threads and run_threads[0] != threading.current_thread().name

    asyncio.run(_caller())
