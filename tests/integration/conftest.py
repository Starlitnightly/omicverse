"""Pytest fixtures for the integration test harness.

Fixtures in this file are automatically available to every test inside
``tests/integration/``.  Other test directories can import the helper
factories directly from ``tests.integration.helpers`` when needed.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import pytest

from .fakes import (
    AgentRunResult,
    ChatResponse,
    FakeExecutionAdapter,
    FakeLLM,
    FakePresenter,
    FakeRouter,
    FakeToolRuntime,
)
from .helpers import (
    build_fake_llm,
    build_fake_tool_runtime,
    build_jarvis_runtime_deps,
    make_chat_response,
    make_envelope,
    make_route,
    make_tool_call,
)


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}


# -----------------------------------------------------------------
# LLM fixtures
# -----------------------------------------------------------------


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLM]:
    """Factory fixture: call with a list of responses to get a ``FakeLLM``."""
    return build_fake_llm


@pytest.fixture
def simple_fake_llm() -> FakeLLM:
    """Pre-built ``FakeLLM`` with a single text response."""
    return build_fake_llm(["Hello from the fake LLM."])


# -----------------------------------------------------------------
# Tool-runtime fixtures
# -----------------------------------------------------------------


@pytest.fixture
def fake_tool_runtime() -> Callable[..., FakeToolRuntime]:
    """Factory fixture: call with handlers dict to get a ``FakeToolRuntime``."""
    return build_fake_tool_runtime


@pytest.fixture
def echo_tool_runtime() -> FakeToolRuntime:
    """Pre-built ``FakeToolRuntime`` with a single *echo* tool."""

    def _echo(args: Dict[str, Any], adata: Any, request: str) -> str:
        return f"echo: {args}"

    return build_fake_tool_runtime(
        {"echo": _echo},
        tool_schemas=[
            {
                "name": "echo",
                "description": "Echoes its arguments.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
    )


# -----------------------------------------------------------------
# Jarvis fixtures
# -----------------------------------------------------------------


@pytest.fixture
def fake_presenter() -> FakePresenter:
    """Fresh ``FakePresenter`` for recording Jarvis presentation calls."""
    return FakePresenter()


@pytest.fixture
def fake_execution_adapter() -> Callable[..., FakeExecutionAdapter]:
    """Factory fixture: call with an optional ``AgentRunResult``."""

    def _build(result: Optional[AgentRunResult] = None) -> FakeExecutionAdapter:
        return FakeExecutionAdapter(result=result)

    return _build


@pytest.fixture
def fake_route():
    """Default ``ConversationRoute`` for test channel *test*, DM scope."""
    return make_route()


@pytest.fixture
def fake_envelope():
    """Default ``MessageEnvelope`` with text ``'hello'``."""
    return make_envelope()


@pytest.fixture
def jarvis_runtime_deps():
    """Dict of fakes ready to construct a ``MessageRuntime``.

    Keys: ``presenter``, ``execution_adapter``, ``router``, ``deliver``,
    ``delivered`` (list accumulating delivery events).
    """
    return build_jarvis_runtime_deps()
