"""Factory functions and assertion helpers for integration tests.

Every function here creates lightweight data objects used by the harness fakes
and by test assertions.  No function in this module touches the network, disk,
or any production singleton.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .fakes import (
    AgentRunResult,
    ChatResponse,
    FakeExecutionAdapter,
    FakeLLM,
    FakePresenter,
    FakeRouter,
    FakeToolRuntime,
    ToolCall,
    Usage,
)

# Try to import Jarvis runtime models for factory helpers.
try:
    from omicverse.jarvis.runtime.models import (
        ConversationRoute,
        DeliveryEvent,
        MessageEnvelope,
        PolicyDecision,
        RuntimeTaskState,
    )

    _HAS_JARVIS = True
except Exception:  # pragma: no cover
    _HAS_JARVIS = False


# ===================================================================
#  Data-object factories
# ===================================================================


def make_tool_call(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    call_id: Optional[str] = None,
) -> ToolCall:
    """Create a ``ToolCall`` with sensible defaults."""
    return ToolCall(
        id=call_id or f"call_{uuid.uuid4().hex[:12]}",
        name=name,
        arguments=arguments or {},
    )


def make_usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    *,
    model: str = "fake-model",
    provider: str = "fake",
) -> Usage:
    """Create a ``Usage`` with sensible defaults."""
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        model=model,
        provider=provider,
    )


def make_chat_response(
    content: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None,
    stop_reason: str = "end_turn",
    *,
    usage: Optional[Usage] = None,
) -> ChatResponse:
    """Create a ``ChatResponse`` with sensible defaults."""
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=usage or make_usage(),
    )


def make_agent_run_result(
    summary: str = "ok",
    *,
    error: Optional[str] = None,
    figures: Optional[List[bytes]] = None,
    reports: Optional[List[str]] = None,
) -> AgentRunResult:
    """Create an ``AgentRunResult`` with sensible defaults."""
    return AgentRunResult(
        summary=summary,
        error=error,
        figures=figures or [],
        reports=reports or [],
    )


def make_route(
    channel: str = "test",
    scope_type: str = "dm",
    scope_id: str = "user-1",
    *,
    thread_id: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> Any:
    """Create a ``ConversationRoute``.

    Returns ``None`` if the Jarvis runtime models are not importable.
    """
    if not _HAS_JARVIS:
        return None
    return ConversationRoute(
        channel=channel,
        scope_type=scope_type,
        scope_id=scope_id,
        thread_id=thread_id,
        sender_id=sender_id,
    )


def make_envelope(
    text: str = "hello",
    *,
    channel: str = "test",
    scope_type: str = "dm",
    scope_id: str = "user-1",
    sender_id: str = "sender-1",
    trigger: str = "message",
    explicit_trigger: bool = False,
    route: Any = None,
) -> Any:
    """Create a ``MessageEnvelope``.

    Returns ``None`` if the Jarvis runtime models are not importable.
    """
    if not _HAS_JARVIS:
        return None
    r = route or make_route(
        channel=channel,
        scope_type=scope_type,
        scope_id=scope_id,
        sender_id=sender_id,
    )
    return MessageEnvelope(
        route=r,
        text=text,
        sender_id=sender_id,
        trigger=trigger,
        explicit_trigger=explicit_trigger,
    )


# ===================================================================
#  Harness wiring helpers
# ===================================================================


def build_fake_llm(
    responses: Optional[List[Any]] = None,
    *,
    model: str = "fake-model",
    provider: str = "fake",
) -> FakeLLM:
    """Convenience wrapper — same as ``FakeLLM(...)``."""
    return FakeLLM(responses, model=model, provider=provider)


def build_fake_tool_runtime(
    handlers: Optional[Dict[str, Any]] = None,
    *,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> FakeToolRuntime:
    """Convenience wrapper — same as ``FakeToolRuntime(...)``."""
    return FakeToolRuntime(handlers, tool_schemas=tool_schemas)


def build_jarvis_runtime_deps(
    *,
    execution_result: Optional[AgentRunResult] = None,
    default_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a dict of fakes suitable for constructing a ``MessageRuntime``.

    Keys: ``presenter``, ``execution_adapter``, ``router``, ``delivered``.
    ``delivered`` is a list that accumulates every ``DeliveryEvent`` sent
    through the delivery callback.
    """
    delivered: List[Any] = []

    async def _deliver(event: Any) -> None:
        delivered.append(event)

    return {
        "presenter": FakePresenter(),
        "execution_adapter": FakeExecutionAdapter(result=execution_result),
        "router": FakeRouter(default_session=default_session),
        "deliver": _deliver,
        "delivered": delivered,
    }
