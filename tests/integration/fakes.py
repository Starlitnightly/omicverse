"""Reusable fake implementations for integration-grade tests.

All classes in this module are test doubles that satisfy the same interfaces as
production components but operate entirely in-memory without network or disk I/O.

These fakes are intentionally kept in a single module so that any integration
test suite can ``from tests.integration.fakes import FakeLLM, ...`` without
pulling in production dependencies.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lightweight stand-ins for agent_backend_common types.
# We re-declare minimal versions here so the test harness stays importable
# even when the heavy production package cannot be loaded in CI.
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 20
    total_tokens: int = 30
    model: str = "fake-model"
    provider: str = "fake"


@dataclass
class _ToolCall:
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"call_{uuid.uuid4().hex[:12]}"


@dataclass
class _ChatResponse:
    content: Optional[str] = None
    tool_calls: List[_ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Optional[_Usage] = None
    raw_message: Optional[Any] = None


# Try to import the real types so that isinstance checks in production code
# work when the harness is used alongside the real package.  Fall back to the
# local stand-ins when imports fail (e.g. minimal CI without full deps).
try:
    from omicverse.utils.agent_backend_common import (
        BackendConfig,
        ChatResponse,
        ToolCall,
        Usage,
    )
except Exception:  # pragma: no cover – CI-only fallback
    Usage = _Usage  # type: ignore[misc,assignment]
    ToolCall = _ToolCall  # type: ignore[misc,assignment]
    ChatResponse = _ChatResponse  # type: ignore[misc,assignment]
    BackendConfig = None  # type: ignore[misc,assignment]

try:
    from omicverse.jarvis.agent_bridge import AgentRunResult
except Exception:  # pragma: no cover
    @dataclass
    class AgentRunResult:  # type: ignore[no-redef]
        adata: Optional[Any] = None
        figures: List[bytes] = field(default_factory=list)
        reports: List[str] = field(default_factory=list)
        artifacts: list = field(default_factory=list)
        summary: str = ""
        error: Optional[str] = None
        usage: Optional[Any] = None
        diagnostics: List[str] = field(default_factory=list)

try:
    from omicverse.jarvis.runtime.models import (
        ConversationRoute,
        DeliveryEvent,
        MessageEnvelope,
        PolicyDecision,
        RuntimeTaskState,
    )
except Exception:  # pragma: no cover
    ConversationRoute = None  # type: ignore[misc,assignment]
    DeliveryEvent = None  # type: ignore[misc,assignment]
    MessageEnvelope = None  # type: ignore[misc,assignment]
    PolicyDecision = None  # type: ignore[misc,assignment]
    RuntimeTaskState = None  # type: ignore[misc,assignment]


# ===================================================================
#  Fake LLM Backend
# ===================================================================


class FakeLLM:
    """Drop-in replacement for ``OmicVerseLLMBackend`` in tests.

    Parameters
    ----------
    responses : list
        Ordered list of responses.  Each element is either:
        - a ``str`` (returned verbatim from ``run`` / as ``content`` in ``chat``)
        - a ``ChatResponse`` (returned from ``chat`` as-is)
        Responses are consumed FIFO.  When the list is exhausted a default
        ``ChatResponse(content="<exhausted>", ...)`` is returned.
    model : str
        Model name exposed on ``config.model``.
    provider : str
        Provider name exposed on ``config.provider``.
    """

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        *,
        model: str = "fake-model",
        provider: str = "fake",
    ) -> None:
        self._responses: List[Any] = list(responses or [])
        self.config = SimpleNamespace(
            model=model,
            provider=provider,
            system_prompt="You are a test agent.",
            api_key=None,
            endpoint=None,
            max_tokens=1024,
            temperature=0.0,
        )
        self.last_usage: Optional[Any] = None

        # Call recording — tests can inspect these after exercising the agent.
        self.chat_calls: List[Dict[str, Any]] = []
        self.run_calls: List[str] = []
        self.stream_calls: List[str] = []

    # -- response helpers ------------------------------------------------

    def _pop_response(self) -> Any:
        if self._responses:
            return self._responses.pop(0)
        return ChatResponse(
            content="<exhausted>",
            tool_calls=[],
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=self.config.model,
                provider=self.config.provider,
            ),
        )

    def _to_chat_response(self, raw: Any) -> ChatResponse:
        if isinstance(raw, str):
            usage = Usage(
                input_tokens=10,
                output_tokens=len(raw),
                total_tokens=10 + len(raw),
                model=self.config.model,
                provider=self.config.provider,
            )
            return ChatResponse(
                content=raw,
                tool_calls=[],
                stop_reason="end_turn",
                usage=usage,
            )
        return raw  # already a ChatResponse

    # -- public interface (matches OmicVerseLLMBackend) -------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResponse:
        self.chat_calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        resp = self._to_chat_response(self._pop_response())
        self.last_usage = resp.usage
        return resp

    async def run(self, user_prompt: str) -> str:
        self.run_calls.append(user_prompt)
        resp = self._to_chat_response(self._pop_response())
        self.last_usage = resp.usage
        return resp.content or ""

    async def stream(self, user_prompt: str) -> AsyncIterator[str]:
        self.stream_calls.append(user_prompt)
        resp = self._to_chat_response(self._pop_response())
        self.last_usage = resp.usage
        text = resp.content or ""
        # Yield in small chunks to simulate streaming.
        chunk_size = max(1, len(text) // 3) if text else 1
        for i in range(0, max(len(text), 1), chunk_size):
            yield text[i : i + chunk_size]

    def format_tool_result_message(
        self, tool_call_id: str, tool_name: str, result: str
    ) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }

    @property
    def remaining_responses(self) -> int:
        return len(self._responses)


# ===================================================================
#  Fake Tool Registry & Runtime
# ===================================================================


class FakeToolRegistry:
    """Minimal stand-in for ``ToolRegistry``."""

    def __init__(self, tools: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._tools: Dict[str, Dict[str, Any]] = dict(tools or {})

    def resolve_name(self, name_or_alias: str) -> str:
        return name_or_alias

    def get(self, name_or_alias: str) -> Optional[SimpleNamespace]:
        spec = self._tools.get(name_or_alias)
        if spec is None:
            return None
        return SimpleNamespace(canonical_name=name_or_alias, **spec)

    def get_handler(self, name_or_alias: str) -> Optional[Callable[..., Any]]:
        meta = self.get(name_or_alias)
        if meta is None:
            return None
        return getattr(meta, "handler", None)

    def validate_handlers(self) -> List[str]:
        return []


class FakeToolRuntime:
    """Drop-in replacement for ``ToolRuntime`` in tests.

    Parameters
    ----------
    handlers : dict
        Mapping of tool name -> callable.  The callable receives
        ``(tool_args: dict, adata: Any, request: str)`` and returns a ``str``.
    tool_schemas : list
        Optional list of tool-schema dicts to return from
        ``get_visible_agent_tools``.
    """

    def __init__(
        self,
        handlers: Optional[Dict[str, Callable[..., str]]] = None,
        *,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._handlers: Dict[str, Callable[..., str]] = dict(handlers or {})
        self._tool_schemas = list(tool_schemas or [])
        self.registry = FakeToolRegistry(
            {name: {"handler": fn} for name, fn in self._handlers.items()}
        )
        self.dispatch_calls: List[Tuple[str, Dict[str, Any]]] = []

    async def dispatch(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        adata: Any,
        request: str,
    ) -> str:
        self.dispatch_calls.append((tool_name, tool_args))
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"[FakeToolRuntime] unknown tool: {tool_name}"
        result = handler(tool_args, adata, request)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def get_visible_agent_tools(
        self, *, allowed_names: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        if allowed_names is not None:
            return [s for s in self._tool_schemas if s.get("name") in allowed_names]
        return list(self._tool_schemas)

    def set_subagent_controller(self, controller: Any) -> None:
        pass


# ===================================================================
#  Jarvis fakes
# ===================================================================


class FakePresenter:
    """Records every presentation call for later assertion.

    Implements the ``MessagePresenter`` protocol used by ``MessageRuntime``.
    """

    def __init__(self) -> None:
        self.events: List[Tuple[str, Any]] = []

    def _record(self, method: str, **kwargs: Any) -> List[Any]:
        self.events.append((method, kwargs))
        return []

    def ack(self, envelope: Any, session: Any) -> list:
        return self._record("ack", envelope=envelope, session=session)

    def queue_started(self, route: Any, queued_count: int) -> list:
        return self._record("queue_started", route=route, queued_count=queued_count)

    def draft_open(self, route: Any) -> Any:
        self.events.append(("draft_open", {"route": route}))
        if DeliveryEvent is not None:
            return DeliveryEvent(route=route, kind="draft_open")
        return SimpleNamespace(route=route, kind="draft_open")

    def draft_update(self, route: Any, llm_text: str, progress: str) -> Any:
        self.events.append(
            ("draft_update", {"route": route, "llm_text": llm_text, "progress": progress})
        )
        if DeliveryEvent is not None:
            return DeliveryEvent(route=route, kind="draft_update", text=llm_text)
        return SimpleNamespace(route=route, kind="draft_update", text=llm_text)

    def draft_cancelled(self, route: Any) -> Any:
        self.events.append(("draft_cancelled", {"route": route}))
        if DeliveryEvent is not None:
            return DeliveryEvent(route=route, kind="draft_cancelled")
        return SimpleNamespace(route=route, kind="draft_cancelled")

    def analysis_error(self, route: Any, error_text: str) -> Any:
        self.events.append(
            ("analysis_error", {"route": route, "error_text": error_text})
        )
        if DeliveryEvent is not None:
            return DeliveryEvent(route=route, kind="analysis_error", text=error_text)
        return SimpleNamespace(route=route, kind="analysis_error", text=error_text)

    def typing(self, route: Any) -> Optional[Any]:
        self.events.append(("typing", {"route": route}))
        return None

    def quick_chat_reply(self, route: Any, text: str) -> Any:
        self.events.append(("quick_chat_reply", {"route": route, "text": text}))
        if DeliveryEvent is not None:
            return DeliveryEvent(route=route, kind="quick_chat_reply", text=text)
        return SimpleNamespace(route=route, kind="quick_chat_reply", text=text)

    def quick_chat_fallback(self, route: Any) -> Any:
        self.events.append(("quick_chat_fallback", {"route": route}))
        if DeliveryEvent is not None:
            return DeliveryEvent(route=route, kind="quick_chat_fallback")
        return SimpleNamespace(route=route, kind="quick_chat_fallback")

    def analysis_status(
        self,
        route: Any,
        *,
        has_media: bool = False,
        has_reports: bool = False,
        has_artifacts: bool = False,
    ) -> Optional[Any]:
        self.events.append(
            (
                "analysis_status",
                {
                    "route": route,
                    "has_media": has_media,
                    "has_reports": has_reports,
                    "has_artifacts": has_artifacts,
                },
            )
        )
        return None

    def final_events(
        self,
        route: Any,
        *,
        session: Any,
        user_text: str,
        llm_text: str,
        result: Any,
    ) -> list:
        self.events.append(
            (
                "final_events",
                {
                    "route": route,
                    "session": session,
                    "user_text": user_text,
                    "llm_text": llm_text,
                    "result": result,
                },
            )
        )
        return []

    def method_calls(self, method_name: str) -> List[Dict[str, Any]]:
        """Return kwargs dicts for every call to *method_name*."""
        return [kw for name, kw in self.events if name == method_name]


class FakeExecutionAdapter:
    """Drop-in replacement for ``ExecutionAdapter`` in tests.

    Parameters
    ----------
    result : AgentRunResult | None
        The result to return from ``run``.  Defaults to a blank
        ``AgentRunResult(summary="ok")``.
    """

    def __init__(self, result: Optional[AgentRunResult] = None) -> None:
        self._result = result or AgentRunResult(summary="ok")
        self.run_calls: List[Dict[str, Any]] = []

    async def run(
        self,
        session: Any,
        request: str,
        *,
        adata: Optional[Any] = None,
        callbacks: Any = None,
        history: Optional[list] = None,
        request_content: Optional[list] = None,
    ) -> AgentRunResult:
        self.run_calls.append(
            {
                "session": session,
                "request": request,
                "adata": adata,
                "callbacks": callbacks,
                "history": history,
                "request_content": request_content,
            }
        )
        return self._result


class FakeRouter:
    """Minimal stand-in for ``MessageRouter``.

    Returns the same ``session`` object for every route, or per-route
    overrides supplied at construction.
    """

    def __init__(
        self,
        default_session: Optional[Any] = None,
        sessions: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._default = default_session or SimpleNamespace(
            agent=None, adata=None, history=[]
        )
        self._sessions: Dict[str, Any] = dict(sessions or {})

    def get_session(self, route: Any) -> Any:
        key = getattr(route, "route_key", lambda: str(route))()
        return self._sessions.get(key, self._default)
