from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional, Protocol

from ..agent_bridge import AgentRunResult
from .._bridge_session import resolve_bridge_session_id
from .execution_adapter import ExecutionAdapter, ExecutionCallbacks
from .models import (
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    PolicyDecision,
    RuntimeTaskState,
)
from .policy import MessagePolicy
from .router import MessageRouter
from .task_registry import TaskRegistry

logger = logging.getLogger("omicverse.jarvis.runtime")

DeliveryFn = Callable[[DeliveryEvent], Awaitable[None]]


class MessagePresenter(Protocol):
    def ack(self, envelope: MessageEnvelope, session: Any) -> List[DeliveryEvent]:
        ...

    def queue_started(self, route: ConversationRoute, queued_count: int) -> List[DeliveryEvent]:
        ...

    def draft_open(self, route: ConversationRoute) -> DeliveryEvent:
        ...

    def draft_update(self, route: ConversationRoute, llm_text: str, progress: str) -> DeliveryEvent:
        ...

    def draft_cancelled(self, route: ConversationRoute) -> DeliveryEvent:
        ...

    def analysis_error(self, route: ConversationRoute, error_text: str) -> DeliveryEvent:
        ...

    def typing(self, route: ConversationRoute) -> Optional[DeliveryEvent]:
        ...

    def quick_chat_reply(self, route: ConversationRoute, text: str) -> DeliveryEvent:
        ...

    def quick_chat_fallback(self, route: ConversationRoute) -> DeliveryEvent:
        ...

    def analysis_status(
        self,
        route: ConversationRoute,
        *,
        has_media: bool,
        has_reports: bool,
        has_artifacts: bool,
    ) -> Optional[DeliveryEvent]:
        ...

    def final_events(
        self,
        route: ConversationRoute,
        *,
        session: Any,
        user_text: str,
        llm_text: str,
        result: AgentRunResult,
    ) -> List[DeliveryEvent]:
        ...


class MessageRuntime:
    def __init__(
        self,
        *,
        router: MessageRouter,
        presenter: MessagePresenter,
        execution_adapter: ExecutionAdapter,
        deliver: DeliveryFn,
        task_registry: Optional[TaskRegistry] = None,
        policy: Optional[MessagePolicy] = None,
        web_bridge: Optional[Any] = None,
    ) -> None:
        self._router = router
        self._presenter = presenter
        self._execution = execution_adapter
        self._deliver = deliver
        self._tasks = task_registry or TaskRegistry()
        self._policy = policy or MessagePolicy()
        self._web_bridge = web_bridge  # Optional WebSessionBridge

    @property
    def task_registry(self) -> TaskRegistry:
        return self._tasks

    def task_state(self, route: ConversationRoute) -> RuntimeTaskState:
        return self._tasks.snapshot(route)

    def running_task(self, route: ConversationRoute) -> Optional[asyncio.Task]:
        return self._tasks.task_for(route)

    def get_session(self, route: ConversationRoute) -> Any:
        return self._router.get_session(route)

    async def cancel(self, route: ConversationRoute) -> bool:
        return await self._tasks.cancel(route)

    async def handle_message(self, envelope: MessageEnvelope) -> PolicyDecision:
        state = self._tasks.snapshot(envelope.route)
        decision = self._policy.decide(envelope, state)

        if decision.should_ignore:
            return decision

        session = self._router.get_session(envelope.route)

        if decision.action == "quick_chat":
            asyncio.create_task(
                self._quick_chat(
                    session=session,
                    envelope=envelope,
                    running_request=state.request,
                    queued=False,
                )
            )
            return decision

        if decision.should_queue:
            self._tasks.enqueue(envelope)
            if decision.should_quick_chat:
                asyncio.create_task(
                    self._quick_chat(
                        session=session,
                        envelope=envelope,
                        running_request=state.request,
                        queued=True,
                    )
                )
            return decision

        if decision.should_ack:
            for event in self._presenter.ack(envelope, session):
                await self._deliver(event)

        if decision.should_start or decision.action == "start":
            self._start_analysis(session=session, envelope=envelope)
        return decision

    def _start_analysis(
        self,
        *,
        session: Any,
        envelope: MessageEnvelope,
        send_ack: bool = False,
    ) -> None:
        if send_ack:
            raise ValueError("Ack delivery is handled before _start_analysis")

        route = envelope.route
        full_request = self.build_full_request(session, envelope.text)
        request_content = list((envelope.metadata or {}).get("request_content") or [])

        async def _runner() -> None:
            try:
                await self._run_analysis(
                    session=session,
                    route=route,
                    user_text=envelope.text,
                    full_request=full_request,
                    request_content=request_content,
                )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.exception("Analysis task failed for %s", route.route_key())
                await self._deliver(self._presenter.analysis_error(route, str(exc)))
            finally:
                self._tasks.finish(route)
                queued = self._tasks.pop_pending(route)
                if queued:
                    for event in self._presenter.queue_started(route, len(queued)):
                        await self._deliver(event)
                    self._start_analysis(
                        session=session,
                        envelope=self._coalesce_envelopes(route, queued),
                    )

        task = asyncio.create_task(_runner())
        self._tasks.start(route, envelope=envelope, task=task)

    async def _run_analysis(
        self,
        *,
        session: Any,
        route: ConversationRoute,
        user_text: str,
        full_request: str,
        request_content: Optional[List[dict]] = None,
    ) -> None:
        llm_buf = ""
        last_progress = ""
        last_edit = 0.0
        edit_gap = 1.5

        await self._deliver(self._presenter.draft_open(route))

        async def _update_draft(force: bool = False) -> None:
            nonlocal last_edit
            now = time.monotonic()
            if (not force) and (now - last_edit < edit_gap):
                return
            last_edit = now
            await self._deliver(self._presenter.draft_update(route, llm_buf, last_progress))

        async def llm_chunk_cb(chunk: str) -> None:
            nonlocal llm_buf
            if not chunk:
                return
            llm_buf += chunk
            await _update_draft(force=False)

        async def progress_cb(message: str) -> None:
            nonlocal last_progress
            last_progress = message
            await _update_draft(force=True)

        # Load workspace history so the channel agent has multi-turn context.
        # Falls back to [] if the bridge doesn't implement get_prior_history
        # (e.g. plain WebSessionBridge) or if no workspace exists yet.
        prior_history: List[dict] = []
        if self._web_bridge is not None:
            try:
                prior_history = self._web_bridge.get_prior_history(
                    route,
                    session_id=resolve_bridge_session_id(session),
                ) or []
            except Exception:
                pass

        try:
            result = await self._execution.run(
                session,
                full_request,
                adata=session.adata,
                callbacks=ExecutionCallbacks(
                    progress_cb=progress_cb,
                    llm_chunk_cb=llm_chunk_cb,
                ),
                history=prior_history,
                request_content=request_content or [],
            )
        except asyncio.CancelledError:
            await self._deliver(self._presenter.draft_cancelled(route))
            raise

        self._persist_result(session=session, user_text=user_text, result=result)

        # When no LLM text was streamed (e.g. agent called finish() directly
        # without a preceding text-only turn), fall back to result.summary so
        # that all channel presenters and the web bridge receive meaningful text.
        effective_llm_text = llm_buf if llm_buf.strip() else (result.summary or "")

        # Mirror the completed turn into the web session (gateway mode)
        if self._web_bridge is not None:
            try:
                self._web_bridge.on_turn_complete(
                    route=route,
                    user_text=user_text,
                    llm_text=effective_llm_text,
                    adata=result.adata,
                    figures=result.figures or [],
                    session_id=resolve_bridge_session_id(session),
                )
            except Exception:
                logger.warning("web_bridge.on_turn_complete failed (non-fatal)", exc_info=True)

        if result.error:
            await self._deliver(self._presenter.analysis_error(route, result.error))
            return

        has_media = bool(result.figures)
        has_reports = bool(result.reports)
        has_artifacts = bool(result.artifacts)
        if has_media or has_reports or has_artifacts:
            event = self._presenter.analysis_status(
                route,
                has_media=has_media,
                has_reports=has_reports,
                has_artifacts=has_artifacts,
            )
            if event is not None:
                await self._deliver(event)

        for event in self._presenter.final_events(
            route,
            session=session,
            user_text=user_text,
            llm_text=effective_llm_text,
            result=result,
        ):
            await self._deliver(event)

    async def _quick_chat(
        self,
        *,
        session: Any,
        envelope: MessageEnvelope,
        running_request: str,
        queued: bool,
    ) -> None:
        typing_event = self._presenter.typing(envelope.route)
        if typing_event is not None:
            await self._deliver(typing_event)

        try:
            system_lines = [
                "You are OmicVerse Jarvis, a bioinformatics AI assistant.",
                "Answer concisely and helpfully. Do NOT execute code or call tools.",
                "Reply in the same language the user uses.",
            ]
            if running_request:
                system_lines.append("A background analysis is currently running.")
                system_lines.append(f"Currently running analysis: {running_request[:300]}")
            if queued:
                system_lines.append(
                    "If the user's message looks like a new analysis request, "
                    "inform them it has been queued and will start automatically after the current analysis finishes."
                )
            if session.adata is not None:
                a = session.adata
                system_lines.append(f"Loaded data: {a.n_obs:,} cells × {a.n_vars:,} genes")
            memory_ctx = session.get_memory_context()
            if memory_ctx:
                system_lines.append(f"\nRecent analysis history:\n{memory_ctx[:600]}")

            messages = [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user", "content": envelope.text},
            ]
            response = await session.agent._llm.chat(messages, tools=None, tool_choice=None)
            reply = (getattr(response, "content", "") or "").strip()
            if not reply:
                reply = "💬  分析进行中，稍后再试。"
            await self._deliver(self._presenter.quick_chat_reply(envelope.route, reply))
        except Exception as exc:
            logger.warning("Quick chat failed: %s", exc)
            await self._deliver(self._presenter.quick_chat_fallback(envelope.route))

    @staticmethod
    def _coalesce_envelopes(
        route: ConversationRoute,
        envelopes: List[MessageEnvelope],
    ) -> MessageEnvelope:
        if not envelopes:
            raise ValueError("Expected at least one queued envelope")

        multi_sender = len({env.sender_id for env in envelopes if env.sender_id}) > 1
        parts: List[str] = []
        for env in envelopes:
            text = (env.text or "").strip()
            if not text:
                continue
            if route.is_direct or not multi_sender:
                parts.append(text)
                continue
            label = env.sender_username or env.sender_id or "user"
            parts.append(f"[{label}] {text}")

        merged = "\n\n".join(parts).strip()
        last = envelopes[-1]
        metadata = dict(last.metadata)
        metadata["queued_count"] = len(envelopes)
        request_content: List[dict] = []
        for env in envelopes:
            request_content.extend(list((env.metadata or {}).get("request_content") or []))
        if request_content:
            metadata["request_content"] = request_content
        return MessageEnvelope(
            route=route,
            text=merged,
            sender_id=last.sender_id,
            sender_username=last.sender_username,
            message_id=last.message_id,
            trigger=last.trigger,
            explicit_trigger=last.explicit_trigger,
            metadata=metadata,
        )

    @staticmethod
    def build_full_request(session: Any, text: str) -> str:
        ctx_parts: List[str] = []
        try:
            agents_md = session.get_agents_md()
            if agents_md:
                ctx_parts.append(f"[User instructions]\n{agents_md}")
        except Exception:
            pass
        try:
            memory_ctx = session.get_memory_context()
            if memory_ctx:
                ctx_parts.append(f"[Analysis history]\n{memory_ctx}")
        except Exception:
            pass
        if not ctx_parts:
            return text
        return "\n\n".join(ctx_parts) + f"\n\n[Current request]\n{text}"

    @staticmethod
    def _persist_result(*, session: Any, user_text: str, result: AgentRunResult) -> None:
        if result.adata is not None:
            session.adata = result.adata
            session.prompt_count += 1
            try:
                session.save_adata()
            except Exception:
                pass
        if result.usage is not None:
            session.last_usage = result.usage

        a_cur = result.adata or session.adata
        a_info = f"{a_cur.n_obs:,} cells × {a_cur.n_vars:,} genes" if a_cur else ""
        try:
            session.append_memory_log(
                request=user_text,
                summary=result.summary or "分析完成",
                adata_info=a_info,
            )
        except Exception:
            pass
