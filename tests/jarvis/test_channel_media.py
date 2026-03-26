from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from omicverse.jarvis.agent_bridge import AgentRunResult
from omicverse.jarvis.channel_media import build_channel_request, prepare_channel_delivery_figures
from omicverse.jarvis.runtime.models import ConversationRoute, DeliveryEvent, MessageEnvelope
from omicverse.jarvis.runtime.runtime import MessageRuntime


class _FakeSession:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.agent = object()
        self.adata = None
        self.prompt_count = 0
        self.last_usage = None
        self.memory_logs: List[dict[str, Any]] = []

    @property
    def workspace(self) -> Path:
        return self.workspace_dir / "workspace"

    def get_agents_md(self) -> str:
        return "Always explain figures briefly."

    def get_memory_context(self) -> str:
        return "The user prefers visual summaries."

    def save_adata(self) -> Path:
        path = self.workspace_dir / "current.h5ad"
        path.write_text("stub")
        return path

    def append_memory_log(self, request: str, summary: str, adata_info: str = "") -> None:
        self.memory_logs.append(
            {"request": request, "summary": summary, "adata_info": adata_info}
        )


class _FakeRouter:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def get_session(self, _route: ConversationRoute) -> _FakeSession:
        return self._session


class _FakePresenter:
    def ack(self, envelope: MessageEnvelope, session: Any) -> List[DeliveryEvent]:
        return []

    def queue_started(self, route: ConversationRoute, queued_count: int) -> List[DeliveryEvent]:
        return []

    def draft_open(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="draft_open")

    def draft_update(self, route: ConversationRoute, llm_text: str, progress: str) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="draft_update", text=f"{progress}|{llm_text}")

    def draft_cancelled(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="draft_cancelled")

    def analysis_error(self, route: ConversationRoute, error_text: str) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="error", text=error_text)

    def typing(self, route: ConversationRoute) -> DeliveryEvent | None:
        return None

    def quick_chat_reply(self, route: ConversationRoute, text: str) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="quick_chat", text=text)

    def quick_chat_fallback(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(route=route, kind="quick_chat_fallback")

    def analysis_status(
        self,
        route: ConversationRoute,
        *,
        has_media: bool,
        has_reports: bool,
        has_artifacts: bool,
    ) -> DeliveryEvent | None:
        if not (has_media or has_reports or has_artifacts):
            return None
        return DeliveryEvent(route=route, kind="status", text="sending")

    def final_events(
        self,
        route: ConversationRoute,
        *,
        session: Any,
        user_text: str,
        llm_text: str,
        result: AgentRunResult,
    ) -> List[DeliveryEvent]:
        events: List[DeliveryEvent] = []
        for index, figure in enumerate(result.figures, start=1):
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="photo",
                    binary=figure,
                    filename=f"figure_{index}.png",
                )
            )
        if result.summary:
            events.append(DeliveryEvent(route=route, kind="text", text=result.summary))
        return events


class _FakeExecutionAdapter:
    def __init__(self) -> None:
        self.requests: List[dict[str, Any]] = []

    async def run(
        self,
        session: Any,
        request: str,
        *,
        adata: Any,
        callbacks: Any,
        history: list | None = None,
        request_content: list | None = None,
    ) -> AgentRunResult:
        self.requests.append(
            {
                "request": request,
                "history": list(history or []),
                "request_content": list(request_content or []),
            }
        )
        if callbacks.progress_cb is not None:
            await callbacks.progress_cb("绘图中")
        if callbacks.llm_chunk_cb is not None:
            await callbacks.llm_chunk_cb("图已生成")
        return AgentRunResult(
            figures=[b"stale-figure", b"latest-figure"],
            summary="分析完成",
        )


def test_prepare_channel_delivery_figures_saves_workspace_png_and_keeps_latest(tmp_path: Path) -> None:
    session = _FakeSession(tmp_path / "session")

    request = build_channel_request(session, "请画一个UMAP图", channel_label="Feishu")
    selected = prepare_channel_delivery_figures(session, [b"first-figure", b"latest-figure"])

    outputs = sorted((session.workspace / "outputs").glob("analysis_figure_*.png"))
    assert len(outputs) == 2
    assert outputs[0].read_bytes() == b"first-figure"
    assert outputs[1].read_bytes() == b"latest-figure"
    assert selected == [b"latest-figure"]
    assert "[Channel: Feishu" in request
    assert "figures/<descriptive_name>.png" in request
    assert "newest PNG" in request
    assert "[Current request]\n请画一个UMAP图" in request


@pytest.mark.asyncio
async def test_message_runtime_routes_only_latest_workspace_figure(tmp_path: Path) -> None:
    session = _FakeSession(tmp_path / "runtime-session")
    adapter = _FakeExecutionAdapter()
    delivered: List[DeliveryEvent] = []

    async def _deliver(event: DeliveryEvent) -> None:
        delivered.append(event)

    runtime = MessageRuntime(
        router=_FakeRouter(session),
        presenter=_FakePresenter(),
        execution_adapter=adapter,
        deliver=_deliver,
    )

    route = ConversationRoute(channel="telegram", scope_type="dm", scope_id="user-1")
    envelope = MessageEnvelope(
        route=route,
        text="请生成分析图并发给我",
        sender_id="user-1",
    )

    decision = await runtime.handle_message(envelope)
    assert decision.should_start is True

    task = runtime.running_task(route)
    if task is not None:
        await task

    outputs = sorted((session.workspace / "outputs").glob("analysis_figure_*.png"))
    photos = [event for event in delivered if event.kind == "photo"]
    assert len(outputs) == 2
    assert len(photos) == 1
    assert photos[0].binary == b"latest-figure"
    assert any(event.kind == "status" for event in delivered)
    assert "[Channel:" in adapter.requests[0]["request"]
    assert "figures/<descriptive_name>.png" in adapter.requests[0]["request"]
    assert "newest PNG" in adapter.requests[0]["request"]
