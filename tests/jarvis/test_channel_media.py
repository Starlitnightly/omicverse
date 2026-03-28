from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from omicverse.jarvis.agent_bridge import AgentRunResult
from omicverse.jarvis.channel_media import (
    MAX_INBOUND_IMAGES,
    build_channel_request,
    format_h5ad_load_result,
    inbound_upload_dir,
    is_image_attachment,
    load_h5ad_to_session,
    prepare_channel_delivery_figures,
    prepare_inbound_image,
    prepare_inbound_image_from_file,
)
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
    assert "show=False before saving" in request
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
    assert "show=False before saving" in adapter.requests[0]["request"]
    assert "newest PNG" in adapter.requests[0]["request"]


# ── MAX_INBOUND_IMAGES ─────────────────────────────────────────────────────

class TestMaxInboundImages:
    def test_constant_is_four(self) -> None:
        assert MAX_INBOUND_IMAGES == 4

    def test_constant_is_int(self) -> None:
        assert isinstance(MAX_INBOUND_IMAGES, int)


# ── is_image_attachment ────────────────────────────────────────────────────

class TestIsImageAttachment:
    def test_by_mime_type(self) -> None:
        assert is_image_attachment("", "image/png") is True
        assert is_image_attachment("", "image/jpeg") is True
        assert is_image_attachment("", "image/webp") is True

    def test_by_filename(self) -> None:
        assert is_image_attachment("photo.png") is True
        assert is_image_attachment("scan.jpg") is True
        assert is_image_attachment("diagram.webp") is True

    def test_non_image_rejected(self) -> None:
        assert is_image_attachment("data.csv") is False
        assert is_image_attachment("", "application/json") is False
        assert is_image_attachment("") is False

    def test_h5ad_rejected(self) -> None:
        assert is_image_attachment("sample.h5ad") is False

    def test_mime_takes_precedence(self) -> None:
        assert is_image_attachment("data.csv", "image/png") is True

    def test_case_insensitive_mime(self) -> None:
        assert is_image_attachment("", "Image/PNG") is True

    def test_whitespace_stripped(self) -> None:
        assert is_image_attachment("", "  image/png  ") is True


# ── inbound_upload_dir ─────────────────────────────────────────────────────

class TestInboundUploadDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        result = inbound_upload_dir(tmp_path, "discord")
        assert result == tmp_path / "uploads" / "discord"
        assert result.is_dir()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        result = inbound_upload_dir(str(tmp_path), "telegram")
        assert isinstance(result, Path)
        assert result.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        first = inbound_upload_dir(tmp_path, "qq")
        second = inbound_upload_dir(tmp_path, "qq")
        assert first == second


# ── prepare_inbound_image ──────────────────────────────────────────────────

class TestPrepareInboundImage:
    def _make_png(self) -> bytes:
        """Create minimal valid PNG bytes for testing."""
        try:
            from PIL import Image
            import io
            img = Image.new("RGB", (2, 2), color="red")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("PIL not available")

    def test_creates_file_in_upload_dir(self, tmp_path: Path) -> None:
        png = self._make_png()
        result = prepare_inbound_image(
            png,
            workspace_root=tmp_path,
            channel_name="discord",
            filename="photo.png",
            mime_type="image/png",
        )
        assert result.path.exists()
        assert result.path.parent == tmp_path / "uploads" / "discord"
        assert result.mime_type == "image/png"
        assert result.source == "discord"
        assert result.size == len(png)

    def test_uses_channel_name_as_prefix(self, tmp_path: Path) -> None:
        png = self._make_png()
        result = prepare_inbound_image(
            png,
            workspace_root=tmp_path,
            channel_name="qq",
            filename="test.png",
        )
        assert "qq_image" in result.path.name

    def test_default_filename(self, tmp_path: Path) -> None:
        png = self._make_png()
        result = prepare_inbound_image(
            png,
            workspace_root=tmp_path,
            channel_name="wechat",
        )
        assert "wechat_image" in result.path.name

    def test_request_block_has_data_url(self, tmp_path: Path) -> None:
        png = self._make_png()
        result = prepare_inbound_image(
            png,
            workspace_root=tmp_path,
            channel_name="discord",
            filename="img.png",
            mime_type="image/png",
        )
        assert result.request_block["type"] == "input_image"
        assert result.request_block["image_url"].startswith("data:image/png;base64,")


# ── prepare_inbound_image_from_file ────────────────────────────────────────

class TestPrepareInboundImageFromFile:
    def _write_png(self, path: Path) -> Path:
        try:
            from PIL import Image
            import io
            img = Image.new("RGB", (2, 2), color="blue")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.getvalue())
            return path
        except ImportError:
            pytest.skip("PIL not available")

    def test_processes_local_file(self, tmp_path: Path) -> None:
        src = self._write_png(tmp_path / "raw" / "photo.png")
        result = prepare_inbound_image_from_file(
            src,
            workspace_root=tmp_path,
            channel_name="telegram",
        )
        assert result.path.exists()
        assert result.mime_type == "image/png"
        assert result.source == "telegram"

    def test_uses_channel_prefix(self, tmp_path: Path) -> None:
        src = self._write_png(tmp_path / "raw" / "image.png")
        result = prepare_inbound_image_from_file(
            src,
            workspace_root=tmp_path,
            channel_name="telegram",
        )
        assert "telegram_image" in result.path.name

    def test_passes_mime_type(self, tmp_path: Path) -> None:
        src = self._write_png(tmp_path / "raw" / "doc.png")
        result = prepare_inbound_image_from_file(
            src,
            workspace_root=tmp_path,
            channel_name="telegram",
            mime_type="image/png",
        )
        assert result.mime_type == "image/png"


# ── load_h5ad_to_session ──────────────────────────────────────────────────

class TestLoadH5adToSession:
    def test_writes_data_and_calls_load(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        load_calls: list = []

        session = SimpleNamespace(
            workspace=workspace,
            load_from_workspace=lambda fname: (load_calls.append(fname), "mock_adata")[-1],
        )
        result = load_h5ad_to_session(session, b"fake-h5ad-data", "test.h5ad")
        assert result == "mock_adata"
        assert load_calls == ["test.h5ad"]
        assert (workspace / "test.h5ad").read_bytes() == b"fake-h5ad-data"

    def test_returns_none_when_load_fails(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        session = SimpleNamespace(
            workspace=workspace,
            load_from_workspace=lambda fname: None,
        )
        result = load_h5ad_to_session(session, b"data", "bad.h5ad")
        assert result is None

    def test_returns_none_when_no_workspace(self) -> None:
        session = SimpleNamespace()
        result = load_h5ad_to_session(session, b"data", "test.h5ad")
        assert result is None

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        workspace = tmp_path / "deep" / "workspace"
        session = SimpleNamespace(
            workspace=workspace,
            load_from_workspace=lambda fname: "ok",
        )
        result = load_h5ad_to_session(session, b"data", "test.h5ad")
        assert result == "ok"
        assert (workspace / "test.h5ad").exists()


# ── format_h5ad_load_result ───────────────────────────────────────────────

class TestFormatH5adLoadResult:
    def test_success_format(self) -> None:
        adata = SimpleNamespace(n_obs=5000, n_vars=2000)
        text = format_h5ad_load_result(adata, "sample.h5ad")
        assert "加载成功" in text
        assert "5,000" in text
        assert "2,000" in text
        assert "sample.h5ad" in text

    def test_failure_format(self) -> None:
        text = format_h5ad_load_result(None, "bad.h5ad")
        assert "bad.h5ad" in text
        assert "加载失败" in text

    def test_matches_discord_format(self) -> None:
        adata = SimpleNamespace(n_obs=100, n_vars=50)
        text = format_h5ad_load_result(adata, "data.h5ad")
        assert "✅" in text
        assert "🔬" in text
        assert "📁" in text


# ── Channel imports use shared media helpers ──────────────────────────────

class TestChannelMediaImports:
    """Verify that channel modules import and use the shared helpers."""

    def test_discord_uses_shared_helpers(self) -> None:
        from omicverse.jarvis.channels import discord as mod
        from omicverse.jarvis import channel_media
        assert hasattr(mod, "is_image_attachment")
        assert mod.is_image_attachment is channel_media.is_image_attachment
        assert mod.MAX_INBOUND_IMAGES is channel_media.MAX_INBOUND_IMAGES
        assert mod.prepare_inbound_image is channel_media.prepare_inbound_image

    def test_qq_uses_shared_helpers(self) -> None:
        from omicverse.jarvis.channels import qq as mod
        from omicverse.jarvis import channel_media
        assert hasattr(mod, "is_image_attachment")
        assert mod.is_image_attachment is channel_media.is_image_attachment
        assert mod.MAX_INBOUND_IMAGES is channel_media.MAX_INBOUND_IMAGES

    def test_telegram_uses_shared_helpers(self) -> None:
        from omicverse.jarvis.channels import telegram as mod
        from omicverse.jarvis import channel_media
        assert mod.MAX_INBOUND_IMAGES is channel_media.MAX_INBOUND_IMAGES
        assert mod.prepare_inbound_image_from_file is channel_media.prepare_inbound_image_from_file
