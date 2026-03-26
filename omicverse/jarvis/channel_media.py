from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, List, Optional


def build_channel_request(session: Any, text: str, *, channel_label: str = "Mobile channel") -> str:
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
    ctx_parts.append(channel_image_hint(channel_label))
    ctx_parts.append(f"[Current request]\n{text}")
    return "\n\n".join(part for part in ctx_parts if part)


def channel_image_hint(channel_label: str = "Mobile channel") -> str:
    label = (channel_label or "Mobile channel").strip()
    return (
        f"[Channel: {label} — image delivery rules]\n"
        "1. If the user asks for a plot, figure, visualization, or any image output, "
        "you MUST execute Python code that creates the figure with matplotlib/scanpy.\n"
        "2. If you save figures explicitly, save the final figure to "
        "figures/<descriptive_name>.png with plt.savefig(..., dpi=200, bbox_inches='tight'). "
        "If you create multiple figures, save the one that should be sent to the user LAST. "
        "For ov.pl.*, sc.pl.*, scv.pl.*, or similar plotting helpers, set show=False before saving.\n"
        "3. After code runs, reply with plain text only. Do NOT output local paths, "
        "sandbox links, download links, or Markdown image syntax.\n"
        "4. The system will automatically save matplotlib figures generated during execution "
        "into the agent context figures/ directory and send only the newest PNG from this run "
        "back to the user.\n"
        "5. Do not narrate internal tool usage or mention mandatory tool calls."
    )


def prepare_channel_delivery_figures(
    session: Any,
    figures: Optional[Iterable[Any]],
    *,
    prefix: str = "analysis_figure",
) -> List[bytes]:
    items = list(figures or [])
    if not items:
        return []

    saved_paths = persist_workspace_figures(session, items, prefix=prefix)
    if saved_paths:
        try:
            return [saved_paths[-1].read_bytes()]
        except Exception:
            pass

    latest_bytes = _figure_bytes(items[-1])
    return [latest_bytes] if latest_bytes else []


def persist_workspace_figures(
    session: Any,
    figures: Iterable[Any],
    *,
    prefix: str = "analysis_figure",
) -> List[Path]:
    workspace = _session_workspace(session)
    if workspace is None:
        return []

    out_dir = workspace / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    timestamp = int(time.time() * 1000)
    for index, figure in enumerate(figures, start=1):
        data = _figure_bytes(figure)
        if not data:
            continue
        dest = out_dir / f"{prefix}_{timestamp}_{index:02d}.png"
        try:
            dest.write_bytes(data)
            saved.append(dest)
        except Exception:
            continue
    return saved


def _session_workspace(session: Any) -> Optional[Path]:
    workspace = getattr(session, "workspace", None)
    if workspace:
        return Path(workspace)

    workspace_dir = getattr(session, "workspace_dir", None)
    if workspace_dir:
        return Path(workspace_dir) / "workspace"
    return None


def _figure_bytes(figure: Any) -> bytes:
    if isinstance(figure, bytes):
        return figure
    if isinstance(figure, bytearray):
        return bytes(figure)

    path_value = None
    if isinstance(figure, (str, Path)):
        path_value = figure
    elif hasattr(figure, "path"):
        path_value = getattr(figure, "path", None)

    if path_value is None:
        return b""

    path = Path(path_value)
    try:
        if not path.exists() or not path.is_file():
            return b""
        return path.read_bytes()
    except Exception:
        return b""
