"""
Telegram message formatting utilities — OpenClaw-inspired design.

Design principles:
  • HTML parse mode throughout
  • Markdown → Telegram HTML conversion for agent output
  • <blockquote expandable> for long prose (collapses to 3 lines)
  • <pre> blocks for code/shell, auto-split at newlines
  • Consistent structure: header + divider + body + footer
  • Split at paragraph boundaries, not arbitrary character positions
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

_MAX_MSG   = 4000   # safe ceiling (Telegram limit is 4096)
_MAX_CAP   = 950    # photo caption limit
_DIV       = "─" * 20


# ---------------------------------------------------------------------------
# Escaping & markdown conversion
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    """HTML-escape raw text for safe embedding inside HTML tags."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def md_to_html(text: str) -> str:
    """Convert a Markdown subset to Telegram-safe HTML.

    Handles: ```code```, `inline`, **bold**, *italic*, # headers
    Escapes raw HTML first so the output is always safe.
    """
    result = esc(text)

    # Fenced code blocks  (must precede inline code)
    result = re.sub(
        r"```(?:\w+)?\n?(.*?)```",
        lambda m: f"<pre>{m.group(1).strip()}</pre>",
        result,
        flags=re.DOTALL,
    )
    # Inline code
    result = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", result)
    # Bold  **text** or __text__
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"__(.+?)__",     r"<b>\1</b>", result)
    # Italic  *text*  (single asterisk, not preceded/followed by *)
    result = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", result)
    # H1-H3 → bold heading
    result = re.sub(
        r"^#{1,3}\s+(.+)$", r"<b>\1</b>", result, flags=re.MULTILINE
    )
    return result


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _para_chunks(text: str, max_len: int) -> List[str]:
    """Split *text* into chunks ≤ *max_len*, preferring paragraph boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    buf = ""

    for para in text.split("\n\n"):
        candidate = (buf + "\n\n" + para).lstrip() if buf else para
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # Para too long → split by lines
            if len(para) <= max_len:
                buf = para
            else:
                for line in para.split("\n"):
                    c2 = (buf + "\n" + line).lstrip() if buf else line
                    if len(c2) <= max_len:
                        buf = c2
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = line[:max_len]

    if buf:
        chunks.append(buf)
    return chunks or [text[:max_len]]


# ---------------------------------------------------------------------------
# send_code  —  <pre> blocks for shell / code output
# ---------------------------------------------------------------------------

async def send_code(
    bot: Any,
    chat_id: int,
    raw_text: str,
    header: str = "",
) -> None:
    """Send *raw_text* as one or more ``<pre>`` code blocks.

    *header* is shown above the first block (e.g. ``"$ ls -lh"``).
    Auto-splits at newlines when the escaped text exceeds the message limit.
    """
    escaped = esc(raw_text)
    prefix  = f"{header}\n{_DIV}\n" if header else ""
    max_body = _MAX_MSG - len(prefix) - len("<pre></pre>")

    first = True
    pos = 0
    while pos < len(escaped):
        end = pos + max_body
        if end < len(escaped):
            nl = escaped.rfind("\n", pos, end)
            if nl > pos:
                end = nl + 1
        chunk = escaped[pos:end]
        body  = f"{prefix}<pre>{chunk}</pre>" if first else f"<pre>{chunk}</pre>"
        await bot.send_message(chat_id=chat_id, text=body, parse_mode="HTML")
        first = False
        pos   = end


# ---------------------------------------------------------------------------
# send_prose  —  markdown prose, expandable blockquote when long
# ---------------------------------------------------------------------------

async def send_prose(
    bot: Any,
    chat_id: int,
    raw_text: str,
    header: str = "",
    always_expand: bool = False,
) -> None:
    """Send markdown prose as Telegram HTML.

    Short text (≤ 600 chars) is sent inline.
    Long text is wrapped in ``<blockquote expandable>`` and split at
    paragraph boundaries.

    Args:
        header: Optional title shown above the first block with a divider.
        always_expand: Force expandable blockquote even for short text.
    """
    body   = md_to_html(raw_text)
    prefix = f"{header}\n{_DIV}\n" if header else ""

    bq_o = "<blockquote expandable>"
    bq_c = "</blockquote>"
    max_body = _MAX_MSG - len(prefix) - len(bq_o) - len(bq_c)

    if len(body) <= 600 and not always_expand:
        # Short — send directly, prepend header if any
        msg = f"{prefix}{body}" if prefix else body
        await bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
        return

    first = True
    for chunk in _para_chunks(body, max_body):
        if first:
            msg = f"{prefix}{bq_o}{chunk}{bq_c}"
        else:
            msg = f"{bq_o}{chunk}{bq_c}"
        await bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
        first = False


# ---------------------------------------------------------------------------
# Canned message builders
# ---------------------------------------------------------------------------

def ack_message(request: str, adata_info: str = "", workspace_hint: str = "") -> str:
    """Acknowledgment sent immediately when a request is received.

    Structure:
        ⚙️  <code>[request]</code>
        ────────────────────
        🔬 [adata info]   OR   💡 [workspace hint]
    """
    short = esc(request[:80] + ("…" if len(request) > 80 else ""))
    lines = [f"⚙️  <code>{short}</code>", _DIV]
    if adata_info:
        lines.append(f"🔬  {adata_info}")
    elif workspace_hint:
        lines.append(workspace_hint)
    return "\n".join(lines)


def progress_message(code_snippet: str) -> str:
    """Progress update shown while agent executes a code cell."""
    snippet = esc(code_snippet[:120])
    return f"🔄  <code>{snippet}</code>"


def result_header(n_figures: int, adata_info: str = "") -> str:
    """Header sent once after analysis completes (before figures + summary)."""
    fig_note = f"  ·  {n_figures} 张图" if n_figures else ""
    lines = [f"✅  <b>分析完成</b>{fig_note}", _DIV]
    if adata_info:
        lines.append(f"📊  <code>{adata_info}</code>")
    return "\n".join(lines)


def error_message(exc_text: str) -> str:
    """Friendly error block."""
    text = exc_text or ""
    if "KernelNotFound" in text or "ipykernel" in text.lower():
        detail = "Jupyter 内核未找到，请安装：\n<code>pip install ipykernel</code>"
    elif "TimeoutError" in text or "timeout" in text.lower():
        detail = "分析超时，请简化请求后重试。"
    elif "MemoryError" in text or "memory" in text.lower():
        detail = "内存不足，请使用 /reset 清理后重试。"
    elif "API" in text or "api" in text.lower():
        detail = f"API 调用失败：\n<code>{esc(text[:200])}</code>"
    else:
        detail = f"<code>{esc(text[:300])}</code>"
    return f"❌  <b>出错</b>\n{_DIV}\n{detail}"


def figure_caption(index: int, total: int) -> str:
    """Caption for a figure photo."""
    return f"🖼  图 {index} / {total}"
