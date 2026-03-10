"""
Shared /model command output helpers for Jarvis channels.
"""
from __future__ import annotations

from html import escape
from typing import List, Sequence

from .model_registry import iter_supported_model_catalog

_MODEL_SWITCH_EXAMPLES: Sequence[str] = (
    "gpt-5.3-codex",
    "claude-sonnet-4-6",
    "qwen-max",
    "qwen3:8b",
)
_CUSTOM_PROVIDER_NAMES = {"ollama", "openai_compatible"}


def _example_models(current_model: str) -> List[str]:
    examples: List[str] = []
    current = str(current_model or "").strip()
    for candidate in (current, *_MODEL_SWITCH_EXAMPLES):
        if candidate and candidate not in examples:
            examples.append(candidate)
    return examples[:4]


def render_model_help(current_model: str, *, html: bool = False) -> str:
    examples = _example_models(current_model)
    lines: List[str] = []

    if html:
        lines.append(f"🤖  当前模型：<code>{escape(str(current_model or 'unknown'))}</code>")
        lines.append("────────────")
        lines.append("切换示例：")
        for model_id in examples:
            lines.append(f"• <code>/model {escape(model_id)}</code>")
        lines.append("")
        lines.append("<b>支持模型</b>")
        for provider_name, display_name, models in iter_supported_model_catalog():
            lines.append(f"<b>{escape(display_name)}</b>")
            for model_id in models:
                lines.append(f"• <code>{escape(model_id)}</code>")
            if provider_name in _CUSTOM_PROVIDER_NAMES:
                lines.append("  <i>也支持输入该 endpoint 上的其他模型 ID</i>")
            lines.append("")
        lines.append("<i>切换后请 /reset 重启 kernel 使新模型生效。</i>")
        return "\n".join(lines).strip()

    lines.append(f"🤖 当前模型: {current_model or 'unknown'}")
    lines.append("--------------------")
    lines.append("切换示例:")
    for model_id in examples:
        lines.append(f"- /model {model_id}")
    lines.append("")
    lines.append("支持模型:")
    for provider_name, display_name, models in iter_supported_model_catalog():
        lines.append(f"{display_name}:")
        for model_id in models:
            lines.append(f"- {model_id}")
        if provider_name in _CUSTOM_PROVIDER_NAMES:
            lines.append("  也支持输入该 endpoint 上的其他模型 ID")
        lines.append("")
    lines.append("切换后请 /reset 重启 kernel 使新模型生效。")
    return "\n".join(lines).strip()
