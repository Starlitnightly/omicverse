"""Alibaba DashScope (Qwen) adapter helpers for OmicVerseLLMBackend.

Internal module — import from ``omicverse.utils.agent_backend`` instead.
"""
from __future__ import annotations

import logging

from .agent_backend_common import Usage, _coerce_int, _compute_total

logger = logging.getLogger(__name__)


def _chat_via_dashscope(backend, user_prompt: str) -> str:
    api_key = backend._resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY for Qwen provider")
    try:
        from http import HTTPStatus
        from dashscope import Generation  # type: ignore

        messages = [
            {"role": "system", "content": backend.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Wrap DashScope SDK call with retry logic
        def _make_dashscope_call():
            resp = Generation.call(
                model=backend.config.model.replace("/", ":"),
                messages=messages,
                api_key=api_key,
                temperature=backend.config.temperature,
                max_tokens=backend.config.max_tokens,
            )
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(f"DashScope call failed: {getattr(resp, 'message', resp.status_code)}")

            # Capture usage information from DashScope
            if hasattr(resp, 'usage') and resp.usage is not None:
                usage = resp.usage
                # DashScope uses input_tokens and output_tokens
                input_tokens = _coerce_int(getattr(usage, 'input_tokens', None))
                output_tokens = _coerce_int(getattr(usage, 'output_tokens', None))
                total_tokens = _compute_total(input_tokens, output_tokens, _coerce_int(getattr(usage, 'total_tokens', None)))
                if total_tokens is not None:
                    backend.last_usage = Usage(
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                        total_tokens=total_tokens,
                        model=backend.config.model,
                        provider=backend.config.provider
                    )

            # Qwen responses are OpenAI-like
            choices = getattr(resp, "output", {}).get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

        return backend._retry(_make_dashscope_call)

    except ImportError:
        raise RuntimeError(
            "dashscope package not installed. Install it or choose an OpenAI-compatible model."
        )
