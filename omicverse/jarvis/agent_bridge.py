"""
AgentBridge — wraps ov.Agent.stream_async() and harvests matplotlib figures.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional, Set


@dataclass
class AgentRunResult:
    adata:   Optional[Any]   = None
    figures: List[bytes]     = field(default_factory=list)
    summary: str             = ""
    error:   Optional[str]   = None
    usage:   Optional[Any]   = None   # token usage object from the LLM


class AgentBridge:
    """
    Thin async wrapper around ``ov.Agent`` that:
    - drives ``stream_async()``
    - fires *progress_cb* with human-readable status strings (code events)
    - fires *llm_chunk_cb* with each raw LLM text token (for streaming UI)
    - harvests PNG figures from the live notebook after each ``'result'`` event
    - captures token usage from the final ``'usage'`` event
    """

    def __init__(
        self,
        agent: Any,
        progress_cb:   Optional[Callable[[str], Coroutine]] = None,
        llm_chunk_cb:  Optional[Callable[[str], Coroutine]] = None,
    ) -> None:
        self._agent        = agent
        self._progress_cb  = progress_cb
        self._llm_chunk_cb = llm_chunk_cb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: str, adata: Optional[Any]) -> AgentRunResult:
        result = AgentRunResult()
        seen: Set[str] = set()

        async for event in self._agent.stream_async(request, adata):
            etype   = event.get("type")
            content = event.get("content")

            if etype == "llm_chunk":
                chunk = str(content or "")
                if chunk and self._llm_chunk_cb is not None:
                    try:
                        await self._llm_chunk_cb(chunk)
                    except Exception:
                        pass

            elif etype == "code":
                # Prefer the human-readable description the agent supplied;
                # fall back to extracting the first meaningful line of code.
                desc = event.get("description") or self._extract_description(str(content))
                await self._progress(desc)

            elif etype == "result":
                result.adata = content
                result.figures.extend(self._harvest_figures(seen))

            elif etype == "finish":
                result.summary = str(content) if content else ""

            elif etype == "error":
                result.error = str(content)

            elif etype == "usage":
                result.usage = content

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _progress(self, msg: str) -> None:
        if self._progress_cb is not None:
            try:
                await self._progress_cb(msg)
            except Exception:
                pass

    @staticmethod
    def _extract_description(code: str) -> str:
        """Return the first non-empty, non-comment line of *code* (≤ 80 chars)."""
        for line in code.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped[:80]
        return code[:80]

    def _harvest_figures(self, seen: Set[str]) -> List[bytes]:
        """
        Walk the last 10 notebook cells for ``display_data`` outputs
        containing ``image/png`` and return new PNG bytes (not yet in *seen*).
        """
        figs: List[bytes] = []
        try:
            executor = getattr(self._agent, "_notebook_executor", None)
            if executor is None:
                return figs
            session = getattr(executor, "current_session", None)
            if session is None:
                return figs
            nb = session.get("notebook")
            if nb is None:
                return figs
            for cell in nb.get("cells", [])[-10:]:
                for output in cell.get("outputs", []):
                    if output.get("output_type") != "display_data":
                        continue
                    b64 = output.get("data", {}).get("image/png")
                    if not b64:
                        continue
                    key = b64[:64]
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        figs.append(base64.b64decode(b64))
                    except Exception:
                        pass
        except Exception:
            pass
        return figs
