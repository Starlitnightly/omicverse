"""
AgentBridge — wraps ov.Agent.stream_async() and harvests matplotlib figures.
"""
from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional, Set


@dataclass
class AgentArtifact:
    filename: str
    data: bytes


@dataclass
class AgentRunResult:
    adata:   Optional[Any]   = None
    figures: List[bytes]     = field(default_factory=list)
    reports: List[str]       = field(default_factory=list)
    artifacts: List[AgentArtifact] = field(default_factory=list)
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
        self._run_started_at = time.time()

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

        # Final harvest in case figures were written to disk late in the run.
        result.figures.extend(self._harvest_figures(seen))
        result.reports.extend(self._harvest_reports())
        result.artifacts.extend(self._harvest_artifacts())
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
        containing ``image/png`` and also scan common output directories
        for freshly written PNG files.
        """
        figs: List[bytes] = []
        figs.extend(self._harvest_notebook_figures(seen))
        figs.extend(self._harvest_file_figures(seen))
        return figs

    def _harvest_notebook_figures(self, seen: Set[str]) -> List[bytes]:
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

    def _harvest_file_figures(self, seen: Set[str]) -> List[bytes]:
        figs: List[bytes] = []
        run_started_at = getattr(self, "_run_started_at", 0.0)

        roots: List[Path] = []
        try:
            executor = getattr(self._agent, "_notebook_executor", None)
            if executor is not None:
                session = getattr(executor, "current_session", None)
                if session and session.get("session_dir"):
                    roots.append(Path(session["session_dir"]))
        except Exception:
            pass
        try:
            fs_ctx = getattr(self._agent, "_filesystem_context", None)
            ws = getattr(fs_ctx, "_workspace_dir", None)
            if ws:
                roots.append(Path(ws))
        except Exception:
            pass
        # Always include cwd so relative-path saves (e.g. ./output/*.png) are found
        try:
            roots.append(Path.cwd())
        except Exception:
            pass

        scanned: Set[Path] = set()
        png_candidates: List[Path] = []
        for root in roots:
            for d in (root, root / "output"):
                if d in scanned or not d.exists() or not d.is_dir():
                    continue
                scanned.add(d)
                try:
                    png_candidates.extend(d.glob("*.png"))
                except Exception:
                    pass

        # Prefer newest figures; ignore files older than this run.
        png_candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for p in png_candidates[:20]:
            try:
                st = p.stat()
                if st.st_mtime + 0.2 < run_started_at:
                    continue
                key = f"file:{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"
                if key in seen:
                    continue
                data = p.read_bytes()
                if not data:
                    continue
                seen.add(key)
                figs.append(data)
            except Exception:
                pass
        return figs

    def _harvest_reports(self) -> List[str]:
        reports: List[str] = []
        run_started_at = getattr(self, "_run_started_at", 0.0)
        roots: List[Path] = []
        try:
            executor = getattr(self._agent, "_notebook_executor", None)
            if executor is not None:
                session = getattr(executor, "current_session", None)
                if session and session.get("session_dir"):
                    roots.append(Path(session["session_dir"]))
        except Exception:
            pass
        try:
            fs_ctx = getattr(self._agent, "_filesystem_context", None)
            ws = getattr(fs_ctx, "_workspace_dir", None)
            if ws:
                roots.append(Path(ws))
        except Exception:
            pass

        scanned: Set[Path] = set()
        md_candidates: List[Path] = []
        for root in roots:
            for d in (root, root / "output"):
                if d in scanned or not d.exists() or not d.is_dir():
                    continue
                scanned.add(d)
                try:
                    md_candidates.extend(d.glob("*.md"))
                except Exception:
                    pass

        md_candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for p in md_candidates[:5]:
            try:
                st = p.stat()
                if st.st_mtime + 0.2 < run_started_at:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                reports.append(text[:12000])
            except Exception:
                pass
        return reports

    def _harvest_artifacts(self) -> List[AgentArtifact]:
        artifacts: List[AgentArtifact] = []
        run_started_at = getattr(self, "_run_started_at", 0.0)
        roots: List[Path] = []
        try:
            executor = getattr(self._agent, "_notebook_executor", None)
            if executor is not None:
                session = getattr(executor, "current_session", None)
                if session and session.get("session_dir"):
                    roots.append(Path(session["session_dir"]))
        except Exception:
            pass
        try:
            fs_ctx = getattr(self._agent, "_filesystem_context", None)
            ws = getattr(fs_ctx, "_workspace_dir", None)
            if ws:
                roots.append(Path(ws))
        except Exception:
            pass
        # Some tools write to process cwd/output
        roots.append(Path.cwd())

        exts = {".md", ".pdf", ".csv", ".tsv", ".txt", ".html", ".xlsx"}
        scanned: Set[Path] = set()
        candidates: List[Path] = []
        for root in roots:
            for d in (root, root / "output"):
                if d in scanned or not d.exists() or not d.is_dir():
                    continue
                scanned.add(d)
                try:
                    for p in d.iterdir():
                        if p.is_file() and p.suffix.lower() in exts:
                            candidates.append(p)
                except Exception:
                    pass

        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        seen_key: Set[str] = set()
        for p in candidates[:12]:
            try:
                st = p.stat()
                if st.st_mtime + 0.2 < run_started_at:
                    continue
                key = f"{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"
                if key in seen_key:
                    continue
                data = p.read_bytes()
                if not data:
                    continue
                seen_key.add(key)
                artifacts.append(AgentArtifact(filename=p.name, data=data))
            except Exception:
                pass
        return artifacts
