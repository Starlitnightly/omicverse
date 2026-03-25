"""
AgentBridge — wraps ov.Agent.stream_async() and harvests matplotlib figures.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional, Set


def _content_key(data: bytes) -> str:
    """Short content-based dedup key for PNG bytes (first 1 KB is enough)."""
    return "sha:" + hashlib.sha256(data[:1024]).hexdigest()[:20]


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
    diagnostics: List[str]   = field(default_factory=list)


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

    async def run(
        self,
        request: str,
        adata: Optional[Any],
        history: Optional[List[dict]] = None,
        request_content: Optional[List[dict]] = None,
    ) -> AgentRunResult:
        result = AgentRunResult()
        seen: Set[str] = set()
        diag_seen: Set[str] = set()
        self._run_started_at = time.time()

        async for event in self._agent.stream_async(
            request,
            adata,
            history=history or [],
            request_content=request_content or [],
        ):
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

            elif etype == "tool_call":
                call = content if isinstance(content, dict) else {}
                name = str(call.get("name") or "tool")
                args = call.get("arguments")
                arg_keys: List[str] = []
                if isinstance(args, dict):
                    arg_keys = [str(k) for k in args.keys()]
                if arg_keys:
                    await self._progress(f"工具: {name}({', '.join(arg_keys[:4])})")
                else:
                    await self._progress(f"工具: {name}(无参数)")
                    self._push_diagnostic(result, diag_seen, f"{name} 调用时参数为空")

            elif etype == "tool_result":
                payload = content if isinstance(content, dict) else {}
                name = str(payload.get("name") or "tool")
                output = str(payload.get("output") or "")
                if output:
                    msg = self._tool_output_diagnostic(name, output)
                    if msg:
                        self._push_diagnostic(result, diag_seen, msg)

            elif etype == "status":
                info = content if isinstance(content, dict) else {}
                if info.get("follow_up_exhausted"):
                    self._push_diagnostic(
                        result,
                        diag_seen,
                        "多轮尝试后仍未得到有效工具参数或可执行方案",
                    )

            elif etype == "finish":
                result.summary = str(content) if content else ""

            elif etype == "done":
                done_text = str(content or "").strip()
                if done_text:
                    result.summary = done_text
                if event.get("max_turns"):
                    self._push_diagnostic(
                        result,
                        diag_seen,
                        "达到最大轮次，模型未给出可执行完成路径",
                    )
                if event.get("error") and done_text:
                    result.error = done_text

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

    @staticmethod
    def pick_reply_text(
        result: "AgentRunResult",
        llm_buf: str,
        *,
        max_len: int = 1800,
        boring: Optional[Set[str]] = None,
    ) -> str:
        """Return the best text to send back to the user.

        Priority rules
        --------------
        1. When **no analysis artifacts** were produced (no figures, reports,
           or file attachments) *and* the agent streamed LLM text, that text
           IS the reply — the agent gave a conversational answer.  The
           ``finish()`` summary in this case is just a mandatory-tool-use
           no-op, so we ignore it.
        2. Otherwise fall back to ``result.summary`` unless it is boring/empty,
           in which case try ``llm_buf``, then diagnostics, then a generic
           completion string.
        """
        _boring: Set[str] = boring if boring is not None else {
            "分析完成", "分析完成。", "task completed", "done", "完成",
        }
        llm_text = (llm_buf or "").strip()
        summary = (result.summary or "").strip()
        has_artifacts = bool(result.reports or result.figures or result.artifacts)

        if not has_artifacts and llm_text:
            return llm_text[:max_len]

        if not summary or summary.lower() in _boring:
            if llm_text:
                return llm_text[:max_len]
            if result.diagnostics:
                hints = "\n".join(f"- {x}" for x in result.diagnostics[:5])
                return f"未生成有效答复\n{hints}"
            return "分析完成"

        return summary[:max_len]

    async def _progress(self, msg: str) -> None:
        if self._progress_cb is not None:
            try:
                await self._progress_cb(msg)
            except Exception:
                pass

    @staticmethod
    def _push_diagnostic(result: AgentRunResult, seen: Set[str], message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        if msg in seen:
            return
        seen.add(msg)
        result.diagnostics.append(msg)

    @staticmethod
    def _tool_output_diagnostic(tool_name: str, output: str) -> str:
        text = (output or "").strip()
        if not text:
            return ""
        first_line = text.splitlines()[0].strip()
        if first_line.lower().startswith("error:"):
            return f"{tool_name} 返回错误: {first_line[6:].strip() or '未知错误'}"
        if first_line.lower().startswith("search error:"):
            return f"{tool_name} 返回错误: {first_line}"
        return ""

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
                        data = base64.b64decode(b64)
                        # Record content hash so _harvest_file_figures skips duplicates
                        seen.add(_content_key(data))
                        figs.append(data)
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

        png_candidates: List[Path] = []
        for d in self._candidate_dirs(roots):
            try:
                png_candidates.extend(d.rglob("*.png"))
            except Exception:
                pass

        # Prefer newest figures; ignore files older than this run.
        png_candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for p in png_candidates[:40]:
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
                ck = _content_key(data)
                if ck in seen:
                    # Same image already harvested from notebook display_data — skip.
                    seen.add(key)
                    continue
                seen.add(key)
                seen.add(ck)
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

        md_candidates: List[Path] = []
        for d in self._candidate_dirs(roots):
            try:
                md_candidates.extend(d.rglob("*.md"))
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
        candidates: List[Path] = []
        for d in self._candidate_dirs(roots):
            try:
                for p in d.rglob("*"):
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

    @staticmethod
    def _candidate_dirs(roots: List[Path]) -> List[Path]:
        """Return unique existing directories to scan (root + root/output + root/figures)."""
        dirs: List[Path] = []
        seen: Set[Path] = set()
        for root in roots:
            for d in (root, root / "output", root / "figures"):
                try:
                    rd = d.resolve()
                except Exception:
                    rd = d
                if rd in seen or not d.exists() or not d.is_dir():
                    continue
                seen.add(rd)
                dirs.append(d)
        return dirs
