"""
JarvisSession and SessionManager — per-user state for the Telegram bot.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# WorkspaceShell
# ---------------------------------------------------------------------------

ALLOWED_CMDS = {"ls", "find", "cat", "head", "wc", "file", "du", "pwd", "tree"}


class WorkspaceShell:
    """Execute a whitelist of read-only shell commands inside the workspace."""

    def exec(self, cmd: str, cwd: Path, timeout: int = 10) -> str:
        try:
            parts = shlex.split(cmd)
        except ValueError as exc:
            return f"❌ 命令解析失败：{exc}"
        if not parts:
            return "❌ 空命令"
        if parts[0] not in ALLOWED_CMDS:
            return (
                f"❌ 不允许执行 '{parts[0]}'。\n"
                f"允许的命令：{sorted(ALLOWED_CMDS)}"
            )
        try:
            result = subprocess.run(
                parts,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (result.stdout + result.stderr).strip()
            return out if out else "(无输出)"
        except subprocess.TimeoutExpired:
            return f"❌ 命令超时（>{timeout}s）"
        except Exception as exc:
            return f"❌ 执行出错：{exc}"


# ---------------------------------------------------------------------------
# JarvisSession
# ---------------------------------------------------------------------------

@dataclass
class JarvisSession:
    """Holds per-user state: workspace path, live Agent, and loaded adata."""

    user_id: int
    workspace_dir: Path          # ~/.ovjarvis/<user_id>/
    agent: Any                   # ov.Agent (OmicVerseAgent)
    adata: Optional[Any] = None
    prompt_count: int = 0
    shell: WorkspaceShell = field(default_factory=WorkspaceShell)
    last_usage: Optional[Any] = None   # token usage from most recent analysis

    # Convenience property: the actual data directory
    @property
    def workspace(self) -> Path:
        return self.workspace_dir / "workspace"

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def list_h5ad_files(self) -> List[Path]:
        """Return all .h5ad files under the workspace directory."""
        return sorted(self.workspace.rglob("*.h5ad"))

    def load_from_workspace(self, filename: str) -> Optional[Any]:
        """Find *filename* in workspace and load it as adata.

        Returns the loaded AnnData on success, None on failure.
        """
        # Accept bare filename or relative path
        candidates = list(self.workspace.rglob(filename))
        if not candidates:
            return None
        target = candidates[0]
        import scanpy as sc
        self.adata = sc.read_h5ad(str(target))
        self.save_adata()
        return self.adata

    def get_agents_md(self) -> Optional[str]:
        """Return contents of workspace/AGENTS.md if it exists."""
        agents_md = self.workspace / "AGENTS.md"
        if agents_md.exists():
            try:
                return agents_md.read_text(encoding="utf-8").strip()
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Memory system (OpenClaw-style)
    # ------------------------------------------------------------------

    @property
    def memory_dir(self) -> Path:
        """Return (and create) the workspace/memory/ directory."""
        d = self.workspace / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def append_memory_log(
        self, request: str, summary: str, adata_info: str = ""
    ) -> None:
        """Append one analysis entry to today's daily log.

        File: ``workspace/memory/YYYY-MM-DD.md``
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")

        log_file = self.memory_dir / f"{date_str}.md"
        is_new = not log_file.exists()

        # Truncate long fields so the context doesn't blow up
        req_short = request[:200] + ("..." if len(request) > 200 else "")
        sum_short = summary[:500] + ("..." if len(summary) > 500 else "")

        entry = f"\n## {time_str}\n**请求**: {req_short}\n**摘要**: {sum_short}"
        if adata_info:
            entry += f"\n**数据**: {adata_info}"
        entry += "\n"

        with open(str(log_file), "a", encoding="utf-8") as fh:
            if is_new:
                fh.write(f"# 分析日志 {date_str}\n")
            fh.write(entry)

    def get_memory_context(self) -> Optional[str]:
        """Return recent memory for injection into the Agent prompt.

        Loads (in order):
        1. ``workspace/MEMORY.md`` — long-term curated notes
        2. ``workspace/memory/yesterday.md``
        3. ``workspace/memory/today.md``
        """
        parts: List[str] = []

        # Long-term memory
        long_term = self.workspace / "MEMORY.md"
        if long_term.exists():
            try:
                content = long_term.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"## Long-term memory\n{content}")
            except Exception:
                pass

        # Daily logs: yesterday + today
        today = datetime.now().date()
        for date in (today - timedelta(days=1), today):
            log_file = self.memory_dir / f"{date}.md"
            if log_file.exists():
                try:
                    content = log_file.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except Exception:
                    pass

        return "\n\n".join(parts) if parts else None

    def get_recent_memory_text(self, max_entries: int = 5) -> str:
        """Return a human-readable summary of recent analyses (for /memory cmd)."""
        lines: List[str] = []
        today = datetime.now().date()
        for date in (today - timedelta(days=1), today):
            log_file = self.memory_dir / f"{date}.md"
            if log_file.exists():
                try:
                    lines.append(log_file.read_text(encoding="utf-8").strip())
                except Exception:
                    pass
        return "\n\n".join(lines) if lines else "(暂无分析历史)"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_adata(self) -> Optional[Path]:
        """Write adata to ``workspace_dir/current.h5ad``.  Returns path or None."""
        if self.adata is None:
            return None
        path = self.workspace_dir / "current.h5ad"
        self.adata.write_h5ad(str(path))
        return path

    def load_adata(self) -> Optional[Any]:
        """Read ``workspace_dir/current.h5ad`` if it exists.  Updates self.adata."""
        path = self.workspace_dir / "current.h5ad"
        if path.exists():
            import scanpy as sc
            self.adata = sc.read_h5ad(str(path))
        return self.adata

    def kernel_status(self) -> dict:
        """Return kernel health info from the underlying notebook executor."""
        info: dict = {"alive": False, "prompt_count": 0, "max_prompts": "?", "session_id": None}
        try:
            executor = getattr(self.agent, "_notebook_executor", None)
            if executor is None:
                return info
            info["max_prompts"] = getattr(executor, "max_prompts_per_session", "?")
            info["prompt_count"] = getattr(executor, "session_prompt_count", 0)
            session = getattr(executor, "current_session", None)
            if session is None:
                return info
            info["session_id"] = session.get("session_id")
            # Check if kernel process is alive
            try:
                alive = executor._is_kernel_alive()
            except Exception:
                alive = False
            info["alive"] = alive
        except Exception:
            pass
        return info

    def reset(self) -> None:
        """Clear adata and restart the agent session."""
        self.adata = None
        self.prompt_count = 0
        h5ad = self.workspace_dir / "current.h5ad"
        if h5ad.exists():
            try:
                h5ad.unlink()
            except OSError:
                pass
        try:
            self.agent.restart_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Lazy-creates and caches one ``JarvisSession`` per Telegram user."""

    def __init__(
        self,
        *,
        session_dir: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_prompts: int = 50,
        verbose: bool = False,
    ) -> None:
        self._base      = Path(session_dir or os.path.expanduser("~/.ovjarvis"))
        self._model     = model
        self._api_key   = api_key
        self._max_prompts = max_prompts  # 50 keeps kernel alive for ~50 analyses
        self._verbose = verbose
        self._sessions: Dict[int, JarvisSession] = {}

    # ------------------------------------------------------------------

    def get_or_create(self, user_id: int) -> JarvisSession:
        """Return existing session or build a new one for *user_id*."""
        if user_id in self._sessions:
            return self._sessions[user_id]

        user_dir = self._base / str(user_id)

        # Create all sub-directories
        workspace_dir = user_dir / "workspace"
        sessions_dir  = user_dir / "sessions"
        context_dir   = user_dir / "context"
        memory_dir    = workspace_dir / "memory"
        for d in (user_dir, workspace_dir, sessions_dir, context_dir, memory_dir):
            d.mkdir(parents=True, exist_ok=True)

        agent = self._build_agent(user_id)
        session = JarvisSession(
            user_id=user_id,
            workspace_dir=user_dir,
            agent=agent,
        )
        self._sessions[user_id] = session
        return session

    def _build_agent(self, user_id: int) -> Any:
        import omicverse as ov

        user_dir = self._base / str(user_id)
        kwargs: Dict[str, Any] = dict(
            model=self._model,
            use_notebook_execution=True,
            strict_kernel_validation=False,
            verbose=self._verbose,
            max_prompts_per_session=self._max_prompts,
            notebook_storage_dir=str(user_dir / "sessions"),
            context_storage_dir=str(user_dir / "context"),
            enable_filesystem_context=True,
        )
        if self._api_key:
            kwargs["api_key"] = self._api_key

        return ov.Agent(**kwargs)
