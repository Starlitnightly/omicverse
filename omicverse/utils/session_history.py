"""
Append-only JSONL session history for OmicVerse Agent (P2-2).

Records each ``run()`` call (request, generated code, result summary,
token usage) so that future turns can inject prior context into the
system prompt.

Inspired by Codex ``message_history.rs``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HistoryEntry:
    session_id: str
    timestamp: float
    request: str
    generated_code: str = ""
    result_summary: str = ""
    usage: Optional[Dict[str, Any]] = None
    priority_used: int = 0
    success: bool = True


class SessionHistory:
    """Append-only JSONL history file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path.home() / ".ovagent" / "history.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---- write ----

    def append(self, entry: HistoryEntry) -> None:
        line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line)

    # ---- read ----

    def _iter_entries(self):
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        yield HistoryEntry(**json.loads(raw))
                    except Exception:
                        continue

    def get_session(self, session_id: str) -> List[HistoryEntry]:
        return [e for e in self._iter_entries() if e.session_id == session_id]

    def get_recent(self, n: int = 10) -> List[HistoryEntry]:
        # Read all then take last-n (simple for JSONL)
        entries: List[HistoryEntry] = list(self._iter_entries())
        return entries[-n:]

    # ---- LLM context builder ----

    def build_context_for_llm(
        self, session_id: str, max_entries: int = 5
    ) -> str:
        """Build a concise text summary of recent history for the system prompt."""
        entries = self.get_session(session_id)[-max_entries:]
        if not entries:
            return ""
        lines = ["## Previous interactions in this session"]
        for e in entries:
            status = "Success" if e.success else "Failed"
            lines.append(f"- Request: {e.request}")
            lines.append(f"  Result: {status} — {e.result_summary}")
        return "\n".join(lines)
