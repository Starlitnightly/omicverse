"""
Agent Session Service — Per-session state for multi-turn chatbot
================================================================

Manages chat sessions with:
- Per-session chat history and adata reference
- TTL-based cleanup of idle sessions
- Memory budget enforcement (``OV_WEB_MAX_SESSIONS``)
- Optional JSONL history persistence
- Active turn tracking for server-side cancel support
"""

import os
import json
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("omicverse_web.agent_session")

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

MAX_SESSIONS = int(os.environ.get("OV_WEB_MAX_SESSIONS", "3"))
SESSION_TTL_SECONDS = int(os.environ.get("OV_WEB_SESSION_TTL", "3600"))  # 1h default
HISTORY_DIR = os.environ.get("OV_WEB_HISTORY_DIR", "")  # empty = no persistence


# ---------------------------------------------------------------------------
# Chat message dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single message in the chat history."""
    role: str          # "user" | "assistant"
    content: str
    turn_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ApprovalRequest:
    """A pending approval attached to an agent turn."""
    approval_id: str
    turn_id: str
    session_id: str
    title: str
    message: str
    code: str = ""
    violations: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = "pending"   # pending | approved | denied | cancelled
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    decision: str = ""

    def resolve(self, decision: str):
        self.status = "approved" if decision == "approve" else "denied"
        self.decision = decision
        self.resolved_at = time.time()

    def to_dict(self) -> dict:
        return {
            "approval_id": self.approval_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "title": self.title,
            "message": self.message,
            "code": self.code,
            "violations": list(self.violations),
            "metadata": dict(self.metadata),
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "decision": self.decision,
        }


@dataclass
class QuestionRequest:
    """A pending question attached to an agent turn."""
    question_id: str
    turn_id: str
    session_id: str
    title: str
    message: str
    placeholder: str = ""
    options: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = "pending"   # pending | answered | cancelled
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    answer: str = ""

    def resolve(self, answer: str):
        self.status = "answered"
        self.answer = answer
        self.resolved_at = time.time()

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "title": self.title,
            "message": self.message,
            "placeholder": self.placeholder,
            "options": list(self.options),
            "metadata": dict(self.metadata),
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "answer": self.answer,
        }


@dataclass
class TaskRecord:
    """A lightweight runtime task derived from harness item lifecycle."""
    task_id: str
    turn_id: str
    session_id: str
    title: str
    item_type: str = "tool_call"
    step_id: str = ""
    status: str = "pending"   # pending | in_progress | completed | failed | cancelled
    summary: str = ""
    output_summary: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def finish(self, status: str, *, output_summary: str = "", error: str = "", metadata: Optional[dict] = None):
        self.status = status
        self.output_summary = output_summary
        self.error = error
        self.completed_at = time.time()
        if metadata:
            merged = dict(self.metadata)
            merged.update(metadata)
            self.metadata = merged

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "title": self.title,
            "item_type": self.item_type,
            "step_id": self.step_id,
            "status": self.status,
            "summary": self.summary,
            "output_summary": self.output_summary,
            "error": self.error,
            "metadata": dict(self.metadata),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# Session object
# ---------------------------------------------------------------------------

@dataclass
class AgentSession:
    """State for one chat session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    history: list = field(default_factory=list)    # list[ChatMessage]
    adata: Any = None                              # session-scoped adata ref
    adata_is_owned: bool = False                   # True after mutation (copy-on-write)
    active_turn_id: Optional[str] = None           # currently running turn
    active_cancel: Optional[threading.Event] = None  # cancel signal for active turn
    trace_ids: list[str] = field(default_factory=list)
    approvals: dict[str, ApprovalRequest] = field(default_factory=dict)
    questions: dict[str, QuestionRequest] = field(default_factory=dict)
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    loaded_tools: list[str] = field(default_factory=list)
    active_tool_name: str = ""
    active_step_id: str = ""
    last_tool_name: str = ""
    plan_mode: str = "off"
    worktree: dict = field(default_factory=dict)

    def touch(self):
        """Update last-active timestamp."""
        self.last_active = time.time()

    def add_message(self, role: str, content: str, turn_id: str = ""):
        msg = ChatMessage(role=role, content=content, turn_id=turn_id)
        self.history.append(msg)
        self.touch()
        return msg

    def get_history_dicts(self) -> list[dict]:
        """Return serializable history."""
        return [m.to_dict() for m in self.history]

    def set_adata(self, adata_obj: Any):
        """Set session-scoped adata (marks as owned copy)."""
        self.adata = adata_obj
        self.adata_is_owned = True
        self.touch()

    def register_turn(self, turn_id: str, cancel_event: threading.Event):
        """Register an active turn for cancel support."""
        self.active_turn_id = turn_id
        self.active_cancel = cancel_event
        self.touch()

    def clear_turn(self):
        """Clear active turn tracking."""
        self.active_turn_id = None
        self.active_cancel = None

    def register_trace(self, trace_id: str):
        """Record a trace id emitted during this session."""
        if trace_id and trace_id not in self.trace_ids:
            self.trace_ids.append(trace_id)
        self.touch()

    def register_approval(self, approval: ApprovalRequest):
        """Attach a pending approval to this session."""
        self.approvals[approval.approval_id] = approval
        self.touch()

    def register_question(self, question: QuestionRequest):
        """Attach a pending question to this session."""
        self.questions[question.question_id] = question
        self.touch()

    def get_pending_approvals(self) -> list[dict]:
        return [
            approval.to_dict()
            for approval in self.approvals.values()
            if approval.status == "pending"
        ]

    def get_pending_questions(self) -> list[dict]:
        return [
            question.to_dict()
            for question in self.questions.values()
            if question.status == "pending"
        ]

    def resolve_approval(self, approval_id: str, decision: str) -> Optional[dict]:
        approval = self.approvals.get(approval_id)
        if approval is None:
            return None
        approval.resolve(decision)
        self.touch()
        return approval.to_dict()

    def resolve_question(self, question_id: str, answer: str) -> Optional[dict]:
        question = self.questions.get(question_id)
        if question is None:
            return None
        question.resolve(answer)
        self.touch()
        return question.to_dict()

    def register_tool(self, tool_name: str):
        if tool_name and tool_name not in self.loaded_tools:
            self.loaded_tools.append(tool_name)
        if tool_name:
            self.last_tool_name = tool_name
        self.touch()

    def set_plan_mode(self, mode: Any):
        if isinstance(mode, dict):
            if "enabled" in mode:
                self.plan_mode = "on" if mode.get("enabled") else "off"
            else:
                self.plan_mode = str(mode or "off")
        elif isinstance(mode, bool):
            self.plan_mode = "on" if mode else "off"
        else:
            self.plan_mode = str(mode or "off")
        self.touch()

    def set_worktree(self, worktree: Any):
        if isinstance(worktree, dict):
            self.worktree = dict(worktree)
        elif worktree:
            self.worktree = {"label": str(worktree)}
        else:
            self.worktree = {}
        self.touch()

    def _get_or_create_task(
        self,
        *,
        task_id: str,
        turn_id: str,
        title: str,
        item_type: str,
        step_id: str,
        summary: str = "",
        status: str = "pending",
        metadata: Optional[dict] = None,
    ) -> TaskRecord:
        task = self.tasks.get(task_id)
        if task is None:
            task = TaskRecord(
                task_id=task_id,
                turn_id=turn_id,
                session_id=self.session_id,
                title=title,
                item_type=item_type,
                step_id=step_id,
                status=status,
                summary=summary,
                metadata=dict(metadata or {}),
            )
            self.tasks[task_id] = task
        else:
            task.title = title or task.title
            task.item_type = item_type or task.item_type
            task.step_id = step_id or task.step_id
            task.summary = summary or task.summary
            task.status = status or task.status
            if metadata:
                merged = dict(task.metadata)
                merged.update(metadata)
                task.metadata = merged
        return task

    def start_task(
        self,
        *,
        task_id: str,
        turn_id: str,
        title: str,
        item_type: str,
        step_id: str = "",
        summary: str = "",
        metadata: Optional[dict] = None,
    ) -> TaskRecord:
        task = self._get_or_create_task(
            task_id=task_id,
            turn_id=turn_id,
            title=title,
            item_type=item_type,
            step_id=step_id,
            summary=summary,
            status="in_progress",
            metadata=metadata,
        )
        self.active_tool_name = title or self.active_tool_name
        self.active_step_id = step_id or self.active_step_id
        self.touch()
        return task

    def finish_task(
        self,
        task_id: str,
        *,
        status: str,
        output_summary: str = "",
        error: str = "",
        metadata: Optional[dict] = None,
    ) -> Optional[TaskRecord]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        task.finish(status, output_summary=output_summary, error=error, metadata=metadata)
        if task.step_id and task.step_id == self.active_step_id:
            self.active_step_id = ""
            self.active_tool_name = ""
        self.touch()
        return task

    def list_tasks(self, limit: int = 20) -> list[dict]:
        tasks = sorted(
            self.tasks.values(),
            key=lambda task: task.started_at,
            reverse=True,
        )
        return [task.to_dict() for task in tasks[:limit]]

    def get_runtime_state(self, task_limit: int = 8) -> dict:
        worktree_label = (
            self.worktree.get("label")
            or self.worktree.get("path")
            or self.worktree.get("name")
            or ""
        )
        return {
            "loaded_tools": list(self.loaded_tools),
            "active_tool_name": self.active_tool_name,
            "active_step_id": self.active_step_id,
            "last_tool_name": self.last_tool_name,
            "plan_mode": self.plan_mode,
            "worktree": dict(self.worktree),
            "worktree_label": worktree_label,
            "tasks": self.list_tasks(limit=task_limit),
            "task_count": len(self.tasks),
            "pending_questions": self.get_pending_questions(),
            "pending_approvals": self.get_pending_approvals(),
        }

    def apply_runtime_event(self, event: dict):
        """Project a harness/web event into lightweight session runtime state."""
        etype = event.get("type", "")
        content = event.get("content") or {}
        step_id = event.get("step_id", "")
        turn_id = event.get("turn_id", self.active_turn_id or "")
        metadata = dict(event.get("metadata", {}))

        if isinstance(content, dict):
            if "loaded_tools" in content and isinstance(content["loaded_tools"], list):
                for name in content["loaded_tools"]:
                    self.register_tool(str(name))
            if "plan_mode" in content:
                self.set_plan_mode(content.get("plan_mode"))
            if "worktree" in content:
                self.set_worktree(content.get("worktree"))

        if etype == "tool_call":
            name = str(content.get("name", "")).strip()
            if name:
                self.register_tool(name)
                self.active_tool_name = name
                self.active_step_id = step_id or self.active_step_id
        elif etype == "item_started":
            item_type = str(content.get("item_type", "tool_call")).strip() or "tool_call"
            name = str(content.get("name", "")).strip() or item_type
            self.register_tool(name)
            task_id = step_id or f"task_{turn_id}_{name}_{len(self.tasks) + 1}"
            self.start_task(
                task_id=task_id,
                turn_id=turn_id,
                title=name,
                item_type=item_type,
                step_id=step_id,
                summary=f"{name} started",
                metadata=metadata,
            )
        elif etype == "item_completed":
            item_type = str(content.get("item_type", "tool_call")).strip() or "tool_call"
            name = str(content.get("name", "")).strip() or item_type
            status = str(content.get("status", "completed")).strip() or "completed"
            task_id = step_id or f"task_{turn_id}_{name}_{len(self.tasks)}"
            if task_id not in self.tasks:
                self.start_task(
                    task_id=task_id,
                    turn_id=turn_id,
                    title=name,
                    item_type=item_type,
                    step_id=step_id,
                    summary=f"{name} started",
                    metadata=metadata,
                )
            self.finish_task(
                task_id,
                status=status,
                output_summary=f"{name} {status}",
                metadata=metadata,
            )
        elif etype == "task_update":
            task_id = str(content.get("task_id", "")).strip() or step_id
            title = str(content.get("title", "")).strip() or str(content.get("command", "")).strip() or "task"
            status = str(content.get("status", "in_progress")).strip() or "in_progress"
            summary = str(content.get("summary", "")).strip()
            if task_id:
                if task_id not in self.tasks:
                    self.start_task(
                        task_id=task_id,
                        turn_id=turn_id,
                        title=title,
                        item_type="task",
                        step_id=step_id,
                        summary=summary or f"{title} started",
                        metadata=content if isinstance(content, dict) else metadata,
                    )
                if status in {"completed", "failed", "cancelled"}:
                    self.finish_task(
                        task_id,
                        status=status,
                        output_summary=summary,
                        error=str(content.get("error", "")).strip(),
                        metadata=content if isinstance(content, dict) else metadata,
                    )
                else:
                    self.start_task(
                        task_id=task_id,
                        turn_id=turn_id,
                        title=title,
                        item_type="task",
                        step_id=step_id,
                        summary=summary or f"{title} started",
                        metadata=content if isinstance(content, dict) else metadata,
                    )
        self.touch()

    def cancel_active_turn(self) -> bool:
        """Signal cancellation for the active turn.

        Returns True if a cancel was issued, False if no active turn.
        """
        if self.active_cancel is not None and not self.active_cancel.is_set():
            self.active_cancel.set()
            logger.info("session_turn_cancelled", extra={
                "session_id": self.session_id,
                "turn_id": self.active_turn_id,
            })
            return True
        return False

    def to_summary(self) -> dict:
        """Return lightweight session summary (no adata)."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": len(self.history),
            "has_adata": self.adata is not None,
            "active_turn_id": self.active_turn_id,
            "last_trace_id": self.trace_ids[-1] if self.trace_ids else "",
            "trace_count": len(self.trace_ids),
            "pending_approvals": len(self.get_pending_approvals()),
            "pending_questions": len(self.get_pending_questions()),
            "task_count": len(self.tasks),
            "loaded_tools": list(self.loaded_tools),
            "plan_mode": self.plan_mode,
            "worktree": dict(self.worktree),
        }


# ---------------------------------------------------------------------------
# Session manager (thread-safe singleton)
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages agent chat sessions with TTL and memory budget."""

    def __init__(self, max_sessions: int = MAX_SESSIONS,
                 ttl_seconds: int = SESSION_TTL_SECONDS):
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds

    # --- CRUD ---------------------------------------------------------------

    def create_session(self, session_id: str,
                       base_adata: Any = None) -> AgentSession:
        """Create a new session, evicting oldest if at capacity.

        Parameters
        ----------
        session_id : str
            Unique session identifier (typically from frontend).
        base_adata : Any, optional
            Base adata to share into the session (not copied until mutated).
        """
        with self._lock:
            # If session already exists, return it
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            # Evict expired first
            self._evict_expired_locked()

            # Evict oldest idle if still at capacity
            while len(self._sessions) >= self.max_sessions:
                self._evict_oldest_locked()

            # Copy base adata so sessions cannot mutate each other's data
            # via in-place operations (copy-on-create isolation).
            session_adata = base_adata.copy() if (
                base_adata is not None and hasattr(base_adata, 'copy')
            ) else base_adata
            session = AgentSession(
                session_id=session_id,
                adata=session_adata,
                adata_is_owned=True,  # owned from the start (copied)
            )
            self._sessions[session_id] = session
            logger.info("session_created", extra={
                "session_id": session_id,
                "total_sessions": len(self._sessions),
            })
            return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Retrieve session by ID, or None if not found / expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            # Check TTL
            if time.time() - session.last_active > self.ttl_seconds:
                self._remove_session_locked(session_id)
                return None
            session.touch()
            return session

    def get_or_create(self, session_id: str,
                      base_adata: Any = None) -> AgentSession:
        """Get existing session or create a new one."""
        session = self.get_session(session_id)
        if session is not None:
            return session
        return self.create_session(session_id, base_adata=base_adata)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and cancel any active turn."""
        with self._lock:
            return self._remove_session_locked(session_id)

    def list_sessions(self) -> list[dict]:
        """Return summaries of all active sessions."""
        with self._lock:
            self._evict_expired_locked()
            return [s.to_summary() for s in self._sessions.values()]

    # --- Adata management ---------------------------------------------------

    def get_session_adata(self, session_id: str,
                          fallback_adata: Any = None) -> Any:
        """Return the session's adata, falling back to global if needed.

        For copy-on-write: the first time a session needs to mutate adata,
        call ``commit_session_adata`` which stores the result.
        """
        session = self.get_session(session_id)
        if session is not None and session.adata is not None:
            return session.adata
        return fallback_adata

    def commit_session_adata(self, session_id: str, new_adata: Any) -> None:
        """Store mutated adata back into the session (copy-on-write commit)."""
        session = self.get_session(session_id)
        if session is not None:
            session.set_adata(new_adata)
            logger.info("session_adata_committed", extra={
                "session_id": session_id,
                "n_obs": getattr(new_adata, 'n_obs', None),
                "n_vars": getattr(new_adata, 'n_vars', None),
            })

    def register_approval(self, session_id: str, approval: ApprovalRequest) -> None:
        session = self.get_session(session_id)
        if session is not None:
            session.register_approval(approval)
            logger.info("session_approval_registered", extra={
                "session_id": session_id,
                "approval_id": approval.approval_id,
                "turn_id": approval.turn_id,
            })

    def register_question(self, session_id: str, question: QuestionRequest) -> None:
        session = self.get_session(session_id)
        if session is not None:
            session.register_question(question)
            logger.info("session_question_registered", extra={
                "session_id": session_id,
                "question_id": question.question_id,
                "turn_id": question.turn_id,
            })

    def list_pending_approvals(self, session_id: str) -> list[dict]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.get_pending_approvals()

    def list_pending_questions(self, session_id: str) -> list[dict]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.get_pending_questions()

    def resolve_approval(self, session_id: str, approval_id: str, decision: str) -> Optional[dict]:
        session = self.get_session(session_id)
        if session is None:
            return None
        resolved = session.resolve_approval(approval_id, decision)
        if resolved is not None:
            logger.info("session_approval_resolved", extra={
                "session_id": session_id,
                "approval_id": approval_id,
                "decision": decision,
            })
        return resolved

    def resolve_question(self, session_id: str, question_id: str, answer: str) -> Optional[dict]:
        session = self.get_session(session_id)
        if session is None:
            return None
        resolved = session.resolve_question(question_id, answer)
        if resolved is not None:
            logger.info("session_question_resolved", extra={
                "session_id": session_id,
                "question_id": question_id,
            })
        return resolved

    def apply_runtime_event(self, session_id: str, event: dict) -> None:
        session = self.get_session(session_id)
        if session is None:
            return
        session.apply_runtime_event(event)

    def get_runtime_state(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if session is None:
            return {
                "loaded_tools": [],
                "active_tool_name": "",
                "active_step_id": "",
                "last_tool_name": "",
                "plan_mode": "off",
                "worktree": {},
                "worktree_label": "",
                "tasks": [],
                "task_count": 0,
                "pending_questions": [],
                "pending_approvals": [],
            }
        return session.get_runtime_state()

    def list_tasks(self, session_id: str, limit: int = 20) -> list[dict]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.list_tasks(limit=limit)

    # --- Cancel support -----------------------------------------------------

    def cancel_turn(self, session_id: str) -> bool:
        """Cancel the active turn for a session.

        Returns True if cancellation was issued.
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        return session.cancel_active_turn()

    # --- History persistence ------------------------------------------------

    def save_history(self, session_id: str) -> Optional[str]:
        """Save session history to a JSONL file.

        Returns the file path, or None if persistence is disabled.
        """
        if not HISTORY_DIR:
            return None
        session = self.get_session(session_id)
        if session is None or not session.history:
            return None

        try:
            hist_dir = os.path.join(HISTORY_DIR, ".ovagent_web")
            os.makedirs(hist_dir, exist_ok=True)
            filename = f"session_{session_id}.jsonl"
            filepath = os.path.join(hist_dir, filename)
            with open(filepath, "w") as f:
                for msg in session.history:
                    f.write(json.dumps(msg.to_dict(), default=str) + "\n")
            logger.info("session_history_saved", extra={
                "session_id": session_id,
                "path": filepath,
                "message_count": len(session.history),
            })
            return filepath
        except Exception:
            logger.exception("session_history_save_failed", extra={
                "session_id": session_id,
            })
            return None

    def load_history(self, session_id: str) -> list[dict]:
        """Load session history from JSONL file (if persistence is enabled)."""
        if not HISTORY_DIR:
            return []
        filepath = os.path.join(HISTORY_DIR, ".ovagent_web",
                                f"session_{session_id}.jsonl")
        if not os.path.exists(filepath):
            return []
        try:
            messages = []
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
            return messages
        except Exception:
            logger.exception("session_history_load_failed", extra={
                "session_id": session_id,
            })
            return []

    # --- Internal eviction --------------------------------------------------

    def _evict_expired_locked(self):
        """Remove all sessions past TTL (must hold self._lock)."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.ttl_seconds
        ]
        for sid in expired:
            self._remove_session_locked(sid)

    def _evict_oldest_locked(self):
        """Remove the oldest idle session (must hold self._lock)."""
        if not self._sessions:
            return
        # Prefer evicting sessions without active turns
        candidates = [
            (sid, s) for sid, s in self._sessions.items()
            if s.active_turn_id is None
        ]
        if not candidates:
            # All sessions have active turns; evict oldest anyway
            candidates = list(self._sessions.items())

        oldest_sid = min(candidates, key=lambda x: x[1].last_active)[0]
        self._remove_session_locked(oldest_sid)

    def _remove_session_locked(self, session_id: str) -> bool:
        """Remove a session, cancelling any active turn (must hold self._lock)."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        # Cancel active turn if any
        if session.active_cancel is not None:
            session.active_cancel.set()
        # Persist history before removal
        if HISTORY_DIR and session.history:
            # Release lock briefly for I/O — safe because session is already
            # removed from the dict, so no concurrent access.
            threading.Thread(
                target=self._persist_history_bg,
                args=(session,),
                daemon=True,
            ).start()
        logger.info("session_removed", extra={
            "session_id": session_id,
            "total_sessions": len(self._sessions),
            "age_s": round(time.time() - session.created_at, 1),
        })
        return True

    @staticmethod
    def _persist_history_bg(session: AgentSession):
        """Background persist for evicted sessions."""
        try:
            hist_dir = os.path.join(HISTORY_DIR, ".ovagent_web")
            os.makedirs(hist_dir, exist_ok=True)
            filepath = os.path.join(hist_dir,
                                    f"session_{session.session_id}.jsonl")
            with open(filepath, "w") as f:
                for msg in session.history:
                    f.write(json.dumps(msg.to_dict(), default=str) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

session_manager = SessionManager()
