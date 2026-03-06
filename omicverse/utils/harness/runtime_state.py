"""
Session-scoped runtime state for Claude-style OVAgent tools.
"""

from __future__ import annotations

import io
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from .tool_catalog import (
    CORE_TOOL_NAMES,
    ToolSearchResult,
    get_default_loaded_tool_names,
    get_default_tool_catalog,
)


TaskStatus = Literal["pending", "in_progress", "completed", "failed", "cancelled"]
QuestionStatus = Literal["pending", "resolved", "cancelled"]


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


def _coerce_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


@dataclass
class PlanModeState:
    enabled: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    entered_at: Optional[float] = None
    exited_at: Optional[float] = None

    def enter(self, *, reason: str = "", metadata: Optional[dict[str, Any]] = None) -> None:
        self.enabled = True
        self.reason = reason
        self.entered_at = _now()
        self.exited_at = None
        self.metadata = dict(metadata or {})

    def exit(self, *, reason: str = "", metadata: Optional[dict[str, Any]] = None) -> None:
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        if reason:
            merged["exit_reason"] = reason
        self.enabled = False
        self.reason = ""
        self.metadata = merged
        self.exited_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorktreeMetadata:
    worktree_id: str
    path: str
    repo_root: str = ""
    branch: str = ""
    base_branch: str = ""
    head_sha: str = ""
    created_at: float = field(default_factory=_now)
    entered_at: float = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["question"] = self.prompt
        payload["header"] = self.title
        return payload


@dataclass
class QuestionRecord:
    question_id: str
    session_id: str
    prompt: str
    turn_id: str = ""
    trace_id: str = ""
    title: str = ""
    options: list[str] = field(default_factory=list)
    allow_free_text: bool = True
    status: QuestionStatus = "pending"
    answer: Any = None
    created_at: float = field(default_factory=_now)
    resolved_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self, answer: Any) -> None:
        self.answer = answer
        self.status = "resolved"
        self.resolved_at = _now()

    def cancel(self, reason: str = "") -> None:
        self.status = "cancelled"
        self.resolved_at = _now()
        if reason:
            self.metadata = dict(self.metadata)
            self.metadata["cancel_reason"] = reason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskOutputChunk:
    stream: str
    text: str
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    title: str
    description: str = ""
    turn_id: str = ""
    trace_id: str = ""
    kind: str = ""
    tool_name: str = ""
    status: TaskStatus = "pending"
    summary: str = ""
    background: bool = False
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    output_chunks: list[TaskOutputChunk] = field(default_factory=list)
    total_output_chunks: int = 0
    total_output_bytes: int = 0
    output_truncated: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}

    def start(self) -> None:
        self.status = "in_progress"
        now = _now()
        self.started_at = now
        self.updated_at = now
        if "started_at" not in self.metadata:
            self.metadata["started_at"] = now

    def update(
        self,
        *,
        status: Optional[TaskStatus] = None,
        summary: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if status is not None:
            self.status = status
        if summary is not None:
            self.summary = summary
        if metadata:
            merged = dict(self.metadata)
            merged.update(metadata)
            self.metadata = merged
        self.updated_at = _now()

    def complete(self, *, summary: str = "", exit_code: Optional[int] = 0) -> None:
        self.status = "completed"
        self.summary = summary or self.summary
        self.exit_code = exit_code
        now = _now()
        if self.started_at is None:
            self.started_at = now
        self.finished_at = now
        self.updated_at = now

    def fail(self, *, error: str, summary: str = "", exit_code: Optional[int] = None) -> None:
        self.status = "failed"
        self.error = error
        self.summary = summary or self.summary or error
        self.exit_code = exit_code
        now = _now()
        if self.started_at is None:
            self.started_at = now
        self.finished_at = now
        self.updated_at = now

    def cancel(self, *, summary: str = "", exit_code: Optional[int] = None) -> None:
        self.status = "cancelled"
        self.summary = summary or self.summary
        self.exit_code = exit_code
        now = _now()
        self.finished_at = now
        self.updated_at = now

    def append_output(self, stream: str, text: str, *, max_chunks: int) -> None:
        payload = _coerce_text(text)
        if not payload:
            return
        self.output_chunks.append(TaskOutputChunk(stream=stream, text=payload))
        self.total_output_chunks += 1
        self.total_output_bytes += len(payload.encode("utf-8", errors="replace"))
        if len(self.output_chunks) > max_chunks:
            overflow = len(self.output_chunks) - max_chunks
            del self.output_chunks[:overflow]
            self.output_truncated = True
        self.updated_at = _now()

    def read_output(
        self,
        *,
        max_chunks: Optional[int] = None,
        include_stream: bool = False,
    ) -> str:
        chunks = self.output_chunks[-max_chunks:] if max_chunks else self.output_chunks
        if include_stream:
            return "".join(f"[{chunk.stream}] {chunk.text}" for chunk in chunks)
        return "".join(chunk.text for chunk in chunks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_chunks"] = [chunk.to_dict() for chunk in self.output_chunks]
        return payload


@dataclass
class SessionRuntimeState:
    session_id: str
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    loaded_tools: set[str] = field(default_factory=lambda: set(get_default_loaded_tool_names()))
    working_directory: str = field(default_factory=os.getcwd)
    plan_mode: PlanModeState = field(default_factory=PlanModeState)
    worktree: Optional[WorktreeMetadata] = None
    pending_questions: dict[str, QuestionRecord] = field(default_factory=dict)
    tasks: dict[str, TaskRecord] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _now()

    def load_tools(self, names: Iterable[str]) -> tuple[str, ...]:
        added: list[str] = []
        for name in names:
            if name and name not in self.loaded_tools:
                self.loaded_tools.add(name)
                added.append(name)
        if added:
            self.touch()
        return tuple(added)

    def unload_tools(self, names: Iterable[str]) -> tuple[str, ...]:
        removed: list[str] = []
        for name in names:
            if name in CORE_TOOL_NAMES:
                continue
            if name in self.loaded_tools:
                self.loaded_tools.remove(name)
                removed.append(name)
        if removed:
            self.touch()
        return tuple(removed)

    def set_worktree(self, metadata: WorktreeMetadata) -> WorktreeMetadata:
        self.worktree = metadata
        self.touch()
        return metadata

    def clear_worktree(self) -> None:
        self.worktree = None
        self.touch()

    def create_question(
        self,
        *,
        prompt: str = "",
        question: str = "",
        title: str = "",
        header: str = "",
        options: Optional[Iterable[str]] = None,
        allow_free_text: bool = True,
        turn_id: str = "",
        trace_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> QuestionRecord:
        prompt = prompt or question
        title = title or header
        if not prompt:
            raise ValueError("prompt/question is required")
        question = QuestionRecord(
            question_id=_make_id("question"),
            session_id=self.session_id,
            prompt=prompt,
            title=title,
            options=list(options or ()),
            allow_free_text=allow_free_text,
            turn_id=turn_id,
            trace_id=trace_id,
            metadata=dict(metadata or {}),
        )
        self.pending_questions[question.question_id] = question
        self.touch()
        return question

    def get_question(self, question_id: str) -> Optional[QuestionRecord]:
        return self.pending_questions.get(question_id)

    def resolve_question(self, question_id: str, answer: Any) -> Optional[QuestionRecord]:
        question = self.pending_questions.get(question_id)
        if question is None:
            return None
        question.resolve(answer)
        self.touch()
        return question

    def cancel_question(self, question_id: str, *, reason: str = "") -> Optional[QuestionRecord]:
        question = self.pending_questions.get(question_id)
        if question is None:
            return None
        question.cancel(reason=reason)
        self.touch()
        return question

    def list_questions(self, *, status: Optional[QuestionStatus] = None) -> list[QuestionRecord]:
        values = list(self.pending_questions.values())
        if status is None:
            return values
        return [question for question in values if question.status == status]

    def create_task(
        self,
        *,
        title: str,
        description: str = "",
        turn_id: str = "",
        trace_id: str = "",
        kind: str = "",
        tool_name: str = "",
        background: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TaskRecord:
        task = TaskRecord(
            task_id=_make_id("task"),
            session_id=self.session_id,
            title=title,
            description=description,
            turn_id=turn_id,
            trace_id=trace_id,
            kind=kind,
            tool_name=tool_name,
            background=background,
            metadata=dict(metadata or {}),
        )
        self.tasks[task.task_id] = task
        self.touch()
        return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self.tasks.get(task_id)

    def list_tasks(
        self,
        *,
        status: Optional[TaskStatus] = None,
        include_terminal: bool = True,
    ) -> list[TaskRecord]:
        values = list(self.tasks.values())
        if status is not None:
            values = [task for task in values if task.status == status]
        if not include_terminal:
            values = [task for task in values if not task.terminal]
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "loaded_tools": sorted(self.loaded_tools),
            "working_directory": self.working_directory,
            "plan_mode": self.plan_mode.to_dict(),
            "worktree": self.worktree.to_dict() if self.worktree else None,
            "pending_questions": [question.to_dict() for question in self.pending_questions.values()],
            "tasks": [task.to_dict() for task in self.tasks.values()],
        }


@dataclass
class BackgroundTaskHandle:
    session_id: str
    task_id: str
    process: subprocess.Popen[Any]
    stop_event: threading.Event = field(default_factory=threading.Event)
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    wait_thread: Optional[threading.Thread] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "pid": getattr(self.process, "pid", None),
            "running": self.process.poll() is None,
        }


class RuntimeStateManager:
    """Thread-safe manager for Claude-style per-session runtime state."""

    def __init__(
        self,
        *,
        output_root: Optional[Path] = None,
        max_output_chunks_per_task: int = 200,
    ) -> None:
        self.output_root = Path(output_root or (Path.home() / ".ovagent" / "runtime_state"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_output_chunks_per_task = max_output_chunks_per_task
        self._states: dict[str, SessionRuntimeState] = {}
        self._handles: dict[tuple[str, str], BackgroundTaskHandle] = {}
        self._lock = threading.RLock()
        self._catalog = get_default_tool_catalog()

    def get_session(self, session_id: str) -> Optional[SessionRuntimeState]:
        with self._lock:
            return self._states.get(session_id)

    def get_or_create_session(self, session_id: str) -> SessionRuntimeState:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = SessionRuntimeState(session_id=session_id)
                self._states[session_id] = state
            state.touch()
            return state

    def delete_session(self, session_id: str, *, cancel_background: bool = True) -> bool:
        with self._lock:
            state = self._states.pop(session_id, None)
            handles = [
                handle for key, handle in self._handles.items()
                if key[0] == session_id
            ]
        if state is None:
            return False
        if cancel_background:
            for handle in handles:
                self._stop_handle(handle, kill=True, wait_timeout=1.0)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [state.to_dict() for state in self._states.values()]

    def get_summary(self, session_id: str) -> dict[str, Any]:
        return self.get_or_create_session(session_id).to_dict()

    def get_loaded_tools(self, session_id: str) -> tuple[str, ...]:
        return tuple(sorted(self.get_or_create_session(session_id).loaded_tools))

    def load_tools(self, session_id: str, names: Iterable[str]) -> tuple[str, ...]:
        state = self.get_or_create_session(session_id)
        return state.load_tools(names)

    def set_working_directory(self, session_id: str, path: str) -> str:
        state = self.get_or_create_session(session_id)
        state.working_directory = path
        state.touch()
        return state.working_directory

    def get_working_directory(self, session_id: str) -> str:
        state = self.get_or_create_session(session_id)
        return state.working_directory

    def resolve_tool_query(
        self,
        session_id: str,
        query: str,
        *,
        max_results: int = 5,
        auto_load_search_matches: bool = False,
    ) -> ToolSearchResult:
        state = self.get_or_create_session(session_id)
        result = self._catalog.resolve_loading_query(
            query,
            loaded_tools=state.loaded_tools,
            max_results=max_results,
            auto_load_search_matches=auto_load_search_matches,
        )
        if result.selected_tools:
            state.load_tools(result.selected_tools)
        return result

    def enter_plan_mode(
        self,
        session_id: str,
        *,
        reason: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> PlanModeState:
        state = self.get_or_create_session(session_id)
        state.plan_mode.enter(reason=reason, metadata=metadata)
        state.touch()
        return state.plan_mode

    def exit_plan_mode(
        self,
        session_id: str,
        *,
        reason: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> PlanModeState:
        state = self.get_or_create_session(session_id)
        state.plan_mode.exit(reason=reason, metadata=metadata)
        state.touch()
        return state.plan_mode

    def set_worktree(
        self,
        session_id: str,
        *,
        path: str,
        repo_root: str = "",
        branch: str = "",
        base_branch: str = "",
        head_sha: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorktreeMetadata:
        state = self.get_or_create_session(session_id)
        worktree = WorktreeMetadata(
            worktree_id=_make_id("worktree"),
            path=path,
            repo_root=repo_root,
            branch=branch,
            base_branch=base_branch,
            head_sha=head_sha,
            metadata=dict(metadata or {}),
        )
        return state.set_worktree(worktree)

    def clear_worktree(self, session_id: str) -> None:
        state = self.get_or_create_session(session_id)
        state.clear_worktree()

    def create_question(self, session_id: str, **kwargs: Any) -> QuestionRecord:
        state = self.get_or_create_session(session_id)
        return state.create_question(**kwargs)

    def resolve_question(self, session_id: str, question_id: str, answer: Any) -> Optional[QuestionRecord]:
        state = self.get_or_create_session(session_id)
        return state.resolve_question(question_id, answer)

    def cancel_question(self, session_id: str, question_id: str, *, reason: str = "") -> Optional[QuestionRecord]:
        state = self.get_or_create_session(session_id)
        return state.cancel_question(question_id, reason=reason)

    def list_questions(self, session_id: str, *, status: Optional[QuestionStatus] = None) -> list[dict[str, Any]]:
        state = self.get_or_create_session(session_id)
        return [question.to_dict() for question in state.list_questions(status=status)]

    def create_task(self, session_id: str, **kwargs: Any) -> TaskRecord:
        state = self.get_or_create_session(session_id)
        status = kwargs.pop("status", None)
        task = state.create_task(**kwargs)
        if status:
            task.update(status=status)
        task.log_path = str(self._task_log_path(session_id, task.task_id))
        state.touch()
        return task

    def get_task(self, session_id: str, task_id: str) -> Optional[TaskRecord]:
        state = self.get_or_create_session(session_id)
        return state.get_task(task_id)

    def list_tasks(
        self,
        session_id: str,
        *,
        status: Optional[TaskStatus] = None,
        include_terminal: bool = True,
    ) -> list[dict[str, Any]]:
        state = self.get_or_create_session(session_id)
        return [
            task.to_dict()
            for task in state.list_tasks(status=status, include_terminal=include_terminal)
        ]

    def start_task(self, session_id: str, task_id: str) -> Optional[TaskRecord]:
        task = self.get_task(session_id, task_id)
        if task is None:
            return None
        task.start()
        self.get_or_create_session(session_id).touch()
        return task

    def update_task(
        self,
        session_id: str,
        task_id: str,
        *,
        status: Optional[TaskStatus] = None,
        summary: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[TaskRecord]:
        task = self.get_task(session_id, task_id)
        if task is None:
            return None
        task.update(status=status, summary=summary, metadata=metadata)
        self.get_or_create_session(session_id).touch()
        return task

    def complete_task(
        self,
        session_id: str,
        task_id: str,
        *,
        summary: str = "",
        exit_code: Optional[int] = 0,
    ) -> Optional[TaskRecord]:
        task = self.get_task(session_id, task_id)
        if task is None:
            return None
        task.complete(summary=summary, exit_code=exit_code)
        self.get_or_create_session(session_id).touch()
        self._handles.pop((session_id, task_id), None)
        return task

    def fail_task(
        self,
        session_id: str,
        task_id: str,
        *,
        error: str,
        summary: str = "",
        exit_code: Optional[int] = None,
    ) -> Optional[TaskRecord]:
        task = self.get_task(session_id, task_id)
        if task is None:
            return None
        task.fail(error=error, summary=summary, exit_code=exit_code)
        self.get_or_create_session(session_id).touch()
        self._handles.pop((session_id, task_id), None)
        return task

    def cancel_task(
        self,
        session_id: str,
        task_id: str,
        *,
        summary: str = "",
    ) -> Optional[TaskRecord]:
        handle = self._handles.get((session_id, task_id))
        if handle is not None:
            self._stop_handle(handle, kill=True, wait_timeout=1.0)
        task = self.get_task(session_id, task_id)
        if task is None:
            return None
        if not task.terminal:
            task.cancel(summary=summary or "Task cancelled")
        self.get_or_create_session(session_id).touch()
        return task

    def append_task_output(self, session_id: str, task_id: str, text: Any, *, stream: str = "stdout") -> Optional[TaskRecord]:
        task = self.get_task(session_id, task_id)
        if task is None:
            return None
        payload = _coerce_text(text)
        if not payload:
            return task
        task.append_output(stream, payload, max_chunks=self.max_output_chunks_per_task)
        self._write_task_output(session_id, task_id, payload, stream=stream)
        self.get_or_create_session(session_id).touch()
        return task

    def read_task_output(
        self,
        session_id: str,
        task_id: str,
        *,
        max_chunks: int = 50,
        include_stream: bool = False,
        from_log: bool = True,
        max_bytes: int = 32768,
    ) -> dict[str, Any]:
        task = self.get_task(session_id, task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")

        output = ""
        log_path = self._task_log_path(session_id, task_id)
        if from_log and log_path.exists():
            output = self._read_tail(log_path, max_bytes=max_bytes)
        if not output:
            output = task.read_output(max_chunks=max_chunks, include_stream=include_stream)
        return {
            "task_id": task.task_id,
            "status": task.status,
            "output": output,
            "log_path": task.log_path,
            "total_output_chunks": task.total_output_chunks,
            "total_output_bytes": task.total_output_bytes,
            "output_truncated": task.output_truncated,
        }

    def attach_background_process(
        self,
        session_id: str,
        task_id: str,
        process: subprocess.Popen[Any],
    ) -> BackgroundTaskHandle:
        task = self.get_task(session_id, task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        task.background = True
        task.metadata = dict(task.metadata)
        task.metadata["pid"] = getattr(process, "pid", None)
        if task.started_at is None:
            task.start()
        handle = BackgroundTaskHandle(session_id=session_id, task_id=task_id, process=process)

        stdout = getattr(process, "stdout", None)
        stderr = getattr(process, "stderr", None)
        if stdout is not None:
            handle.stdout_thread = threading.Thread(
                target=self._drain_stream,
                args=(handle, stdout, "stdout"),
                daemon=True,
            )
            handle.stdout_thread.start()
        if stderr is not None:
            handle.stderr_thread = threading.Thread(
                target=self._drain_stream,
                args=(handle, stderr, "stderr"),
                daemon=True,
            )
            handle.stderr_thread.start()
        handle.wait_thread = threading.Thread(
            target=self._wait_for_process,
            args=(handle,),
            daemon=True,
        )
        handle.wait_thread.start()
        self._handles[(session_id, task_id)] = handle
        self.get_or_create_session(session_id).touch()
        return handle

    def stop_task(
        self,
        session_id: str,
        task_id: str,
        *,
        kill: bool = False,
        wait_timeout: float = 5.0,
    ) -> Optional[TaskRecord]:
        handle = self._handles.get((session_id, task_id))
        if handle is not None:
            self._stop_handle(handle, kill=kill, wait_timeout=wait_timeout)
        return self.cancel_task(session_id, task_id, summary="Task stopped")

    def _task_log_path(self, session_id: str, task_id: str) -> Path:
        session_dir = self.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / f"{task_id}.log"

    def _write_task_output(self, session_id: str, task_id: str, text: str, *, stream: str) -> None:
        path = self._task_log_path(session_id, task_id)
        with path.open("a", encoding="utf-8") as handle:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            handle.write(f"[{timestamp}] [{stream}] {text}")
            if text and not text.endswith("\n"):
                handle.write("\n")

    def _read_tail(self, path: Path, *, max_bytes: int) -> str:
        if max_bytes <= 0:
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            data = handle.read()
        return data.decode("utf-8", errors="replace")

    def _drain_stream(self, handle: BackgroundTaskHandle, stream: Any, stream_name: str) -> None:
        reader = stream
        while not handle.stop_event.is_set():
            try:
                if hasattr(reader, "readline"):
                    chunk = reader.readline()
                elif isinstance(reader, io.IOBase):
                    chunk = reader.read(4096)
                else:
                    chunk = reader.read(4096)
            except Exception as exc:
                self.append_task_output(handle.session_id, handle.task_id, f"{exc}\n", stream="stderr")
                break
            if not chunk:
                break
            self.append_task_output(handle.session_id, handle.task_id, chunk, stream=stream_name)
        try:
            if hasattr(reader, "close"):
                reader.close()
        except Exception:
            pass

    def _wait_for_process(self, handle: BackgroundTaskHandle) -> None:
        try:
            return_code = handle.process.wait()
        except Exception as exc:
            self.fail_task(
                handle.session_id,
                handle.task_id,
                error=f"Background task wait failed: {exc}",
            )
            return

        task = self.get_task(handle.session_id, handle.task_id)
        if task is None:
            self._handles.pop((handle.session_id, handle.task_id), None)
            return
        if task.status == "cancelled":
            self._handles.pop((handle.session_id, handle.task_id), None)
            return
        if return_code == 0:
            self.complete_task(
                handle.session_id,
                handle.task_id,
                summary=task.summary or "Background task completed",
                exit_code=return_code,
            )
        else:
            self.fail_task(
                handle.session_id,
                handle.task_id,
                error=f"Background task exited with code {return_code}",
                exit_code=return_code,
            )

    def _stop_handle(self, handle: BackgroundTaskHandle, *, kill: bool, wait_timeout: float) -> None:
        handle.stop_event.set()
        process = handle.process
        if process.poll() is None:
            try:
                if kill:
                    process.kill()
                else:
                    process.terminate()
                process.wait(timeout=wait_timeout)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        self._handles.pop((handle.session_id, handle.task_id), None)


def create_runtime_state_manager(*, output_root: Optional[Path] = None) -> RuntimeStateManager:
    return RuntimeStateManager(output_root=output_root)


runtime_state = create_runtime_state_manager()


__all__ = [
    "BackgroundTaskHandle",
    "PlanModeState",
    "QuestionRecord",
    "QuestionStatus",
    "RuntimeStateManager",
    "SessionRuntimeState",
    "TaskOutputChunk",
    "TaskRecord",
    "TaskStatus",
    "WorktreeMetadata",
    "create_runtime_state_manager",
    "runtime_state",
]
