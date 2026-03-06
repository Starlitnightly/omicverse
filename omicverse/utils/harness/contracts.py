"""
Shared harness contracts for OVAgent runtime, tracing, and web streaming.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


HARNESS_EVENT_TYPES = (
    "llm_chunk",
    "tool_call",
    "code",
    "result",
    "status",
    "error",
    "done",
    "usage",
    "item_started",
    "item_completed",
    "approval_request",
    "approval_resolved",
    "question_request",
    "question_resolved",
    "task_update",
    "heartbeat",
    "stream_end",
)
STREAM_EVENT_TYPES = HARNESS_EVENT_TYPES

HarnessEventType = Literal[
    "llm_chunk",
    "tool_call",
    "code",
    "result",
    "status",
    "error",
    "done",
    "usage",
    "item_started",
    "item_completed",
    "approval_request",
    "approval_resolved",
    "question_request",
    "question_resolved",
    "task_update",
    "heartbeat",
    "stream_end",
]


def make_trace_id() -> str:
    return "trace_" + uuid.uuid4().hex[:12]


def make_turn_id() -> str:
    return "turn_" + uuid.uuid4().hex[:12]


def make_step_id() -> str:
    return "step_" + uuid.uuid4().hex[:12]


def make_approval_id() -> str:
    return "approval_" + uuid.uuid4().hex[:12]


def hash_code_block(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def coerce_usage_payload(usage: Any) -> Optional[dict[str, Any]]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "__dict__"):
        return {
            k: v for k, v in usage.__dict__.items()
            if not k.startswith("_")
        }
    return {"value": repr(usage)}


@dataclass
class ArtifactRef:
    kind: str
    label: str = ""
    path: str = ""
    uri: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessEvent:
    type: HarnessEventType
    content: Any
    turn_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    step_id: str = ""
    category: str = ""
    latency_ms: Optional[float] = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "content": self.content,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "category": self.category,
            "timestamp": self.timestamp,
        }
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.artifact_refs:
            payload["artifact_refs"] = self.artifact_refs
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HarnessEvent":
        return cls(
            type=payload["type"],
            content=payload.get("content"),
            turn_id=payload.get("turn_id", ""),
            session_id=payload.get("session_id", ""),
            trace_id=payload.get("trace_id", ""),
            step_id=payload.get("step_id", ""),
            category=payload.get("category", ""),
            latency_ms=payload.get("latency_ms"),
            artifact_refs=list(payload.get("artifact_refs", [])),
            timestamp=payload.get("timestamp", time.time()),
        )


def build_stream_event(
    event_type: HarnessEventType,
    content: Any,
    *,
    turn_id: str = "",
    session_id: str = "",
    trace_id: str = "",
    step_id: str = "",
    category: str = "",
    latency_ms: Optional[float] = None,
    artifact_refs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return HarnessEvent(
        type=event_type,
        content=content,
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        step_id=step_id,
        category=category,
        latency_ms=latency_ms,
        artifact_refs=artifact_refs or [],
    ).to_dict()


@dataclass
class StepTrace:
    step_id: str
    step_type: str
    name: str = ""
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    output_summary: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    latency_ms: Optional[float] = None

    @property
    def tool_name(self) -> str:
        return self.name

    @property
    def tool_arguments(self) -> dict[str, Any]:
        return dict(self.data.get("arguments", {}))

    @property
    def code_hash(self) -> str:
        return str(self.data.get("code_hash", ""))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.data.get("metadata", {}))

    def finish(
        self,
        *,
        status: str,
        output_summary: str = "",
        error: str = "",
        artifact_refs: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status = status
        self.output_summary = output_summary
        self.error = error
        self.completed_at = time.time()
        self.latency_ms = round((self.completed_at - self.started_at) * 1000, 2)
        if metadata:
            merged = dict(self.data.get("metadata", {}))
            merged.update(metadata)
            self.data["metadata"] = merged
        if artifact_refs:
            self.data["artifact_refs"] = artifact_refs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunTrace:
    trace_id: str
    turn_id: str
    request: str
    model: str
    provider: str
    session_id: str = ""
    adata_shape: Optional[tuple[int, int]] = None
    history_size: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    status: str = "started"
    result_summary: str = ""
    usage: Optional[dict[str, Any]] = None
    usage_breakdown: dict[str, Any] = field(default_factory=dict)
    loaded_tools: list[str] = field(default_factory=list)
    plan_mode: bool = False
    worktree: dict[str, Any] = field(default_factory=dict)
    task_ids: list[str] = field(default_factory=list)
    approval_ids: list[str] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == "success"

    def finish(
        self,
        *,
        status: str,
        success: Optional[bool] = None,
        summary: str = "",
        result_summary: str = "",
        error: str = "",
        usage: Optional[dict[str, Any]] = None,
        usage_breakdown: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status = status
        self.result_summary = result_summary or summary or error
        self.usage = coerce_usage_payload(usage)
        if usage_breakdown is not None:
            self.usage_breakdown = usage_breakdown
        self.finished_at = time.time()
        if success is not None and success and self.status != "success":
            self.status = "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
            "request": self.request,
            "model": self.model,
            "provider": self.provider,
            "session_id": self.session_id,
            "adata_shape": self.adata_shape,
            "history_size": self.history_size,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "result_summary": self.result_summary,
            "usage": self.usage,
            "usage_breakdown": self.usage_breakdown,
            "loaded_tools": list(self.loaded_tools),
            "plan_mode": self.plan_mode,
            "worktree": dict(self.worktree),
            "task_ids": list(self.task_ids),
            "approval_ids": list(self.approval_ids),
            "question_ids": list(self.question_ids),
            "steps": [step.to_dict() for step in self.steps],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "events": list(self.events),
        }


@dataclass
class ScenarioSpec:
    name: str
    description: str = ""
    expected_event_types: list[str] = field(default_factory=list)
    expected_tool_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario_name: str
    trace_id: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    request_id: str
    turn_id: str
    trace_id: str
    session_id: str
    code_preview: str
    violations: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    status: str = "pending"
    decision: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    def resolve(self, decision: str) -> None:
        self.decision = decision
        self.status = "resolved"
        self.resolved_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessCapabilities:
    trace_replay: bool = True
    approval_requests: bool = True
    question_requests: bool = True
    cancel_turn: bool = True
    session_history: bool = True
    context_compaction: bool = True
    server_only_validation: bool = True
    task_tracking: bool = True
    plan_mode: bool = True
    deferred_tool_loading: bool = True
    worktrees: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
