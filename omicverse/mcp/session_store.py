"""
In-memory session state management for MCP tool execution.

Manages ``adata_id``, ``artifact_id``, and ``instance_id`` references so that
AnnData objects and class instances never cross the MCP protocol boundary.

Supports session-scoped isolation: each handle belongs to a session, and
cross-session access is rejected with a structured ``SessionError``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Handle dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdataHandle:
    """Reference to a stored AnnData object."""

    adata_id: str
    obj: Any  # AnnData
    created_at: float
    last_accessed: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactHandle:
    """Reference to a generated file artifact."""

    artifact_id: str
    path: str
    content_type: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact_type: str = "file"  # file|image|table|json|plot|report|export
    source_tool: str = ""
    updated_at: float = 0.0
    session_id: str = ""


@dataclass
class InstanceHandle:
    """Reference to a managed class instance (Phase 2+).

    Instance handles are **memory-only** and cannot be persisted or
    restored across process restarts.
    """

    instance_id: str
    obj: Any
    class_name: str
    created_at: float
    wrapper_spec: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SessionError
# ---------------------------------------------------------------------------


@dataclass
class TraceRecord:
    """Record of a single tool call with timing and handle refs."""

    trace_id: str
    session_id: str
    tool_name: str
    tool_type: str  # "meta", "registry", "class"
    started_at: float
    finished_at: float
    duration_ms: float
    ok: bool
    error_code: Optional[str] = None
    handle_refs_in: List[str] = field(default_factory=list)
    handle_refs_out: List[str] = field(default_factory=list)


@dataclass
class EventRecord:
    """Structured lifecycle event."""

    event_id: str
    event_type: str  # adata_created, tool_called, etc.
    session_id: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class SessionError(KeyError):
    """Session-aware error with structured ``error_code``.

    Subclasses ``KeyError`` so that existing ``except KeyError`` blocks
    still catch it, while callers that need specificity can inspect
    ``error_code`` and ``details``.
    """

    def __init__(self, error_code: str, message: str, details: Optional[dict] = None):
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


# ---------------------------------------------------------------------------
# Runtime limits
# ---------------------------------------------------------------------------


@dataclass
class RuntimeLimits:
    """Configurable quota and TTL settings for a session.

    Hard limits raise ``SessionError("quota_exceeded")`` when exceeded.
    Ring buffer caps silently discard the oldest entries.
    TTL values of ``None`` mean no automatic expiration.
    """

    # Hard limits — raise quota_exceeded on create
    max_adata_per_session: int = 50
    max_artifacts_per_session: int = 200
    max_instances_per_session: int = 50
    # Ring buffer caps — keep newest N, discard oldest silently
    max_events_per_session: int = 10_000
    max_traces_per_session: int = 5_000
    # TTL — None = disabled (no auto-expiry)
    event_ttl_seconds: Optional[float] = None
    trace_ttl_seconds: Optional[float] = None
    artifact_ttl_seconds: Optional[float] = None
    session_ttl_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Unified in-memory store for session state.

    All handles (``adata_id``, ``artifact_id``, ``instance_id``) are scoped
    to a session identified by ``session_id``.  Cross-session access raises
    ``SessionError`` with ``error_code="cross_session_access"``.

    Parameters
    ----------
    session_id : str, optional
        Logical session identifier.  Defaults to ``"default"``.
    persist_dir : str, optional
        Directory for persisting adata.  Created lazily on first persist
        call if ``None``.

    Thread-safety note: single-threaded.  Add locking if concurrent
    access is needed later.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        persist_dir: Optional[str] = None,
        limits: Optional[RuntimeLimits] = None,
    ):
        self._session_id: str = session_id or "default"
        self._persist_dir: Optional[str] = persist_dir
        self._limits: RuntimeLimits = limits or RuntimeLimits()
        self._created_at: float = time.time()
        self._last_cleanup_at: Optional[float] = None

        # Nested dicts: session_id → handle_id → Handle
        self._adata: Dict[str, Dict[str, AdataHandle]] = {self._session_id: {}}
        self._artifacts: Dict[str, Dict[str, ArtifactHandle]] = {self._session_id: {}}
        self._instances: Dict[str, Dict[str, InstanceHandle]] = {self._session_id: {}}

        # Observability buffers (session-local)
        self._events: List[EventRecord] = []
        self._traces: List[TraceRecord] = []
        self._obs_metrics: Dict[str, Any] = {
            "persisted_adata_count": 0,
            "restored_adata_count": 0,
            "tool_calls_total": 0,
            "tool_calls_failed": 0,
            "tool_stats": {},
            "artifacts_registered_total": 0,
            "artifacts_deleted_total": 0,
            "artifact_cleanup_runs": 0,
            "quota_rejections_total": 0,
            "expired_handle_rejections_total": 0,
            "cleanup_runs_total": 0,
            "cleanup_deleted_handles_total": 0,
        }

    # -- Session properties --------------------------------------------------

    @property
    def session_id(self) -> str:
        """Return the current session ID."""
        return self._session_id

    @property
    def limits(self) -> RuntimeLimits:
        """Return the runtime limits configuration."""
        return self._limits

    # -- Internal helpers ----------------------------------------------------

    def _session_adata(self) -> Dict[str, AdataHandle]:
        return self._adata.setdefault(self._session_id, {})

    def _session_artifacts(self) -> Dict[str, ArtifactHandle]:
        return self._artifacts.setdefault(self._session_id, {})

    def _session_instances(self) -> Dict[str, InstanceHandle]:
        return self._instances.setdefault(self._session_id, {})

    def _find_handle_session(
        self, handle_id: str, store_dict: Dict[str, dict]
    ) -> Optional[str]:
        """Find which session owns *handle_id*, or ``None``."""
        for sid, handles in store_dict.items():
            if handle_id in handles:
                return sid
        return None

    # -- AnnData lifecycle ---------------------------------------------------

    def create_adata(self, obj: Any, metadata: Optional[dict] = None) -> str:
        """Store an AnnData object and return its ``adata_id``."""
        current = len(self._session_adata())
        if current >= self._limits.max_adata_per_session:
            self._obs_metrics["quota_rejections_total"] += 1
            try:
                self.record_event("quota_exceeded", {
                    "resource": "adata", "current": current,
                    "limit": self._limits.max_adata_per_session,
                })
            except Exception:
                pass
            raise SessionError(
                "quota_exceeded",
                f"adata quota exceeded ({self._limits.max_adata_per_session})",
                {"resource": "adata", "current": current,
                 "limit": self._limits.max_adata_per_session},
            )
        adata_id = f"adata_{uuid.uuid4().hex[:12]}"
        now = time.time()
        meta = metadata or {}

        # Auto-populate shape metadata
        if hasattr(obj, "shape"):
            meta.setdefault("shape", list(obj.shape))
        if hasattr(obj, "obs") and hasattr(obj.obs, "columns"):
            meta.setdefault("obs_columns", list(obj.obs.columns)[:20])

        self._session_adata()[adata_id] = AdataHandle(
            adata_id=adata_id,
            obj=obj,
            created_at=now,
            last_accessed=now,
            metadata=meta,
        )
        try:
            self.record_event("adata_created", {"adata_id": adata_id, "shape": meta.get("shape")})
        except Exception:
            pass
        return adata_id

    def get_adata(self, adata_id: str) -> Any:
        """Retrieve an AnnData object by ID.

        Raises
        ------
        SessionError
            If the handle belongs to a different session.
        KeyError
            If the handle is not found in any session.
        """
        handle = self._session_adata().get(adata_id)
        if handle is None:
            owner = self._find_handle_session(adata_id, self._adata)
            if owner is not None:
                raise SessionError(
                    "cross_session_access",
                    f"adata_id {adata_id!r} belongs to session {owner!r}, "
                    f"not current session {self._session_id!r}",
                    {
                        "adata_id": adata_id,
                        "owner_session": owner,
                        "current_session": self._session_id,
                    },
                )
            raise KeyError(f"Unknown adata_id: {adata_id!r}")
        handle.last_accessed = time.time()
        return handle.obj

    def update_adata(self, adata_id: str, obj: Any) -> None:
        """Replace the stored object for an existing ``adata_id``."""
        handle = self._session_adata().get(adata_id)
        if handle is None:
            owner = self._find_handle_session(adata_id, self._adata)
            if owner is not None:
                raise SessionError(
                    "cross_session_access",
                    f"adata_id {adata_id!r} belongs to session {owner!r}, "
                    f"not current session {self._session_id!r}",
                    {
                        "adata_id": adata_id,
                        "owner_session": owner,
                        "current_session": self._session_id,
                    },
                )
            raise KeyError(f"Unknown adata_id: {adata_id!r}")
        handle.obj = obj
        handle.last_accessed = time.time()
        if hasattr(obj, "shape"):
            handle.metadata["shape"] = list(obj.shape)

    def list_adata(self) -> List[dict]:
        """Return summaries of all active datasets in the current session."""
        result = []
        for h in self._session_adata().values():
            info = {
                "adata_id": h.adata_id,
                "created_at": h.created_at,
                "last_accessed": h.last_accessed,
                "metadata": h.metadata,
            }
            result.append(info)
        return result

    # -- Artifact lifecycle --------------------------------------------------

    def create_artifact(
        self,
        path: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
        artifact_type: str = "file",
        source_tool: str = "",
    ) -> str:
        """Register a generated file and return its ``artifact_id``."""
        current = len(self._session_artifacts())
        if current >= self._limits.max_artifacts_per_session:
            self._obs_metrics["quota_rejections_total"] += 1
            try:
                self.record_event("quota_exceeded", {
                    "resource": "artifact", "current": current,
                    "limit": self._limits.max_artifacts_per_session,
                })
            except Exception:
                pass
            raise SessionError(
                "quota_exceeded",
                f"artifact quota exceeded ({self._limits.max_artifacts_per_session})",
                {"resource": "artifact", "current": current,
                 "limit": self._limits.max_artifacts_per_session},
            )
        artifact_id = f"artifact_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self._session_artifacts()[artifact_id] = ArtifactHandle(
            artifact_id=artifact_id,
            path=path,
            content_type=content_type,
            created_at=now,
            metadata=metadata or {},
            artifact_type=artifact_type,
            source_tool=source_tool,
            updated_at=now,
            session_id=self._session_id,
        )
        try:
            self.record_event("artifact_registered", {
                "artifact_id": artifact_id, "path": path,
                "content_type": content_type, "artifact_type": artifact_type,
                "source_tool": source_tool,
            })
        except Exception:
            pass
        return artifact_id

    def get_artifact(self, artifact_id: str) -> ArtifactHandle:
        """Retrieve an artifact handle.

        Raises ``SessionError`` for cross-session access, ``KeyError`` if
        not found.
        """
        handle = self._session_artifacts().get(artifact_id)
        if handle is None:
            owner = self._find_handle_session(artifact_id, self._artifacts)
            if owner is not None:
                raise SessionError(
                    "cross_session_access",
                    f"artifact_id {artifact_id!r} belongs to session {owner!r}, "
                    f"not current session {self._session_id!r}",
                    {
                        "artifact_id": artifact_id,
                        "owner_session": owner,
                        "current_session": self._session_id,
                    },
                )
            raise KeyError(f"Unknown artifact_id: {artifact_id!r}")
        return handle

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        content_type: Optional[str] = None,
        source_tool: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Return artifacts in the current session, optionally filtered.

        Results are sorted most-recent first.
        """
        artifacts = list(self._session_artifacts().values())
        if artifact_type:
            artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
        if content_type:
            artifacts = [a for a in artifacts if a.content_type == content_type]
        if source_tool:
            artifacts = [a for a in artifacts if a.source_tool == source_tool]
        artifacts.sort(key=lambda a: a.created_at, reverse=True)
        if limit:
            artifacts = artifacts[:limit]
        return [
            {
                "artifact_id": a.artifact_id,
                "path": a.path,
                "content_type": a.content_type,
                "artifact_type": a.artifact_type,
                "source_tool": a.source_tool,
                "created_at": a.created_at,
                "updated_at": a.updated_at,
                "metadata": a.metadata,
            }
            for a in artifacts
        ]

    def describe_artifact(self, artifact_id: str) -> dict:
        """Return full artifact metadata including file status.

        Raises ``SessionError`` for cross-session access, ``KeyError``
        if not found.
        """
        handle = self.get_artifact(artifact_id)
        file_exists = os.path.isfile(handle.path)
        file_size = os.path.getsize(handle.path) if file_exists else -1
        return {
            "artifact_id": handle.artifact_id,
            "path": handle.path,
            "content_type": handle.content_type,
            "artifact_type": handle.artifact_type,
            "source_tool": handle.source_tool,
            "created_at": handle.created_at,
            "updated_at": handle.updated_at,
            "session_id": handle.session_id,
            "metadata": handle.metadata,
            "file_exists": file_exists,
            "file_size_bytes": file_size,
        }

    def delete_artifact(self, artifact_id: str, delete_file: bool = False) -> dict:
        """Delete an artifact handle and optionally the underlying file.

        File-not-found is graceful (not an error).
        """
        handle = self.get_artifact(artifact_id)
        file_path = handle.path
        deleted_file = False
        if delete_file:
            try:
                os.unlink(file_path)
                deleted_file = True
            except FileNotFoundError:
                pass
        del self._session_artifacts()[artifact_id]
        try:
            self.record_event("artifact_deleted", {
                "artifact_id": artifact_id, "path": file_path,
                "deleted_file": deleted_file,
            })
        except Exception:
            pass
        return {
            "artifact_id": artifact_id,
            "deleted_handle": True,
            "deleted_file": deleted_file,
            "file_path": file_path,
        }

    def cleanup_artifacts(
        self,
        artifact_type: Optional[str] = None,
        older_than_seconds: Optional[float] = None,
        delete_files: bool = False,
        dry_run: bool = True,
    ) -> dict:
        """Batch cleanup artifacts by filters.

        Default ``dry_run=True`` previews without deleting.
        """
        now = time.time()
        matched = []
        for a in list(self._session_artifacts().values()):
            if artifact_type and a.artifact_type != artifact_type:
                continue
            if older_than_seconds and (now - a.created_at) < older_than_seconds:
                continue
            matched.append(a)

        items = []
        deleted_count = 0
        for a in matched:
            item: Dict[str, Any] = {
                "artifact_id": a.artifact_id,
                "path": a.path,
                "artifact_type": a.artifact_type,
                "age_seconds": round(now - a.created_at, 1),
            }
            if not dry_run:
                file_deleted = False
                if delete_files:
                    try:
                        os.unlink(a.path)
                        file_deleted = True
                    except FileNotFoundError:
                        pass
                item["deleted_file"] = file_deleted
                del self._session_artifacts()[a.artifact_id]
                deleted_count += 1
            items.append(item)

        try:
            self.record_event("artifact_cleanup", {
                "matched": len(matched),
                "deleted": deleted_count,
                "dry_run": dry_run,
                "delete_files": delete_files,
            })
        except Exception:
            pass
        return {
            "matched": len(matched),
            "deleted": deleted_count,
            "dry_run": dry_run,
            "items": items,
        }

    def export_artifacts_manifest(self) -> dict:
        """Export all session artifacts as a JSON-serializable manifest."""
        all_artifacts = []
        for a in self._session_artifacts().values():
            all_artifacts.append({
                "artifact_id": a.artifact_id,
                "path": a.path,
                "content_type": a.content_type,
                "artifact_type": a.artifact_type,
                "source_tool": a.source_tool,
                "created_at": a.created_at,
                "updated_at": a.updated_at,
                "metadata": a.metadata,
                "file_exists": os.path.isfile(a.path),
            })
        manifest = {
            "session_id": self._session_id,
            "exported_at": time.time(),
            "artifact_count": len(all_artifacts),
            "artifacts": all_artifacts,
        }
        try:
            self.record_event("artifact_manifest_exported", {
                "artifact_count": len(all_artifacts),
            })
        except Exception:
            pass
        return manifest

    # -- Instance lifecycle --------------------------------------------------

    def create_instance(
        self,
        obj: Any,
        class_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a class instance and return its ``instance_id``.

        Instance handles are memory-only and cannot be persisted.
        """
        current = len(self._session_instances())
        if current >= self._limits.max_instances_per_session:
            self._obs_metrics["quota_rejections_total"] += 1
            try:
                self.record_event("quota_exceeded", {
                    "resource": "instance", "current": current,
                    "limit": self._limits.max_instances_per_session,
                })
            except Exception:
                pass
            raise SessionError(
                "quota_exceeded",
                f"instance quota exceeded ({self._limits.max_instances_per_session})",
                {"resource": "instance", "current": current,
                 "limit": self._limits.max_instances_per_session},
            )
        instance_id = f"inst_{uuid.uuid4().hex[:12]}"
        self._session_instances()[instance_id] = InstanceHandle(
            instance_id=instance_id,
            obj=obj,
            class_name=class_name,
            created_at=time.time(),
            metadata=metadata or {},
        )
        try:
            self.record_event("instance_created", {
                "instance_id": instance_id, "class_name": class_name,
            })
        except Exception:
            pass
        return instance_id

    def get_instance(self, instance_id: str) -> Any:
        """Retrieve a class instance by ID.

        Raises ``SessionError`` for cross-session access, ``KeyError`` if
        not found.
        """
        handle = self._session_instances().get(instance_id)
        if handle is None:
            owner = self._find_handle_session(instance_id, self._instances)
            if owner is not None:
                raise SessionError(
                    "cross_session_access",
                    f"instance_id {instance_id!r} belongs to session {owner!r}, "
                    f"not current session {self._session_id!r}",
                    {
                        "instance_id": instance_id,
                        "owner_session": owner,
                        "current_session": self._session_id,
                    },
                )
            raise KeyError(f"Unknown instance_id: {instance_id!r}")
        return handle.obj

    # -- Generic deletion ----------------------------------------------------

    def delete_handle(self, handle_id: str) -> None:
        """Remove any handle (adata, artifact, or instance) by ID.

        Only handles in the current session can be deleted.  Raises
        ``SessionError`` if the handle belongs to another session.
        """
        session_adata = self._session_adata()
        session_artifacts = self._session_artifacts()
        session_instances = self._session_instances()

        if handle_id in session_adata:
            del session_adata[handle_id]
            try:
                self.record_event("instance_destroyed", {"handle_id": handle_id, "type": "adata"})
            except Exception:
                pass
        elif handle_id in session_artifacts:
            del session_artifacts[handle_id]
            try:
                self.record_event("instance_destroyed", {"handle_id": handle_id, "type": "artifact"})
            except Exception:
                pass
        elif handle_id in session_instances:
            del session_instances[handle_id]
            try:
                self.record_event("instance_destroyed", {"handle_id": handle_id, "type": "instance"})
            except Exception:
                pass
        else:
            # Check if it belongs to another session
            for store_dict, label in [
                (self._adata, "adata"),
                (self._artifacts, "artifact"),
                (self._instances, "instance"),
            ]:
                owner = self._find_handle_session(handle_id, store_dict)
                if owner is not None:
                    raise SessionError(
                        "cross_session_access",
                        f"handle {handle_id!r} belongs to session {owner!r}, "
                        f"not current session {self._session_id!r}",
                        {
                            "handle_id": handle_id,
                            "owner_session": owner,
                            "current_session": self._session_id,
                        },
                    )
            raise KeyError(f"Unknown handle: {handle_id!r}")

    # -- Cleanup -------------------------------------------------------------

    def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        """Remove handles not accessed within *max_age_seconds*.

        Cleans up across ALL sessions (garbage collection is global).
        Returns the number of handles removed.
        """
        cutoff = time.time() - max_age_seconds
        removed = 0

        for session_handles in self._adata.values():
            expired = [
                k for k, h in session_handles.items() if h.last_accessed < cutoff
            ]
            for k in expired:
                del session_handles[k]
                removed += 1

        for session_handles in self._artifacts.values():
            expired = [
                k for k, h in session_handles.items() if h.created_at < cutoff
            ]
            for k in expired:
                del session_handles[k]
                removed += 1

        for session_handles in self._instances.values():
            expired = [
                k for k, h in session_handles.items() if h.created_at < cutoff
            ]
            for k in expired:
                del session_handles[k]
                removed += 1

        return removed

    def check_session_expired(self) -> bool:
        """Return ``True`` if the session TTL has been exceeded."""
        if self._limits.session_ttl_seconds is None:
            return False
        return (time.time() - self._created_at) > self._limits.session_ttl_seconds

    def cleanup_expired_events(self, dry_run: bool = True) -> dict:
        """Remove events older than ``event_ttl_seconds``.

        Returns stats dict.  Default ``dry_run=True`` previews only.
        """
        ttl = self._limits.event_ttl_seconds
        if ttl is None:
            return {"target": "events", "matched": 0, "deleted": 0, "dry_run": dry_run}
        cutoff = time.time() - ttl
        expired = [e for e in self._events if e.timestamp < cutoff]
        deleted = 0
        if not dry_run and expired:
            self._events = [e for e in self._events if e.timestamp >= cutoff]
            deleted = len(expired)
            self._obs_metrics["cleanup_deleted_handles_total"] += deleted
        return {"target": "events", "matched": len(expired), "deleted": deleted, "dry_run": dry_run}

    def cleanup_expired_traces(self, dry_run: bool = True) -> dict:
        """Remove traces older than ``trace_ttl_seconds``.

        Returns stats dict.  Default ``dry_run=True`` previews only.
        """
        ttl = self._limits.trace_ttl_seconds
        if ttl is None:
            return {"target": "traces", "matched": 0, "deleted": 0, "dry_run": dry_run}
        cutoff = time.time() - ttl
        expired = [t for t in self._traces if t.finished_at < cutoff]
        deleted = 0
        if not dry_run and expired:
            self._traces = [t for t in self._traces if t.finished_at >= cutoff]
            deleted = len(expired)
            self._obs_metrics["cleanup_deleted_handles_total"] += deleted
        return {"target": "traces", "matched": len(expired), "deleted": deleted, "dry_run": dry_run}

    def cleanup_runtime(
        self,
        target: str = "all",
        dry_run: bool = True,
        delete_files: bool = False,
    ) -> dict:
        """Unified runtime cleanup dispatcher.

        Parameters
        ----------
        target : str
            ``"events"``, ``"traces"``, ``"artifacts"``, or ``"all"``.
        dry_run : bool
            Preview without deleting (default ``True``).
        delete_files : bool
            For artifact cleanup, also delete underlying files (default ``False``).
        """
        results: Dict[str, dict] = {}
        total_deleted = 0

        if target in ("events", "all"):
            r = self.cleanup_expired_events(dry_run=dry_run)
            results["events"] = r
            total_deleted += r["deleted"]

        if target in ("traces", "all"):
            r = self.cleanup_expired_traces(dry_run=dry_run)
            results["traces"] = r
            total_deleted += r["deleted"]

        if target == "artifacts" or (target == "all" and self._limits.artifact_ttl_seconds is not None):
            r = self.cleanup_artifacts(
                older_than_seconds=self._limits.artifact_ttl_seconds,
                delete_files=delete_files,
                dry_run=dry_run,
            )
            results["artifacts"] = r
            total_deleted += r["deleted"]

        if not dry_run:
            self._obs_metrics["cleanup_runs_total"] += 1
            self._last_cleanup_at = time.time()

        try:
            self.record_event("runtime_cleanup", {
                "target": target, "dry_run": dry_run,
                "total_deleted": total_deleted,
            })
        except Exception:
            pass

        return {
            "target": target,
            "dry_run": dry_run,
            "total_deleted": total_deleted,
            "results": results,
        }

    # -- Session info --------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return counts of stored handles in the current session."""
        return {
            "session_id": self._session_id,
            "adata_count": len(self._session_adata()),
            "artifact_count": len(self._session_artifacts()),
            "instance_count": len(self._session_instances()),
        }

    def session_info(self) -> dict:
        """Return current session summary."""
        return {
            "session_id": self._session_id,
            "persist_dir": self._persist_dir,
            "handles": {
                "adata": list(self._session_adata().keys()),
                "artifacts": list(self._session_artifacts().keys()),
                "instances": list(self._session_instances().keys()),
            },
            "stats": self.stats,
        }

    def list_handles(self) -> List[dict]:
        """List all handles in the current session with metadata."""
        handles: List[dict] = []
        for h in self._session_adata().values():
            handles.append({
                "handle_id": h.adata_id,
                "type": "adata",
                "created_at": h.created_at,
                "last_accessed": h.last_accessed,
                "metadata": h.metadata,
            })
        for h in self._session_artifacts().values():
            handles.append({
                "handle_id": h.artifact_id,
                "type": "artifact",
                "created_at": h.created_at,
                "path": h.path,
                "content_type": h.content_type,
                "artifact_type": h.artifact_type,
                "source_tool": h.source_tool,
                "updated_at": h.updated_at,
                "metadata": h.metadata,
            })
        for h in self._session_instances().values():
            handles.append({
                "handle_id": h.instance_id,
                "type": "instance",
                "created_at": h.created_at,
                "class_name": h.class_name,
                "metadata": h.metadata,
            })
        return handles

    # -- Observability -------------------------------------------------------

    def record_event(self, event_type: str, details: Optional[dict] = None) -> str:
        """Record a structured event.  Returns ``event_id``."""
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        self._events.append(EventRecord(
            event_id=event_id,
            event_type=event_type,
            session_id=self._session_id,
            timestamp=time.time(),
            details=details or {},
        ))
        # Ring buffer: keep newest N events
        cap = self._limits.max_events_per_session
        if len(self._events) > cap:
            self._events = self._events[-cap:]
        if event_type == "adata_persisted":
            self._obs_metrics["persisted_adata_count"] += 1
        elif event_type == "adata_restored":
            self._obs_metrics["restored_adata_count"] += 1
        elif event_type == "artifact_registered":
            self._obs_metrics["artifacts_registered_total"] += 1
        elif event_type == "artifact_deleted":
            self._obs_metrics["artifacts_deleted_total"] += 1
        elif event_type == "artifact_cleanup":
            self._obs_metrics["artifact_cleanup_runs"] += 1
        elif event_type == "runtime_cleanup":
            self._obs_metrics["artifact_cleanup_runs"] += 1
        return event_id

    def record_trace(self, trace: TraceRecord) -> None:
        """Append a completed trace and update tool-level metrics."""
        self._traces.append(trace)
        # Ring buffer: keep newest N traces
        cap = self._limits.max_traces_per_session
        if len(self._traces) > cap:
            self._traces = self._traces[-cap:]
        self._obs_metrics["tool_calls_total"] += 1
        if not trace.ok:
            self._obs_metrics["tool_calls_failed"] += 1
        ts = self._obs_metrics["tool_stats"].setdefault(
            trace.tool_name, {"call_count": 0, "fail_count": 0, "last_called_at": 0.0},
        )
        ts["call_count"] += 1
        if not trace.ok:
            ts["fail_count"] += 1
        ts["last_called_at"] = trace.finished_at

    def list_events(
        self,
        limit: int = 50,
        event_type: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> List[dict]:
        """Return recent events (most-recent first), optionally filtered."""
        filtered = self._events
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if tool_name:
            filtered = [e for e in filtered if e.details.get("tool_name") == tool_name]
        result = filtered[-limit:][::-1] if limit else filtered[::-1]
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "session_id": e.session_id,
                "timestamp": e.timestamp,
                "details": e.details,
            }
            for e in result
        ]

    def list_traces(
        self,
        limit: int = 50,
        tool_name: Optional[str] = None,
        ok: Optional[bool] = None,
    ) -> List[dict]:
        """Return recent trace summaries (most-recent first)."""
        filtered = self._traces
        if tool_name:
            filtered = [t for t in filtered if t.tool_name == tool_name]
        if ok is not None:
            filtered = [t for t in filtered if t.ok == ok]
        result = filtered[-limit:][::-1] if limit else filtered[::-1]
        return [
            {
                "trace_id": t.trace_id,
                "tool_name": t.tool_name,
                "tool_type": t.tool_type,
                "started_at": t.started_at,
                "duration_ms": round(t.duration_ms, 2),
                "ok": t.ok,
                "error_code": t.error_code,
            }
            for t in result
        ]

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Return full trace details by ``trace_id``, or ``None``."""
        for t in reversed(self._traces):
            if t.trace_id == trace_id:
                return {
                    "trace_id": t.trace_id,
                    "session_id": t.session_id,
                    "tool_name": t.tool_name,
                    "tool_type": t.tool_type,
                    "started_at": t.started_at,
                    "finished_at": t.finished_at,
                    "duration_ms": round(t.duration_ms, 2),
                    "ok": t.ok,
                    "error_code": t.error_code,
                    "handle_refs_in": t.handle_refs_in,
                    "handle_refs_out": t.handle_refs_out,
                }
        return None

    def get_metrics(self, scope: str = "session") -> dict:
        """Return aggregated metrics.

        Parameters
        ----------
        scope : str
            ``"session"`` for handle counts + cumulative counters,
            ``"tools"`` for per-tool call/fail stats.
        """
        if scope == "tools":
            return {"tool_stats": dict(self._obs_metrics["tool_stats"])}
        return {
            **self.stats,
            "persisted_adata_count": self._obs_metrics["persisted_adata_count"],
            "restored_adata_count": self._obs_metrics["restored_adata_count"],
            "tool_calls_total": self._obs_metrics["tool_calls_total"],
            "tool_calls_failed": self._obs_metrics["tool_calls_failed"],
            "artifacts_registered_total": self._obs_metrics["artifacts_registered_total"],
            "artifacts_deleted_total": self._obs_metrics["artifacts_deleted_total"],
            "artifact_cleanup_runs": self._obs_metrics["artifact_cleanup_runs"],
            "quota_rejections_total": self._obs_metrics["quota_rejections_total"],
            "expired_handle_rejections_total": self._obs_metrics["expired_handle_rejections_total"],
            "cleanup_runs_total": self._obs_metrics["cleanup_runs_total"],
            "cleanup_deleted_handles_total": self._obs_metrics["cleanup_deleted_handles_total"],
        }

    # -- Persistence ---------------------------------------------------------

    def persist_adata(self, adata_id: str, path: Optional[str] = None) -> dict:
        """Save an AnnData dataset to disk as ``.h5ad`` with JSON sidecar.

        Parameters
        ----------
        adata_id : str
            Handle to persist (must belong to current session).
        path : str, optional
            Explicit file path.  If ``None``, uses
            ``{persist_dir}/{adata_id}.h5ad``.

        Returns
        -------
        dict
            ``{"adata_id": str, "path": str, "metadata_path": str}``

        Raises
        ------
        SessionError / KeyError
            If adata_id not found or belongs to another session.
        RuntimeError
            If write fails.
        """
        adata = self.get_adata(adata_id)  # validates session ownership
        handle = self._session_adata()[adata_id]

        if path is None:
            if self._persist_dir is None:
                self._persist_dir = tempfile.mkdtemp(prefix="ov_persist_")
            os.makedirs(self._persist_dir, exist_ok=True)
            path = os.path.join(self._persist_dir, f"{adata_id}.h5ad")

        # Write .h5ad
        write_fn = getattr(adata, "write_h5ad", None) or getattr(adata, "write", None)
        if write_fn is None:
            raise RuntimeError(
                f"Object for {adata_id} does not support write_h5ad() or write(). "
                f"Got type: {type(adata).__name__}"
            )
        try:
            write_fn(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to write {adata_id} to {path}: {exc}") from exc

        # Write sidecar JSON
        meta_path = _sidecar_path(path)
        sidecar = {
            "adata_id": adata_id,
            "session_id": self._session_id,
            "file_path": path,
            "content_type": "application/x-h5ad",
            "created_at": handle.created_at,
            "persisted_at": time.time(),
            "original_metadata": handle.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(sidecar, f, indent=2, default=str)

        try:
            self.record_event("adata_persisted", {"adata_id": adata_id, "path": path})
        except Exception:
            pass
        return {"adata_id": adata_id, "path": path, "metadata_path": meta_path}

    def restore_adata(self, path: str, adata_id: Optional[str] = None) -> str:
        """Restore an AnnData dataset from a ``.h5ad`` file.

        Parameters
        ----------
        path : str
            Path to ``.h5ad`` file.
        adata_id : str, optional
            If provided and exists in current session, update in-place.
            Otherwise generate a new handle.

        Returns
        -------
        str
            The ``adata_id`` of the restored dataset.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If ``anndata`` is not installed.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            import anndata as ad
        except ImportError:
            raise ImportError(
                "The 'anndata' package is required for restore_adata. "
                "Install it with: pip install anndata"
            )

        adata = ad.read_h5ad(path)

        # Load sidecar metadata if present
        meta_path = _sidecar_path(path)
        sidecar_meta: dict = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                sidecar_meta = json.load(f)

        metadata = dict(sidecar_meta.get("original_metadata", {}))
        metadata["restored_from"] = path
        metadata["restored_at"] = time.time()

        if adata_id and adata_id in self._session_adata():
            self.update_adata(adata_id, adata)
            handle = self._session_adata()[adata_id]
            handle.metadata.update(metadata)
            try:
                self.record_event("adata_restored", {"adata_id": adata_id, "path": path})
            except Exception:
                pass
            return adata_id
        else:
            new_id = self.create_adata(adata, metadata=metadata)
            try:
                self.record_event("adata_restored", {"adata_id": new_id, "path": path})
            except Exception:
                pass
            return new_id


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _sidecar_path(h5ad_path: str) -> str:
    """Derive the JSON sidecar path from an .h5ad path."""
    base = h5ad_path.rsplit(".", 1)[0] if "." in h5ad_path else h5ad_path
    return base + ".meta.json"
