"""Tests for runtime quota enforcement and ring buffers."""

import pytest

from omicverse.mcp.session_store import (
    SessionStore, SessionError, RuntimeLimits, TraceRecord,
)


class TestRuntimeLimitsDefaults:
    def test_default_limits(self):
        lim = RuntimeLimits()
        assert lim.max_adata_per_session == 50
        assert lim.max_artifacts_per_session == 200
        assert lim.max_instances_per_session == 50
        assert lim.max_events_per_session == 10_000
        assert lim.max_traces_per_session == 5_000
        assert lim.event_ttl_seconds is None
        assert lim.trace_ttl_seconds is None
        assert lim.artifact_ttl_seconds is None
        assert lim.session_ttl_seconds is None

    def test_custom_limits(self):
        lim = RuntimeLimits(max_adata_per_session=3, session_ttl_seconds=600)
        assert lim.max_adata_per_session == 3
        assert lim.session_ttl_seconds == 600
        # Unset fields keep defaults
        assert lim.max_artifacts_per_session == 200

    def test_store_exposes_limits_property(self):
        lim = RuntimeLimits(max_adata_per_session=5)
        store = SessionStore(limits=lim)
        assert store.limits.max_adata_per_session == 5

    def test_store_default_limits_when_none(self):
        store = SessionStore()
        assert store.limits.max_adata_per_session == 50


class _FakeAdata:
    """Minimal stand-in for AnnData."""
    shape = (100, 200)


class TestQuotaEnforcement:
    def test_adata_quota_allows_up_to_limit(self):
        store = SessionStore(limits=RuntimeLimits(max_adata_per_session=3))
        ids = [store.create_adata(_FakeAdata()) for _ in range(3)]
        assert len(ids) == 3

    def test_adata_quota_rejects_at_limit(self):
        store = SessionStore(limits=RuntimeLimits(max_adata_per_session=2))
        store.create_adata(_FakeAdata())
        store.create_adata(_FakeAdata())
        with pytest.raises(SessionError) as exc_info:
            store.create_adata(_FakeAdata())
        assert exc_info.value.error_code == "quota_exceeded"
        assert exc_info.value.details["resource"] == "adata"
        assert exc_info.value.details["limit"] == 2

    def test_artifact_quota_rejects_at_limit(self):
        store = SessionStore(limits=RuntimeLimits(max_artifacts_per_session=2))
        store.create_artifact("/a.txt")
        store.create_artifact("/b.txt")
        with pytest.raises(SessionError) as exc_info:
            store.create_artifact("/c.txt")
        assert exc_info.value.error_code == "quota_exceeded"
        assert exc_info.value.details["resource"] == "artifact"

    def test_instance_quota_rejects_at_limit(self):
        store = SessionStore(limits=RuntimeLimits(max_instances_per_session=1))
        store.create_instance(object(), "TestClass")
        with pytest.raises(SessionError) as exc_info:
            store.create_instance(object(), "TestClass")
        assert exc_info.value.error_code == "quota_exceeded"
        assert exc_info.value.details["resource"] == "instance"

    def test_quota_rejection_records_event(self):
        store = SessionStore(limits=RuntimeLimits(max_adata_per_session=1))
        store.create_adata(_FakeAdata())
        with pytest.raises(SessionError):
            store.create_adata(_FakeAdata())
        events = store.list_events(event_type="quota_exceeded")
        assert len(events) >= 1
        assert events[0]["details"]["resource"] == "adata"

    def test_quota_rejection_increments_metric(self):
        store = SessionStore(limits=RuntimeLimits(max_adata_per_session=1))
        store.create_adata(_FakeAdata())
        with pytest.raises(SessionError):
            store.create_adata(_FakeAdata())
        m = store.get_metrics()
        assert m["quota_rejections_total"] >= 1


class TestRingBuffer:
    def test_events_ring_buffer(self):
        store = SessionStore(limits=RuntimeLimits(max_events_per_session=10))
        for i in range(15):
            store.record_event("test_event", {"i": i})
        assert len(store._events) == 10
        # Oldest events (0-4) should be gone, newest (5-14) kept
        details = [e.details["i"] for e in store._events]
        assert details[0] == 5
        assert details[-1] == 14

    def test_traces_ring_buffer(self):
        store = SessionStore(limits=RuntimeLimits(max_traces_per_session=5))
        import time
        for i in range(8):
            store.record_trace(TraceRecord(
                trace_id=f"t{i}", session_id="default",
                tool_name="test", tool_type="meta",
                started_at=time.time(), finished_at=time.time(),
                duration_ms=1.0, ok=True,
            ))
        assert len(store._traces) == 5
        assert store._traces[0].trace_id == "t3"
        assert store._traces[-1].trace_id == "t7"
