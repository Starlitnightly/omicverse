"""Tests for SessionContextFacadeMixin extraction (task-012).

Validates that session/history, filesystem-context, tracing, and
service-wiring delegation methods have been moved out of OmicVerseAgent
into the mixin while preserving the public API.
"""

import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module bootstrap (same pattern as test_smart_agent.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]
}
for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

smart_agent_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.smart_agent", PACKAGE_ROOT / "utils" / "smart_agent.py"
)
smart_agent_module = importlib.util.module_from_spec(smart_agent_spec)
sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
assert smart_agent_spec.loader is not None
smart_agent_spec.loader.exec_module(smart_agent_module)

OmicVerseAgent = smart_agent_module.OmicVerseAgent

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module

from omicverse.utils.ovagent.session_facade import SessionContextFacadeMixin


_RUN_HARNESS_TESTS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}


# -----------------------------------------------------------------------
# AC-001.1: Methods are owned by the mixin, not the agent class body
# -----------------------------------------------------------------------


def test_agent_inherits_from_mixin():
    """OmicVerseAgent must be a subclass of SessionContextFacadeMixin."""
    assert issubclass(OmicVerseAgent, SessionContextFacadeMixin)


MIXIN_METHODS = [
    "_get_session_service",
    "_get_context_service",
    "_get_harness_session_id",
    "_get_runtime_session_id",
    "_refresh_runtime_working_directory",
    "_detect_repo_root",
    "_initialize_session_context_tracing",
    "_wire_session_context_services",
    "get_current_session_info",
    "restart_session",
    "get_session_history",
    "write_note",
    "search_context",
    "get_relevant_context",
    "save_plan",
    "update_plan_step",
    "get_workspace_summary",
    "get_context_stats",
]


@pytest.mark.parametrize("method_name", MIXIN_METHODS)
def test_method_defined_on_mixin(method_name):
    """Session/context/tracing methods must be defined on the mixin."""
    assert hasattr(SessionContextFacadeMixin, method_name)


def test_filesystem_context_property_on_mixin():
    """filesystem_context property must live on the mixin."""
    assert isinstance(
        SessionContextFacadeMixin.__dict__.get("filesystem_context"),
        property,
    )


DELEGATION_METHODS = [
    "_get_session_service",
    "_get_context_service",
    "_get_harness_session_id",
    "_get_runtime_session_id",
    "_refresh_runtime_working_directory",
    "_detect_repo_root",
    "get_current_session_info",
    "restart_session",
    "get_session_history",
    "write_note",
    "search_context",
    "get_relevant_context",
    "save_plan",
    "update_plan_step",
    "get_workspace_summary",
    "get_context_stats",
]


@pytest.mark.parametrize("method_name", DELEGATION_METHODS)
def test_method_not_redefined_on_agent(method_name):
    """Extracted methods must be inherited, not re-declared on OmicVerseAgent."""
    assert method_name not in OmicVerseAgent.__dict__, (
        f"{method_name} is still defined directly on OmicVerseAgent"
    )
    assert hasattr(OmicVerseAgent, method_name)


# -----------------------------------------------------------------------
# AC-001.2: Public methods and call signatures remain unchanged
# -----------------------------------------------------------------------


PUBLIC_API = [
    "get_current_session_info",
    "restart_session",
    "get_session_history",
    "write_note",
    "search_context",
    "get_relevant_context",
    "save_plan",
    "update_plan_step",
    "get_workspace_summary",
    "get_context_stats",
    "run",
    "run_async",
    "stream_async",
    "generate_code",
    "generate_code_async",
    "help_short",
]


@pytest.mark.parametrize("method_name", PUBLIC_API)
def test_public_api_accessible(method_name):
    """All public methods must still be accessible on OmicVerseAgent."""
    assert hasattr(OmicVerseAgent, method_name)


def test_filesystem_context_property_accessible():
    """filesystem_context property must be accessible on OmicVerseAgent."""
    assert isinstance(
        OmicVerseAgent.__dict__.get("filesystem_context", None)
        or getattr(type, "__dict__", {}).get("filesystem_context", None)
        or next(
            (
                cls.__dict__["filesystem_context"]
                for cls in OmicVerseAgent.__mro__
                if "filesystem_context" in cls.__dict__
            ),
            None,
        ),
        property,
    )


# -----------------------------------------------------------------------
# AC-001.3: Lazy service construction works correctly
# -----------------------------------------------------------------------


def test_lazy_session_service_creation():
    """SessionService is lazily created on __new__-based instances."""
    from omicverse.utils.ovagent.session_context import SessionService

    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    svc = agent._get_session_service()
    assert isinstance(svc, SessionService)
    # Idempotent
    assert agent._get_session_service() is svc


def test_lazy_context_service_creation():
    """ContextService is lazily created on __new__-based instances."""
    from omicverse.utils.ovagent.session_context import ContextService

    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    svc = agent._get_context_service()
    assert isinstance(svc, ContextService)
    assert agent._get_context_service() is svc


# -----------------------------------------------------------------------
# AC: Thread-safe lazy initialization — concurrent access atomicity
# -----------------------------------------------------------------------

def test_concurrent_session_service_same_instance():
    """Concurrent _get_session_service calls must all return the same instance."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from omicverse.utils.ovagent.session_context import SessionService

    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    barrier = threading.Barrier(4)

    def get_service():
        barrier.wait()
        return agent._get_session_service()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get_service) for _ in range(4)]
        results = [f.result() for f in as_completed(futures)]

    assert all(r is results[0] for r in results)
    assert isinstance(results[0], SessionService)


def test_concurrent_context_service_same_instance():
    """Concurrent _get_context_service calls must all return the same instance."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from omicverse.utils.ovagent.session_context import ContextService

    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    barrier = threading.Barrier(4)

    def get_service():
        barrier.wait()
        return agent._get_context_service()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get_service) for _ in range(4)]
        results = [f.result() for f in as_completed(futures)]

    assert all(r is results[0] for r in results)
    assert isinstance(results[0], ContextService)


def test_service_init_lock_exists_on_mixin():
    """The mixin must expose a threading lock for service initialization."""
    assert hasattr(SessionContextFacadeMixin, "_service_init_lock")
    lock = SessionContextFacadeMixin._service_init_lock
    # Verify lock semantics without isinstance(lock, threading.Lock),
    # which is not portable because threading.Lock is a factory function
    # rather than a type on some Python builds.
    assert callable(getattr(lock, "acquire", None))
    assert callable(getattr(lock, "release", None))


# -----------------------------------------------------------------------
# AC-001.5: No new runtime dependencies
# -----------------------------------------------------------------------


def test_no_new_external_dependencies():
    """The mixin module must only use stdlib and internal imports."""
    import omicverse.utils.ovagent.session_facade as sf_mod

    source = Path(sf_mod.__file__).read_text()
    allowed_prefixes = (
        "from __future__",
        "import logging",
        "import threading",
        "from contextlib",
        "from pathlib",
        "from typing",
        "from ..",
        "from .",
    )
    for line in source.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        # Skip TYPE_CHECKING-guarded imports
        if "TYPE_CHECKING" in stripped:
            continue
        assert any(stripped.startswith(p) for p in allowed_prefixes), (
            f"Unexpected import in session_facade.py: {stripped}"
        )
