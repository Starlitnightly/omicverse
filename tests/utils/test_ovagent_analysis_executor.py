from contextlib import nullcontext
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for name in [
    "omicverse",
    "omicverse.utils",
    "omicverse.utils.ovagent",
    "omicverse.utils.ovagent.analysis_executor",
]:
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

ovagent_pkg = types.ModuleType("omicverse.utils.ovagent")
ovagent_pkg.__path__ = [str(PACKAGE_ROOT / "utils" / "ovagent")]
ovagent_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils.ovagent", loader=None, is_package=True
)
sys.modules["omicverse.utils.ovagent"] = ovagent_pkg
utils_pkg.ovagent = ovagent_pkg

analysis_executor_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.ovagent.analysis_executor",
    PACKAGE_ROOT / "utils" / "ovagent" / "analysis_executor.py",
)
analysis_executor_module = importlib.util.module_from_spec(analysis_executor_spec)
sys.modules["omicverse.utils.ovagent.analysis_executor"] = analysis_executor_module
assert analysis_executor_spec.loader is not None
analysis_executor_spec.loader.exec_module(analysis_executor_module)

AnalysisExecutor = analysis_executor_module.AnalysisExecutor


class _DummySecurityScanner:
    def scan(self, code: str):
        return []

    def format_report(self, violations):
        return ""

    def has_critical(self, violations):
        return False


class _DummyNotebookExecutor:
    def __init__(self):
        self.current_session = {
            "session_id": "session-1",
            "notebook_path": Path("/tmp/session.ipynb"),
        }
        self.session_prompt_count = 3

    def execute(self, code, adata):
        return adata


class _DummyAdata:
    def __init__(self):
        self.uns = {}


class _DummyCtx:
    def __init__(self):
        self._security_scanner = _DummySecurityScanner()
        self._security_config = type("Cfg", (), {"approval_mode": object()})()
        self.use_notebook_execution = True
        self._notebook_executor = _DummyNotebookExecutor()
        self.enable_filesystem_context = False
        self._filesystem_context = None

    def _temporary_api_keys(self):
        return nullcontext()


def test_notebook_execution_capture_stdout_returns_mapping():
    ctx = _DummyCtx()
    executor = AnalysisExecutor(ctx)
    adata = _DummyAdata()

    result = executor.execute_generated_code("x = 1", adata, capture_stdout=True)

    assert isinstance(result, dict)
    assert result["adata"] is adata
    assert result["stdout"] == ""
    assert adata.uns["_ovagent_session"]["session_id"] == "session-1"


def test_notebook_execution_without_stdout_returns_adata():
    ctx = _DummyCtx()
    executor = AnalysisExecutor(ctx)
    adata = _DummyAdata()

    result = executor.execute_generated_code("x = 1", adata, capture_stdout=False)

    assert result is adata
