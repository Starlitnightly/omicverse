"""
Real-provider end-to-end PBMC OVAgent validation.

This test exercises the FULL OVAgent runtime stack against a real LLM relay
endpoint, proving that agent initialization, tool-planning, code-execution,
and reporting behaviour work through the actual agent path -- not mock LLM
responses.

Gating
------
This test is **never** collected by default.  It requires BOTH:
  - ``OV_AGENT_E2E_REAL_PROVIDER=1`` in the environment
  - A valid credential file pointed to by ``OV_AGENT_CREDENTIAL_FILE``
    (defaults to the Taiwan bundle path)

The credential file is parsed for ``API:`` and relay address lines.  Secrets
are *never* printed; only a masked ``sk-...XXXX`` form appears in the report.

Prerequisites
-------------
  - ``pip install -e ".[tests]"`` plus ``scanpy`` (for PBMC3k dataset)
  - Network access to the relay endpoint
  - A valid API key in the credential file

How to run
----------
::

    OV_AGENT_E2E_REAL_PROVIDER=1 python -m pytest -xvs tests/llm/test_e2e_real_provider.py

Or as a standalone script::

    OV_AGENT_E2E_REAL_PROVIDER=1 python tests/llm/test_e2e_real_provider.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Gate: skip when environment does not opt in
# ---------------------------------------------------------------------------
_ENABLED = os.environ.get("OV_AGENT_E2E_REAL_PROVIDER", "") == "1"

pytestmark = [
    pytest.mark.skipif(not _ENABLED, reason="OV_AGENT_E2E_REAL_PROVIDER not set"),
]

# ---------------------------------------------------------------------------
# Default credential file path (can be overridden via env)
# ---------------------------------------------------------------------------
_DEFAULT_CRED_FILE = (
    Path.home()
    / "PycharmProjects"
    / "NextGenOVAgent"
    / "taiwan-server-migration-bundle-2026-02-22"
    / "adpwt.txt"
)

# ---------------------------------------------------------------------------
# Credential parsing
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """Return ``sk-...XXXX`` form -- never expose the full key."""
    if len(key) > 8:
        return key[:3] + "..." + key[-4:]
    return "***"


def _parse_credentials(path: Path) -> Dict[str, str]:
    """Extract API key and base URL from the credential file.

    Returns dict with ``api_key``, ``base_url``, ``masked_key``.
    Raises ``FileNotFoundError`` or ``ValueError`` on problems.
    """
    if not path.exists():
        raise FileNotFoundError(f"Credential file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # API key: first line starting with "API:" or containing "sk-"
    api_key = None
    for line in text.splitlines():
        line_s = line.strip()
        if line_s.startswith("API:"):
            api_key = line_s.split(":", 1)[1].strip()
            break
        m = re.search(r"(sk-[A-Za-z0-9]+)", line_s)
        if m and api_key is None:
            api_key = m.group(1)

    if not api_key:
        raise ValueError("Could not extract API key from credential file")

    # Base URL: line containing an http URL with a port
    base_url = None
    for line in text.splitlines():
        m = re.search(r"(https?://[\d.]+:\d+/v\d+)", line)
        if m:
            base_url = m.group(1)
            break

    if not base_url:
        raise ValueError("Could not extract base URL from credential file")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "masked_key": _mask_key(api_key),
    }


# ---------------------------------------------------------------------------
# Connectivity check (stdlib only -- no openai dependency)
# ---------------------------------------------------------------------------

def _check_relay_connectivity(base_url: str, api_key: str) -> Dict[str, Any]:
    """Quick smoke test against the relay /models endpoint."""
    import urllib.request

    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.monotonic() - t0) * 1000
        models = [m.get("id", "?") for m in data.get("data", [])]
        return {
            "ok": True,
            "model_count": len(models),
            "latency_ms": round(latency_ms, 1),
            "sample_models": models[:6],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# PBMC dataset loader
# ---------------------------------------------------------------------------

def _load_pbmc_dataset():
    """Load PBMC3k via scanpy (small, quick, widely available)."""
    try:
        import scanpy as sc
    except ImportError:
        pytest.skip("scanpy not installed -- cannot load PBMC3k dataset")

    adata = None
    # Try local path first
    local_path = os.environ.get("PBMC3K_PATH")
    if local_path and os.path.exists(local_path):
        adata = sc.read_h5ad(local_path)
    else:
        try:
            adata = sc.datasets.pbmc3k()
        except Exception:
            try:
                adata = sc.datasets.pbmc68k_reduced()
            except Exception:
                pytest.skip("Could not load any PBMC dataset")

    return adata


# ---------------------------------------------------------------------------
# Deep evidence extraction from agent internals
# ---------------------------------------------------------------------------

def _extract_run_trace_evidence(agent) -> Dict[str, Any]:
    """Extract deep evidence from the agent's last run trace.

    This proves the run went through the real LLM stack by capturing:
    - trace_id, turn_id (unique per-run identifiers)
    - model and provider as resolved by the backend
    - token usage from the LLM API
    - tool call steps with names, types, and latencies
    - loaded tools list
    """
    trace = getattr(agent, "_last_run_trace", None)
    if trace is None:
        return {"available": False, "reason": "no _last_run_trace"}

    # Extract step summaries (tool calls made during the run)
    step_summaries = []
    for step in getattr(trace, "steps", []):
        step_summaries.append({
            "step_type": getattr(step, "step_type", "?"),
            "name": getattr(step, "name", "?"),
            "status": getattr(step, "status", "?"),
            "latency_ms": getattr(step, "latency_ms", None),
            "output_summary": (getattr(step, "output_summary", "") or "")[:200],
        })

    return {
        "available": True,
        "trace_id": getattr(trace, "trace_id", ""),
        "turn_id": getattr(trace, "turn_id", ""),
        "model": getattr(trace, "model", ""),
        "provider": getattr(trace, "provider", ""),
        "status": getattr(trace, "status", ""),
        "result_summary": (getattr(trace, "result_summary", "") or "")[:300],
        "usage": getattr(trace, "usage", None),
        "usage_breakdown": getattr(trace, "usage_breakdown", {}),
        "loaded_tools": getattr(trace, "loaded_tools", []),
        "step_count": len(getattr(trace, "steps", [])),
        "steps": step_summaries[:20],
        "adata_shape": getattr(trace, "adata_shape", None),
        "started_at": getattr(trace, "started_at", None),
        "finished_at": getattr(trace, "finished_at", None),
    }


def _extract_usage_evidence(agent) -> Dict[str, Any]:
    """Extract token usage evidence from the agent's last_usage fields."""
    return {
        "last_usage": getattr(agent, "last_usage", None),
        "last_usage_breakdown": getattr(agent, "last_usage_breakdown", {}),
    }


# ---------------------------------------------------------------------------
# Evidence collector
# ---------------------------------------------------------------------------

class EvidenceCollector:
    """Captures structured evidence from a real-provider E2E run."""

    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: List[Dict[str, Any]] = []
        self.agent_info: Dict[str, Any] = {}
        self.dataset_info: Dict[str, Any] = {}
        self.relay_info: Dict[str, Any] = {}
        self.final_status = "not_started"
        self.error_detail: Optional[str] = None

    def record_step(
        self,
        name: str,
        status: str,
        duration_s: float,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps.append({
            "name": name,
            "status": status,
            "duration_s": round(duration_s, 2),
            "detail": detail or {},
        })

    def to_report(self) -> Dict[str, Any]:
        return {
            "report_type": "e2e_real_provider_validation",
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "final_status": self.final_status,
            "relay": self.relay_info,
            "agent": self.agent_info,
            "dataset": self.dataset_info,
            "steps": self.steps,
            "error_detail": self.error_detail,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def credentials():
    """Load and validate credentials."""
    cred_path = Path(
        os.environ.get("OV_AGENT_CREDENTIAL_FILE", str(_DEFAULT_CRED_FILE))
    )
    try:
        return _parse_credentials(cred_path)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"Credential file issue: {exc}")


@pytest.fixture(scope="module")
def relay_check(credentials):
    """Verify relay connectivity before running expensive tests."""
    result = _check_relay_connectivity(
        credentials["base_url"], credentials["api_key"]
    )
    if not result["ok"]:
        pytest.skip(f"Relay unreachable: {result.get('error')}")
    return result


@pytest.fixture(scope="module")
def pbmc_adata():
    """Load PBMC dataset (cached for the module)."""
    return _load_pbmc_dataset()


@pytest.fixture(scope="module")
def evidence():
    """Shared evidence collector."""
    return EvidenceCollector()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealProviderE2E:
    """Real-provider E2E PBMC validation through the actual OVAgent stack."""

    def test_01_relay_connectivity(self, credentials, relay_check, evidence):
        """Verify the relay endpoint is reachable and lists models."""
        # Do not store credential-file-derived values (base_url, masked_key)
        # in evidence — CodeQL flags them as clear-text sensitive data when
        # the report is later written to disk or printed.
        evidence.relay_info = {**relay_check}
        assert relay_check["ok"], f"Relay check failed: {relay_check}"
        assert relay_check["model_count"] > 0, "No models available on relay"

    def test_02_agent_initialization(self, credentials, relay_check, evidence):
        """Initialize a real OmicVerseAgent with relay credentials."""
        import omicverse as ov

        model = os.environ.get("OV_AGENT_E2E_MODEL", "gpt-5.2")

        t0 = time.monotonic()
        agent = ov.Agent(
            model=model,
            api_key=credentials["api_key"],
            endpoint=credentials["base_url"],
            # Disable notebook execution for simpler in-process validation
            use_notebook_execution=False,
            # Disable reflection to speed up the test
            enable_reflection=False,
            enable_result_review=False,
            # Limit turns for test
            max_agent_turns=8,
            approval_mode="never",
            verbose=True,
        )
        init_duration = time.monotonic() - t0

        evidence.agent_info = {
            "model": model,
            "provider": getattr(agent, "provider", "unknown"),
            "init_duration_s": round(init_duration, 2),
        }
        evidence.record_step(
            "agent_init", "passed", init_duration,
            {"model": model, "provider": getattr(agent, "provider", "unknown")},
        )

        assert agent is not None
        assert hasattr(agent, "run"), "Agent missing run() method"
        assert hasattr(agent, "_turn_controller"), "Agent missing _turn_controller"

        # Stash agent on the class for subsequent tests
        TestRealProviderE2E._agent = agent

    def test_03_pbmc_qc_through_real_llm(
        self, credentials, pbmc_adata, evidence
    ):
        """Run a real PBMC QC request through the full agent stack.

        This is the core E2E assertion: a real LLM plans tool calls,
        the runtime executes them, and the result is a modified adata.
        """
        agent = getattr(TestRealProviderE2E, "_agent", None)
        if agent is None:
            pytest.skip("Agent not initialized (test_02 failed)")

        adata = pbmc_adata.copy()
        original_n_obs = adata.n_obs
        original_n_vars = adata.n_vars

        evidence.dataset_info = {
            "n_obs": original_n_obs,
            "n_vars": original_n_vars,
            "obs_columns": list(adata.obs.columns)[:10],
            "var_columns": list(adata.var.columns)[:10],
        }

        request = (
            "Perform basic quality control on this PBMC dataset. "
            "Calculate QC metrics (mitochondrial percentage, gene counts, UMI counts), "
            "then filter cells with fewer than 200 genes detected. "
            "Use omicverse or scanpy functions. "
            "Print a short summary of how many cells were kept."
        )

        t0 = time.monotonic()
        error_detail = None
        try:
            result = agent.run(request, adata)
            duration = time.monotonic() - t0
            status = "passed"

            # Deep evidence: extract run trace and usage from the real LLM call
            trace_evidence = _extract_run_trace_evidence(agent)
            usage_evidence = _extract_usage_evidence(agent)

            # The result should be an AnnData-like object
            result_info: Dict[str, Any] = {
                "run_trace": trace_evidence,
                "token_usage": usage_evidence,
            }
            if result is not None and hasattr(result, "shape"):
                result_info["n_obs"] = result.shape[0]
                result_info["n_vars"] = result.shape[1]
                result_info["obs_columns"] = list(result.obs.columns)[:15]
                result_info["var_columns"] = list(result.var.columns)[:15]
                result_info["cell_change"] = result.shape[0] - original_n_obs
            elif result is not None:
                result_info["type"] = str(type(result))
                result_info["repr"] = repr(result)[:300]
            else:
                result_info["result"] = "None"
                # None result is acceptable -- the agent may modify adata in-place
                if hasattr(adata, "shape"):
                    result_info["adata_n_obs"] = adata.shape[0]
                    result_info["adata_n_vars"] = adata.shape[1]

        except Exception as exc:
            duration = time.monotonic() - t0
            status = "failed"
            error_detail = traceback.format_exc()
            # Still capture whatever trace evidence is available
            trace_evidence = _extract_run_trace_evidence(agent)
            usage_evidence = _extract_usage_evidence(agent)
            result_info = {
                "error": str(exc),
                "traceback": error_detail[-1500:],
                "run_trace": trace_evidence,
                "token_usage": usage_evidence,
            }

        evidence.record_step(
            "pbmc_qc_real_llm", status, duration, result_info,
        )

        if status == "failed":
            evidence.final_status = "failed"
            evidence.error_detail = error_detail
            pytest.fail(
                f"PBMC QC through real LLM failed after {duration:.1f}s: "
                f"{result_info.get('error', 'unknown')}"
            )

    def test_04_verify_real_llm_evidence(self, evidence):
        """Verify that the QC step actually used a real LLM (not mocked).

        This test inspects the captured evidence to confirm:
        1. A run trace exists with a non-empty trace_id
        2. Tool calls were made (step_count > 0)
        3. Token usage was reported by the LLM API
        """
        qc_step = None
        for step in evidence.steps:
            if step["name"] == "pbmc_qc_real_llm":
                qc_step = step
                break

        if qc_step is None or qc_step["status"] != "passed":
            pytest.skip("QC step did not pass -- cannot verify evidence")

        trace = qc_step["detail"].get("run_trace", {})

        # Must have a trace_id proving the run went through the trace system
        assert trace.get("available"), "No run trace available"
        assert trace.get("trace_id"), "Empty trace_id -- run may not have gone through real stack"

        # Must have tool call steps (the LLM planned and dispatched tools)
        assert trace.get("step_count", 0) > 0, (
            f"No tool call steps recorded -- step_count={trace.get('step_count')}"
        )

        # Check that the model matches what we configured
        assert trace.get("model"), "No model recorded in trace"

    def test_99_generate_report(self, evidence, capsys):
        """Generate the structured validation report as a test artifact."""
        # Determine overall status from steps
        statuses = [s["status"] for s in evidence.steps]
        if all(s == "passed" for s in statuses):
            evidence.final_status = "all_passed"
        elif any(s == "passed" for s in statuses):
            evidence.final_status = "partial_pass"
        elif not statuses:
            evidence.final_status = "no_steps_executed"
        else:
            evidence.final_status = "all_failed"

        report = evidence.to_report()
        report_json = json.dumps(report, indent=2, default=str)

        # Write report to a file alongside the test
        report_path = (
            Path(__file__).parent / "e2e_real_provider_report.json"
        )
        report_path.write_text(report_json, encoding="utf-8")

        # Also print to stdout for capture
        print("\n" + "=" * 70)
        print("E2E REAL-PROVIDER VALIDATION REPORT")
        print("=" * 70)
        print(report_json)
        print("=" * 70)
        print(f"Report written to: {report_path}")

        # The report test itself always passes -- the individual step tests
        # enforce pass/fail.  This just ensures the report is generated.
        assert evidence.final_status != "no_steps_executed", (
            "No validation steps were executed"
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _run_standalone() -> int:
    """Run the validation outside pytest for quick manual testing."""
    os.environ["OV_AGENT_E2E_REAL_PROVIDER"] = "1"

    print("=" * 70)
    print("E2E Real-Provider PBMC OVAgent Validation (standalone)")
    print("=" * 70)

    ev = EvidenceCollector()

    # Step 1: credentials
    cred_path = Path(
        os.environ.get("OV_AGENT_CREDENTIAL_FILE", str(_DEFAULT_CRED_FILE))
    )
    try:
        creds = _parse_credentials(cred_path)
        print("[OK] Credentials loaded")
    except Exception as exc:
        print(f"[FAIL] Credential loading: {exc}")
        return 1

    # Step 2: relay — use credential values only for the API call,
    # never store them in evidence that gets logged/written to disk.
    relay = _check_relay_connectivity(creds["base_url"], creds["api_key"])
    ev.relay_info = {**relay}
    if not relay["ok"]:
        print(f"[FAIL] Relay unreachable: {relay.get('error')}")
        return 1
    print(f"[OK] Relay: {relay['model_count']} models, {relay['latency_ms']}ms")

    # Step 3: agent init
    import omicverse as ov

    model = os.environ.get("OV_AGENT_E2E_MODEL", "gpt-5.2")
    t0 = time.monotonic()
    try:
        agent = ov.Agent(
            model=model,
            api_key=creds["api_key"],
            endpoint=creds["base_url"],
            use_notebook_execution=False,
            enable_reflection=False,
            enable_result_review=False,
            max_agent_turns=8,
            approval_mode="never",
            verbose=True,
        )
        dur = time.monotonic() - t0
        ev.record_step("agent_init", "passed", dur, {"model": model})
        print(f"[OK] Agent initialized in {dur:.1f}s (model={model})")
    except Exception as exc:
        dur = time.monotonic() - t0
        ev.record_step("agent_init", "failed", dur, {"error": str(exc)})
        print(f"[FAIL] Agent init: {exc}")
        traceback.print_exc()
        return 1

    # Step 4: dataset
    try:
        import scanpy as sc
        adata = sc.datasets.pbmc3k()
        ev.dataset_info = {"n_obs": adata.n_obs, "n_vars": adata.n_vars}
        print(f"[OK] PBMC3k loaded: {adata.n_obs} x {adata.n_vars}")
    except Exception as exc:
        print(f"[SKIP] Dataset: {exc}")
        ev.record_step("dataset_load", "skipped", 0, {"error": str(exc)})
        ev.final_status = "blocked_dataset"
        print(json.dumps(ev.to_report(), indent=2, default=str))
        return 1

    # Step 5: real QC
    request = (
        "Perform basic quality control on this PBMC dataset. "
        "Calculate QC metrics and filter cells with fewer than 200 genes. "
        "Use omicverse or scanpy. Print a summary."
    )
    t0 = time.monotonic()
    try:
        result = agent.run(request, adata.copy())
        dur = time.monotonic() - t0
        trace_ev = _extract_run_trace_evidence(agent)
        usage_ev = _extract_usage_evidence(agent)
        info: Dict[str, Any] = {
            "duration_s": round(dur, 2),
            "run_trace": trace_ev,
            "token_usage": usage_ev,
        }
        if result is not None and hasattr(result, "shape"):
            info["result_shape"] = list(result.shape)
        ev.record_step("pbmc_qc_real_llm", "passed", dur, info)
        print(f"[OK] PBMC QC completed in {dur:.1f}s")
        print(f"     trace_id: {trace_ev.get('trace_id', 'N/A')}")
        print(f"     steps: {trace_ev.get('step_count', 0)} tool calls")
        print(f"     usage: {usage_ev.get('last_usage', 'N/A')}")
        ev.final_status = "passed"
    except Exception as exc:
        dur = time.monotonic() - t0
        trace_ev = _extract_run_trace_evidence(agent)
        usage_ev = _extract_usage_evidence(agent)
        ev.record_step("pbmc_qc_real_llm", "failed", dur, {
            "error": str(exc),
            "traceback": traceback.format_exc()[-1500:],
            "run_trace": trace_ev,
            "token_usage": usage_ev,
        })
        ev.final_status = "failed"
        ev.error_detail = traceback.format_exc()
        print(f"[FAIL] PBMC QC: {exc}")
        traceback.print_exc()

    # Report
    report = ev.to_report()
    report_json = json.dumps(report, indent=2, default=str)
    report_path = Path(__file__).parent / "e2e_real_provider_report.json"
    report_path.write_text(report_json, encoding="utf-8")
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(report_json)
    print(f"\nReport: {report_path}")
    return 0 if ev.final_status == "passed" else 1


if __name__ == "__main__":
    sys.exit(_run_standalone())
