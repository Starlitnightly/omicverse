"""
Runtime availability checking for MCP tools.

First-batch tools are all low-dependency (no GPU, no network, no external
binaries).  The checking infrastructure is defined here so that later phases
can plug in real probes.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Known heavy-dependency markers (populated as tools are classified)
# ---------------------------------------------------------------------------

GPU_REQUIRED_CATEGORIES = {"gpu_preprocessing"}
NETWORK_REQUIRED_CATEGORIES = {"biocontext", "alignment"}
BINARY_REQUIREMENTS: Dict[str, List[str]] = {
    # full_name → list of required binaries
    "omicverse.alignment.STAR.STAR": ["STAR"],
    "omicverse.alignment.fastp.fastp": ["fastp"],
    "omicverse.alignment.featureCount.featureCount": ["featureCounts"],
    "omicverse.alignment.fq_dump.fqdump": ["fasterq-dump"],
    "omicverse.alignment.prefetch.prefetch": ["prefetch"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_availability(entry: dict, *, class_spec=None) -> dict:
    """Return a static availability dict for a manifest entry.

    This is embedded in the manifest at build time and captures known
    dependency flags without doing runtime probes.

    Parameters
    ----------
    entry : dict
        Manifest entry with at least ``full_name`` and ``category``.
    class_spec : ClassWrapperSpec, optional
        If provided, probe class-specific runtime requirements.
    """
    full_name = entry.get("full_name", "")
    category = entry.get("category", "")

    requires_gpu = _needs_gpu(full_name, category)
    requires_network = _needs_network(full_name, category)
    required_binaries = BINARY_REQUIREMENTS.get(full_name, [])
    required_env = _required_env_vars(full_name)

    available = True
    reason = ""

    # Class-specific availability (spec flag + package probing)
    if class_spec is not None:
        available, reason = check_class_availability(class_spec)

    return {
        "available": available,
        "requires_gpu": requires_gpu,
        "requires_network": requires_network,
        "required_binaries": required_binaries,
        "required_env": required_env,
        "reason": reason,
    }


def check_tool_availability(entry: dict) -> dict:
    """Perform runtime availability checks and return an updated dict."""
    avail = dict(entry.get("availability", build_availability(entry)))

    # If already marked unavailable (e.g. class spec gate), preserve that
    if not avail.get("available", True) and avail.get("reason"):
        return avail

    reasons: List[str] = []

    if avail["requires_gpu"]:
        ok, msg = check_gpu_requirement(entry)
        if not ok:
            reasons.append(msg)

    if avail["required_binaries"]:
        ok, missing = check_binary_requirements(entry)
        if not ok:
            reasons.append(f"Missing binaries: {', '.join(missing)}")

    if avail["required_env"]:
        ok, missing = check_env_requirements(entry)
        if not ok:
            reasons.append(f"Missing env vars: {', '.join(missing)}")

    if avail["requires_network"]:
        ok, msg = check_network_requirement(entry)
        if not ok:
            reasons.append(msg)

    avail["available"] = len(reasons) == 0
    avail["reason"] = "; ".join(reasons)
    return avail


def check_gpu_requirement(entry: dict) -> Tuple[bool, str]:
    """Check whether a CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, ""
        return False, "CUDA GPU required but not available"
    except ImportError:
        return False, "PyTorch not installed (GPU check unavailable)"


def check_binary_requirements(entry: dict) -> Tuple[bool, List[str]]:
    """Check whether required external binaries are in PATH."""
    required = entry.get("availability", {}).get("required_binaries", [])
    missing = [b for b in required if shutil.which(b) is None]
    return len(missing) == 0, missing


def check_env_requirements(entry: dict) -> Tuple[bool, List[str]]:
    """Check whether required environment variables are set."""
    required = entry.get("availability", {}).get("required_env", [])
    missing = [v for v in required if not os.environ.get(v)]
    return len(missing) == 0, missing


def check_network_requirement(entry: dict) -> Tuple[bool, str]:
    """Tag-only check for network requirement.  No active probe."""
    # First batch: just mark, don't block
    return True, ""


def check_class_availability(spec) -> Tuple[bool, str]:
    """Probe whether a ClassWrapperSpec's runtime requirements are met.

    Checks ``spec.available`` flag first (catches deferred tools), then
    probes ``spec.runtime_requirements["packages"]`` via
    ``importlib.util.find_spec()`` for zero-cost detection (no heavy import).

    Returns ``(available, reason)``.
    """
    if not spec.available:
        return False, f"Deferred (rollout_phase={spec.rollout_phase})"

    requirements = getattr(spec, "runtime_requirements", {})
    packages = requirements.get("packages", [])
    if not packages:
        return True, ""

    import importlib.util

    missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"

    return True, ""


def merge_availability_reasons(
    gpu: Tuple[bool, str],
    binaries: Tuple[bool, List[str]],
    env: Tuple[bool, List[str]],
    network: Tuple[bool, str],
) -> dict:
    """Combine individual check results into a single availability dict."""
    reasons: List[str] = []
    if not gpu[0]:
        reasons.append(gpu[1])
    if not binaries[0]:
        reasons.append(f"Missing binaries: {', '.join(binaries[1])}")
    if not env[0]:
        reasons.append(f"Missing env vars: {', '.join(env[1])}")
    if not network[0]:
        reasons.append(network[1])
    return {
        "available": len(reasons) == 0,
        "reason": "; ".join(reasons),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _needs_gpu(full_name: str, category: str) -> bool:
    if category in GPU_REQUIRED_CATEGORIES:
        return True
    gpu_keywords = {"_gpu", "GPU", "cuda"}
    return any(kw in full_name for kw in gpu_keywords)


def _needs_network(full_name: str, category: str) -> bool:
    if category in NETWORK_REQUIRED_CATEGORIES:
        return True
    return False


def _required_env_vars(full_name: str) -> List[str]:
    """Return env vars known to be required by a specific tool."""
    # Expand as tools are onboarded
    return []
