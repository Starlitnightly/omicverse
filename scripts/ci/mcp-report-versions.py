#!/usr/bin/env python3
"""MCP CI Version Snapshot Tool.

Probes installed package versions via importlib.metadata.version() and emits
a JSON report.  No heavy imports -- safe to run in any CI tier.
"""

import argparse
import datetime
import json
import os
import platform
import sys
from importlib.metadata import version as _pkg_version

VALID_PROFILES = {
    "fast-mock",
    "core-runtime",
    "scientific-runtime",
    "extended-runtime",
}

PROFILE_PACKAGES = {
    "fast-mock": [
        "omicverse", "numpy", "pandas", "pytest", "pytest-asyncio",
    ],
    "core-runtime": [
        "omicverse", "anndata", "scanpy", "numpy", "pandas", "scipy",
        "matplotlib",
    ],
    "scientific-runtime": [
        "omicverse", "anndata", "scanpy", "numpy", "pandas", "scipy",
        "matplotlib", "scvelo", "squidpy",
    ],
    "extended-runtime": [
        "omicverse", "anndata", "scanpy", "numpy", "pandas", "scipy",
        "matplotlib", "scvelo", "squidpy", "pertpy", "SEACells", "mira",
    ],
}


def _probe_version(package: str):
    """Return installed version string, or None if not installed."""
    try:
        return _pkg_version(package)
    except Exception:
        return None


def build_report(profile: str, source: str = "local") -> dict:
    """Build a version snapshot dict for the given profile."""
    packages = {}
    for pkg in PROFILE_PACKAGES[profile]:
        packages[pkg] = _probe_version(pkg)

    return {
        "schema_version": 1,
        "profile": profile,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCP CI version snapshot",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="CI profile name",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--source",
        default="local",
        choices=["local", "ci"],
        help="Execution context: 'local' (default) or 'ci'",
    )
    args = parser.parse_args()

    if args.profile not in VALID_PROFILES:
        parser.error(
            f"Invalid profile {args.profile!r}. "
            f"Valid profiles: {', '.join(sorted(VALID_PROFILES))}"
        )

    report = build_report(args.profile, source=args.source)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    else:
        json.dump(report, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
