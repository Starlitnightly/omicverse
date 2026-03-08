"""Dependency detection helpers for MCP environment-gated tests.

Uses ``importlib.util.find_spec()`` for zero-cost detection — no heavy imports
are triggered.  Provides convenience ``skipif`` decorators for each tier.
"""

from __future__ import annotations

import importlib.util

import pytest


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _can_import(name: str) -> bool:
    """Return True if *name* is importable (no actual import)."""
    return importlib.util.find_spec(name) is not None


# Tier 1: Core
def has_anndata() -> bool:
    return _can_import("anndata")


def has_scanpy() -> bool:
    return _can_import("scanpy")


def core_available() -> bool:
    return has_anndata() and has_scanpy()


# Tier 2: Scientific
def has_scvelo() -> bool:
    return _can_import("scvelo")


def has_squidpy() -> bool:
    return _can_import("squidpy")


def scientific_stack_available() -> bool:
    return core_available() and has_scvelo() and has_squidpy()


# Tier 3: Extended (P2 class-tool dependencies)
def has_seacells() -> bool:
    return _can_import("SEACells")


def has_pertpy() -> bool:
    return _can_import("pertpy")


def has_mira() -> bool:
    return _can_import("mira")


# ---------------------------------------------------------------------------
# Convenience skipif decorators
# ---------------------------------------------------------------------------

skip_no_core = pytest.mark.skipif(
    not core_available(), reason="requires anndata + scanpy",
)
skip_no_scientific = pytest.mark.skipif(
    not scientific_stack_available(), reason="requires scientific stack (scvelo + squidpy)",
)
skip_no_seacells = pytest.mark.skipif(
    not has_seacells(), reason="requires SEACells",
)
skip_no_pertpy = pytest.mark.skipif(
    not has_pertpy(), reason="requires pertpy",
)
skip_no_mira = pytest.mark.skipif(
    not has_mira(), reason="requires mira",
)
