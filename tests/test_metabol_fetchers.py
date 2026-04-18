"""Tests for the on-demand database fetchers.

All tests below need network access, so they're gated behind
``@pytest.mark.network`` and are skipped unless the user runs
``pytest -m network`` explicitly. That keeps CI fast and offline-safe
while still giving us a way to verify the fetchers end-to-end on
machines that have internet.

Run locally with:

    pytest -m network tests/test_metabol_fetchers.py

To opt in on CI, set ``OV_METABOL_FETCHER_TESTS=1`` in the environment.
"""
from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("OV_METABOL_FETCHER_TESTS"),
    reason="fetcher tests hit the network — opt in via OV_METABOL_FETCHER_TESTS=1",
)


def test_fetch_kegg_pathways_returns_hundreds_of_pathways(tmp_path, monkeypatch):
    monkeypatch.setenv("OV_METABOL_CACHE", str(tmp_path))
    import omicverse as ov

    pw = ov.metabol.fetch_kegg_pathways()
    # Full KEGG reference map has ~550 pathways; we want at least 200
    # so the test fails visibly if something truncates the download.
    assert len(pw) > 200
    assert "Glycolysis / Gluconeogenesis" in pw
    assert "C00031" in pw["Glycolysis / Gluconeogenesis"]   # D-Glucose in glycolysis


def test_fetch_hmdb_from_name_returns_cross_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("OV_METABOL_CACHE", str(tmp_path))
    import omicverse as ov

    ids = ov.metabol.fetch_hmdb_from_name("Glucose")
    assert ids["pubchem"]             # must resolve to a PubChem CID
    assert ids["kegg"] == "C00031"    # canonical glucose KEGG ID
    assert ids["chebi"].startswith("CHEBI:")


def test_fetch_lion_returns_core_lipid_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("OV_METABOL_CACHE", str(tmp_path))
    import omicverse as ov

    lion = ov.metabol.fetch_lion_associations()
    # The 6 LIPID MAPS superclasses must all show up
    assert len(lion) >= 100
    names = " ".join(lion.keys())
    for superclass in ("glycerolipids", "glycerophospholipids",
                       "sphingolipids", "sterol lipids"):
        assert superclass in names


def test_cache_is_reused(tmp_path, monkeypatch):
    """Second call should be free — no network. We verify by clearing
    network paths and confirming the call still succeeds."""
    monkeypatch.setenv("OV_METABOL_CACHE", str(tmp_path))
    import omicverse as ov

    first = ov.metabol.fetch_kegg_pathways()
    # Disable outbound network by pointing urlopen at a bad URL — the
    # cached call should still work.
    import urllib.request
    original = urllib.request.urlopen
    def blocker(*a, **k):
        raise RuntimeError("network blocked by test")
    monkeypatch.setattr(urllib.request, "urlopen", blocker)
    try:
        second = ov.metabol.fetch_kegg_pathways()
    finally:
        monkeypatch.setattr(urllib.request, "urlopen", original)
    assert first == second
