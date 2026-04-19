"""R parity tests for ``omicverse.metabol``.

Drives MetaboAnalystR via subprocess (no rpy2) on the cachexia CSV,
dumps intermediate outputs to TSV, then asserts that the Python port
produces matching values. Gracefully ``pytest.skip``s when R /
MetaboAnalystR aren't available — same pattern as
``test_doubletfinder_qc.py``'s R-parity tests.

What we check (and don't check)
-------------------------------
- **Top-hit ranking overlap** on the t-test — the top-15 metabolites
  by p-value from Python and MetaboAnalystR should overlap ≥ 10.
- **VIP score rank correlation** Spearman ρ ≥ 0.6 (different weight
  normalizations mean exact equality isn't expected).
- **PLS-DA score separation** — both should give similar group
  centroids in the t1/t2 plane (we measure via silhouette).

We do NOT check bit-for-bit score equality because MetaboAnalystR's
PLS-DA wraps ``pls`` whose NIPALS tolerance and Q² CV splits differ
slightly from our implementation — the goal is biological agreement,
not numerical equality.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


HERE = Path(__file__).parent
DRIVER = HERE / "metabol_r_reference_driver.R"


# ---------------------------------------------------------------------------
# CI-safe driver smoke test — runs without R, catches script-file drift
# ---------------------------------------------------------------------------
def test_r_reference_driver_exists_and_is_well_formed():
    """The R-parity driver script must stay syntactically valid even when
    the full R environment isn't present on CI. Checks structural anchors
    so any refactor drift is caught on every push, not just the dev box."""
    assert DRIVER.is_file(), f"driver missing at {DRIVER}"
    src = DRIVER.read_text()
    # Anchors that the Python side (below) depends on
    for required in (
        "library(MetaboAnalystR)",
        "Ttests.Anal",
        "PLSR.Anal",
        "norm_matrix.tsv",
        "ttest.tsv",
        "plsda_scores.tsv",
        "plsda_vip.tsv",
    ):
        assert required in src, (
            f"driver script missing expected anchor {required!r} — "
            "refactor drift; update tests or driver"
        )


def _find_rscript() -> str | None:
    """Locate Rscript.

    In order: the ``OV_METABOL_RSCRIPT`` env var (for dev machines with
    Rscript in a non-standard location), then whatever ``shutil.which``
    finds on ``PATH``. Portable across machines.
    """
    env_path = os.environ.get("OV_METABOL_RSCRIPT")
    if env_path and Path(env_path).exists():
        return env_path
    from_path = shutil.which("Rscript")
    return from_path if from_path else None


@pytest.fixture(scope="module")
def r_ref_dir(tmp_path_factory) -> Path:
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript not found")
    # Need the cachexia CSV; reuse the fixture's cache
    csv_path = Path(os.environ.get("OV_METABOL_TEST_CACHE", "/tmp")) / "human_cachexia.csv"
    if not csv_path.exists():
        import urllib.request
        try:
            urllib.request.urlretrieve(
                "https://rest.xialab.ca/api/download/metaboanalyst/human_cachexia.csv",
                csv_path,
            )
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"cannot fetch cachexia CSV: {exc}")

    outdir = tmp_path_factory.mktemp("metabol_r_ref")
    env = os.environ.copy()
    # Dev machines may have a dedicated MetaboAnalystR R-lib dir; respect
    # OV_METABOL_R_LIBS_USER to point at one without hard-coding a path.
    r_libs = os.environ.get("OV_METABOL_R_LIBS_USER")
    if r_libs and Path(r_libs).is_dir():
        env["R_LIBS_USER"] = r_libs
    proc = subprocess.run(
        [rscript, str(DRIVER), str(csv_path), str(outdir)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        pytest.skip(
            "MetaboAnalystR driver failed — install with "
            "`BiocManager::install('MetaboAnalystR')` in the CMAP env:\n"
            f"STDERR:\n{proc.stderr[-1500:]}"
        )
    return outdir


def test_ttest_top_hits_overlap_with_MetaboAnalystR(r_ref_dir, cachexia_adata):
    from omicverse.metabol import differential, normalize, transform

    r_tt = pd.read_csv(r_ref_dir / "ttest.tsv", sep="\t").set_index("metabolite")
    py = normalize(cachexia_adata, method="pqn")
    py = transform(py, method="log")
    py_tt = differential(py, method="welch_t", log_transformed=True)

    top_r = set(r_tt.sort_values("pvalue").head(15).index)
    top_py = set(py_tt.sort_values("pvalue").head(15).index)
    overlap = len(top_r & top_py)
    assert overlap >= 10, (
        f"Only {overlap}/15 top-p-value metabolites overlap between "
        f"MetaboAnalystR and pyMetabo. R={sorted(top_r)}, py={sorted(top_py)}"
    )


def test_vip_rank_correlates_with_MetaboAnalystR(r_ref_dir, cachexia_adata):
    from omicverse.metabol import normalize, plsda, transform

    r_vip = pd.read_csv(r_ref_dir / "plsda_vip.tsv", sep="\t").set_index("metabolite")
    # MetaboAnalystR reports VIP per component; take Comp.1
    r_vip_1 = r_vip.iloc[:, 0]

    py = normalize(cachexia_adata, method="pqn")
    py = transform(py, method="log")
    py = transform(py, method="pareto", stash_raw=False)
    res = plsda(py, n_components=2)
    py_vip = pd.Series(res.vip, index=py.var_names)

    common = r_vip_1.index.intersection(py_vip.index)
    rho, _ = spearmanr(r_vip_1.loc[common].values, py_vip.loc[common].values)
    assert rho >= 0.6, f"VIP Spearman rho={rho:.3f} < 0.6 — ports have drifted"


def test_plsda_scores_separate_groups(r_ref_dir, cachexia_adata):
    """Both implementations should produce a t1 axis that separates
    cachexic from control. We measure via the silhouette coefficient
    on the 2-D score matrix — both must be > 0 (better than random)."""
    from sklearn.metrics import silhouette_score

    from omicverse.metabol import normalize, plsda, transform

    r_scores = pd.read_csv(r_ref_dir / "plsda_scores.tsv", sep="\t")
    # Align with our adata
    groups = cachexia_adata.obs["group"].reindex(r_scores["sample"]).values
    r_sil = silhouette_score(r_scores[["pls_t1", "pls_t2"]].values, groups)

    py = normalize(cachexia_adata, method="pqn")
    py = transform(py, method="log")
    py = transform(py, method="pareto", stash_raw=False)
    res = plsda(py, n_components=2)
    py_sil = silhouette_score(res.scores, cachexia_adata.obs["group"].values)

    assert r_sil > 0 and py_sil > 0, (
        f"Both should give positive silhouette — got R={r_sil:.2f}, py={py_sil:.2f}"
    )
    # And they shouldn't be wildly different
    assert abs(r_sil - py_sil) < 0.3, (
        f"Silhouette drift too large: R={r_sil:.3f} vs py={py_sil:.3f}"
    )
