"""v0.2 tests for ``omicverse.metabol`` — ID mapping, MSEA, mummichog, lipidomics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_id_mapping_resolves_cachexia_metabolites():
    from omicverse.metabol import map_ids

    names = ["Isoleucine", "Uracil", "Glucose", "Acetone", "TMAO",
             "Creatinine", "Carnitine", "totally_made_up_name"]
    df = map_ids(names)
    assert df.loc["Isoleucine", "hmdb"] == "HMDB0000172"
    assert df.loc["Isoleucine", "kegg"] == "C00407"
    assert df.loc["TMAO", "kegg"] == "C01104"              # via alias
    assert df.loc["totally_made_up_name", "hmdb"] == ""   # unresolved


def test_id_mapping_case_and_whitespace_robust():
    from omicverse.metabol import map_ids

    # Mixed-case and extra whitespace shouldn't break lookup
    df = map_ids(["ISOLEUCINE", "  Glucose  ", "l-isoleucine"])
    assert df.iloc[0]["kegg"] == "C00407"
    assert df.iloc[1]["kegg"] == "C00031"
    assert df.iloc[2]["kegg"] == "C00407"   # matched via aliases


def test_msea_ora_finds_relevant_pathways_on_cachexia(cachexia_adata):
    """cachexia urine metabolites are enriched for amino-acid / TCA pathways."""
    from omicverse.metabol import (differential, msea_ora, normalize, transform)

    a = normalize(cachexia_adata, method="pqn")
    a = transform(a, method="log")
    deg = differential(a, method="welch_t", log_transformed=True)
    hits = deg[deg["padj"] < 0.20].index.tolist()
    background = deg.index.tolist()

    ora = msea_ora(hits, background, min_size=3)
    assert not ora.empty
    for col in ("pathway", "overlap", "set_size", "odds_ratio", "pvalue", "padj"):
        assert col in ora.columns
    # Top hits should include amino-acid metabolism pathways
    top_terms = set(ora.head(10)["pathway"].tolist())
    assert any("TCA" in t or "Alanine" in t or "Aminoacyl" in t or "Metabolic" in t
               for t in top_terms), f"Expected amino-acid/TCA pathways, got {top_terms}"


def test_msea_gsea_runs_and_returns_results(cachexia_adata):
    # Vendored gseapy path — skip gracefully if the omicverse install is
    # missing the external subpackage (e.g. a stripped-down build).
    pytest.importorskip("omicverse.external.gseapy")
    from omicverse.metabol import differential, msea_gsea, normalize, transform

    a = normalize(cachexia_adata, method="pqn")
    a = transform(a, method="log")
    deg = differential(a, method="welch_t", log_transformed=True)
    out = msea_gsea(deg, stat_col="stat", n_perm=200, seed=0)
    assert not out.empty
    assert "NES" in out.columns or "es" in out.columns


def test_mummichog_annotates_positive_mode_peaks():
    from omicverse.metabol import annotate_peaks
    from omicverse.metabol._id_mapping import _load_lookup

    lu = _load_lookup()
    lu = lu[lu["mw"].notna() & (lu["kegg"] != "")].reset_index(drop=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lu), size=10, replace=False)
    hit_mz = lu["mw"].iloc[idx].to_numpy() + 1.00728  # [M+H]+
    ann = annotate_peaks(hit_mz, polarity="positive", ppm=10.0)
    # Each synthetic peak was computed from a known KEGG compound — should
    # get at least that compound back (plus possibly isomers with same mass)
    expected_keggs = set(lu["kegg"].iloc[idx])
    got_keggs = set(ann["kegg"])
    assert expected_keggs.issubset(got_keggs), (
        f"Missing: {expected_keggs - got_keggs}"
    )


def test_mummichog_basic_enriches_spiked_pathway():
    """Spike in every member of the TCA pathway as 'hit' peaks — mummichog
    should put TCA cycle at the top of the enrichment table."""
    from omicverse.metabol import load_pathways, mummichog_basic
    from omicverse.metabol._id_mapping import _load_lookup

    pathways = load_pathways()
    tca_ids = pathways["Citrate cycle (TCA cycle)"]
    lu = _load_lookup()
    lu = lu[lu["mw"].notna() & lu["kegg"].isin(tca_ids)].reset_index(drop=True)
    if len(lu) < 4:
        pytest.skip("not enough TCA compounds in local lookup to spike a signal")
    # Synthesize hit peaks from TCA compounds + a big background
    hit_mz = lu["mw"].to_numpy() + 1.00728
    rng = np.random.default_rng(0)
    bg_mz = rng.uniform(50, 1200, size=60)
    all_mz = np.concatenate([hit_mz, bg_mz])
    pvalue = np.concatenate([np.full(len(hit_mz), 0.001), np.full(60, 0.5)])

    res = mummichog_basic(all_mz, pvalue, polarity="positive",
                         ppm=10.0, n_perm=500, min_overlap=2)
    assert not res.empty
    top3 = set(res.head(3)["pathway"])
    assert "Citrate cycle (TCA cycle)" in top3, (
        f"TCA cycle not in top-3 despite spiking; got {top3}"
    )


def test_lipidomics_parse_lipid_names():
    from omicverse.metabol import parse_lipid

    # Canonical forms
    pc = parse_lipid("PC 34:1")
    assert pc is not None
    assert pc.lipid_class == "PC"
    assert pc.total_carbons == 34
    assert pc.total_db == 1
    assert pc.backbone is None

    tag = parse_lipid("TAG 54:3")
    assert tag.lipid_class == "TAG" and tag.total_carbons == 54 and tag.total_db == 3

    cer = parse_lipid("Cer d18:1/24:0")
    assert cer.lipid_class == "CER"
    assert cer.backbone == "d18:1"
    assert cer.total_carbons == 24
    assert cer.total_db == 0

    # Not a lipid
    assert parse_lipid("Glucose") is None
    assert parse_lipid("Isoleucine") is None


def test_lipidomics_annotate_and_aggregate():
    import anndata as ad
    from omicverse.metabol import aggregate_by_class, annotate_lipids

    names = ["PC 34:1", "PC 36:2", "PE 34:1", "PE 36:4", "TAG 54:3", "TAG 52:2",
             "LPC 18:0", "SM 34:1"]
    X = np.random.default_rng(0).uniform(10, 1000, size=(20, len(names)))
    adata = ad.AnnData(X=X,
                       obs=pd.DataFrame(index=[f"s{i}" for i in range(20)]),
                       var=pd.DataFrame(index=names))
    adata = annotate_lipids(adata)
    assert "lipid_class" in adata.var.columns
    assert set(adata.var["lipid_class"]) == {"PC", "PE", "TAG", "LPC", "SM"}

    agg = aggregate_by_class(adata, agg="sum")
    assert agg.n_vars == 5                  # 5 distinct classes
    assert agg.n_obs == 20
    # Sum of PC species must equal aggregated PC total
    pc_cols = [i for i, c in enumerate(adata.var["lipid_class"]) if c == "PC"]
    expected = np.asarray(adata.X[:, pc_cols]).sum(axis=1)
    got = np.asarray(agg.X[:, list(agg.var_names).index("PC")])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_msea_ora_empty_when_no_hits_resolve_to_kegg():
    """Edge case: user hits all unresolvable → clear ValueError, not
    confusing IndexError downstream."""
    from omicverse.metabol import msea_ora
    with pytest.raises(ValueError, match="resolve to KEGG"):
        msea_ora(
            hits=["totally_made_up_1", "fake_compound_2"],
            background=["totally_made_up_1", "fake_compound_2", "another_fake"],
        )


def test_differential_single_group_raises():
    """Edge case: group column has only one level → clear error."""
    import anndata as ad
    from omicverse.metabol import differential

    adata = ad.AnnData(
        X=np.random.default_rng(0).normal(size=(5, 10)),
        obs=pd.DataFrame({"group": ["a"] * 5},
                         index=[f"s{i}" for i in range(5)]),
        var=pd.DataFrame(index=[f"m{i}" for i in range(10)]),
    )
    with pytest.raises(ValueError, match="fewer than 2 unique"):
        differential(adata, group_col="group")


def test_impute_seed_is_deterministic_and_overridable():
    """QRILC should be deterministic per seed, and different seeds should
    produce different imputations (bugfix: seed was hardcoded to 42)."""
    import anndata as ad
    from omicverse.metabol import impute

    rng = np.random.default_rng(0)
    X = rng.uniform(1, 100, size=(30, 20))
    X[rng.random(X.shape) < 0.2] = np.nan
    adata = ad.AnnData(X=X,
                       obs=pd.DataFrame(index=[f"s{i}" for i in range(30)]),
                       var=pd.DataFrame(index=[f"m{i}" for i in range(20)]))
    a = impute(adata, method="qrilc", seed=0).X
    a2 = impute(adata, method="qrilc", seed=0).X
    b = impute(adata, method="qrilc", seed=123).X
    # Same seed ⇒ same draws
    np.testing.assert_allclose(a, a2, rtol=0, atol=0)
    # Different seed ⇒ different imputed values (at least somewhere)
    assert not np.allclose(a, b)


def test_lion_enrichment_spots_glycerophospholipid_hit():
    from omicverse.metabol import lion_enrichment

    hits = ["PC 34:1", "PC 36:2", "PE 34:1", "PE 36:4", "LPE 18:0"]
    background = hits + ["TAG 54:3", "TAG 52:2", "SM 34:1", "Cer d18:1/24:0",
                         "CE 18:2", "DAG 36:2"]
    out = lion_enrichment(hits, background, min_size=2)
    assert not out.empty
    for col in ("term", "overlap", "set_size", "odds_ratio", "pvalue", "padj"):
        assert col in out.columns
    top_terms = set(out.head(5)["term"].tolist())
    # The entire hit list is glycerophospholipids; that term should dominate
    assert "Glycerophospholipids" in top_terms or "Phosphatidylcholines" in top_terms
