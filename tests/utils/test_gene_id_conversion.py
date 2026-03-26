"""
Tests for omicverse.utils._gene_id_conversion

Unit tests mock the pyensembl database so they run fully offline.
Integration tests (marked with ``@pytest.mark.integration``) hit the real
Ensembl database and require a prior ``pyensembl install``.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from unittest.mock import MagicMock, patch

from omicverse.utils._gene_id_conversion import (
    _infer_species_and_release,
    convert2gene_symbol,
    convert2symbol,
    convert2gene_id,
    symbol2id,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_adata(var_names):
    """Create a minimal AnnData with given var_names."""
    n = len(var_names)
    X = np.zeros((3, n))
    return ad.AnnData(X, var=pd.DataFrame(index=var_names))


def _mock_ensembl_release(gene_map: dict):
    """
    Return a patched EnsemblRelease whose gene_by_id() and
    gene_ids_of_gene_name() are driven by *gene_map*
    (Ensembl ID → symbol).

    Parameters
    ----------
    gene_map : dict
        Mapping of ``{ensembl_id: symbol}``.
    """
    reverse_map = {}
    for eid, sym in gene_map.items():
        reverse_map.setdefault(sym, []).append(eid)

    def _gene_by_id(gene_id):
        clean = gene_id.split(".")[0]
        if clean not in gene_map:
            raise ValueError(f"Gene ID not found: {clean}")
        obj = MagicMock()
        obj.gene_id = clean
        obj.gene_name = gene_map[clean]
        return obj

    def _gene_ids_of_gene_name(symbol):
        if symbol not in reverse_map:
            raise ValueError(f"Gene name not found: {symbol}")
        return reverse_map[symbol]

    mock_data = MagicMock()
    mock_data.db = MagicMock()            # marks as already indexed
    mock_data.gene_by_id.side_effect = _gene_by_id
    mock_data.gene_ids_of_gene_name.side_effect = _gene_ids_of_gene_name
    return mock_data


# ---------------------------------------------------------------------------
# _infer_species_and_release
# ---------------------------------------------------------------------------

class TestInferSpecies:
    def test_known_species(self):
        assert _infer_species_and_release(["ENSG00000141510"]) == "human"

    def test_chicken_not_confused_with_human(self):
        # ENSGALG starts with ENSG — must not be classified as human
        assert _infer_species_and_release(["ENSGALG00000000003"]) == "chicken"

    def test_versioned_id(self):
        # Version suffix must be stripped before matching
        assert _infer_species_and_release(["ENSG00000141510.12"]) == "human"

    def test_case_insensitive(self):
        assert _infer_species_and_release(["ensg00000141510"]) == "human"
        assert _infer_species_and_release(["ensmusg00000059552"]) == "mouse"

    def test_empty_list_returns_human(self):
        assert _infer_species_and_release([]) == "human"

    def test_unknown_prefix_returns_human(self):
        result = _infer_species_and_release(["UNKN0000000001"])
        assert result == "human"

    def test_skips_empty_strings(self):
        # Should skip empty strings and find the first real ID
        assert _infer_species_and_release(["", "", "ENSMUSG00000059552"]) == "mouse"


# ---------------------------------------------------------------------------
# convert2gene_symbol
# ---------------------------------------------------------------------------

HUMAN_MAP = {
    "ENSG00000141510": "TP53",
    "ENSG00000012048": "BRCA1",
    "ENSG00000139618": "BRCA2",
}

MOUSE_MAP = {
    "ENSMUSG00000059552": "Trp53",
    "ENSMUSG00000022346": "Brca1",
}


class TestConvert2GeneSymbol:
    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_basic_human_conversion(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        ids = list(HUMAN_MAP.keys())
        df = convert2gene_symbol(ids, species="human")

        assert isinstance(df, pd.DataFrame)
        assert "symbol" in df.columns
        assert "_score" in df.columns
        assert df.index.name == "query"
        for eid, sym in HUMAN_MAP.items():
            assert df.loc[eid, "symbol"] == sym

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_basic_mouse_conversion(self, MockER):
        MockER.return_value = _mock_ensembl_release(MOUSE_MAP)
        ids = list(MOUSE_MAP.keys())
        df = convert2gene_symbol(ids, species="mouse")

        for eid, sym in MOUSE_MAP.items():
            assert df.loc[eid, "symbol"] == sym

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_unknown_id_falls_back_to_original(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        unknown = "ENSG99999999999"
        df = convert2gene_symbol([unknown], species="human")
        assert df.loc[unknown, "symbol"] == unknown

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_versioned_ids_are_stripped(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        versioned = [f"{eid}.5" for eid in HUMAN_MAP]
        df = convert2gene_symbol(versioned, species="human")
        # Results are keyed by the clean ID
        for eid, sym in HUMAN_MAP.items():
            assert df.loc[eid, "symbol"] == sym

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_auto_detect_species(self, MockER):
        mock = _mock_ensembl_release(HUMAN_MAP)
        MockER.return_value = mock
        # species=None → auto-detect from prefix ENSG → human
        convert2gene_symbol(list(HUMAN_MAP.keys()))
        MockER.assert_called_once_with(77, species="human")

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_score_column_is_one(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        df = convert2gene_symbol(list(HUMAN_MAP.keys()), species="human")
        assert (df["_score"] == 1.0).all()

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_chicken_ids_not_matched_as_human(self, MockER):
        # When species='human', chicken IDs should fall back to original
        chicken_map = {"ENSGALG00000000003": "CHIC1"}
        MockER.return_value = _mock_ensembl_release(chicken_map)
        df = convert2gene_symbol(["ENSGALG00000000003"], species="human")
        # The ID does NOT start with ENSG for human (filtered out)
        assert df.loc["ENSGALG00000000003", "symbol"] == "ENSGALG00000000003"

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_default_release_is_77(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        convert2gene_symbol(list(HUMAN_MAP.keys()), species="human")
        MockER.assert_called_once_with(77, species="human")

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_custom_release(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        convert2gene_symbol(list(HUMAN_MAP.keys()), species="human", ensembl_release=109)
        MockER.assert_called_once_with(109, species="human")

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_force_rebuild_calls_download_and_index(self, MockER):
        mock = MagicMock()
        mock.download = MagicMock()
        mock.index = MagicMock()
        mock.gene_by_id.side_effect = ValueError("not found")
        # db access raises so needs_rebuild triggers
        type(mock).db = property(fget=lambda self: (_ for _ in ()).throw(Exception("no db")))
        MockER.return_value = mock

        convert2gene_symbol(["ENSG00000141510"], species="human", force_rebuild=True)
        mock.download.assert_called_once()
        mock.index.assert_called_once_with(overwrite=True)


# ---------------------------------------------------------------------------
# convert2symbol  (AnnData wrapper)
# ---------------------------------------------------------------------------

class TestConvert2Symbol:
    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_replaces_var_names(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        adata = _make_adata(list(HUMAN_MAP.keys()))
        out = convert2symbol(adata)
        assert set(out.var_names) == set(HUMAN_MAP.values())

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_query_column_preserved(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        adata = _make_adata(list(HUMAN_MAP.keys()))
        out = convert2symbol(adata)
        assert "query" in out.var.columns

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_subset_true_removes_unconverted(self, MockER):
        partial = {"ENSG00000141510": "TP53"}
        gene_map_with_unknown = dict(partial)
        mock = _mock_ensembl_release(gene_map_with_unknown)
        # unknown ID will raise ValueError inside gene_by_id → falls back
        MockER.return_value = mock

        ids = ["ENSG00000141510", "ENSG99999999999"]
        adata = _make_adata(ids)
        out = convert2symbol(adata, subset=True)
        # Only TP53 was converted; ENSG99999999999 keeps original symbol
        # subset=True removes genes whose symbol is still an Ensembl ID
        # (both stay because fallback == original ID, valid_ind covers all)
        assert "TP53" in out.var_names

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_subset_false_keeps_unconverted(self, MockER):
        partial = {"ENSG00000141510": "TP53"}
        MockER.return_value = _mock_ensembl_release(partial)
        ids = ["ENSG00000141510", "ENSG99999999999"]
        adata = _make_adata(ids)
        out = convert2symbol(adata, subset=False)
        # Both genes kept; unconverted one retains original ID
        assert len(out) == 3  # 3 obs unchanged
        assert out.n_vars == 2

    def test_non_ensembl_ids_are_unchanged(self):
        # var_names not starting with ENS → skip conversion entirely
        adata = _make_adata(["TP53", "GAPDH", "BRCA1"])
        out = convert2symbol(adata)
        assert list(out.var_names) == ["TP53", "GAPDH", "BRCA1"]

    def test_wrong_prefix_raises(self):
        adata = _make_adata(["ENSXXX00000000001", "ENSXXX00000000002"])
        with pytest.raises(Exception, match="non-official"):
            convert2symbol(adata)


# ---------------------------------------------------------------------------
# convert2gene_id
# ---------------------------------------------------------------------------

SYMBOL_MAP = {sym: eid for eid, sym in HUMAN_MAP.items()}   # TP53 → ENSG…

class TestConvert2GeneId:
    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_basic_conversion(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        symbols = list(SYMBOL_MAP.keys())
        df = convert2gene_id(symbols, species="human")

        assert isinstance(df, pd.DataFrame)
        assert "gene_id" in df.columns
        assert df.index.name == "query"
        for sym, eid in SYMBOL_MAP.items():
            assert df.loc[sym, "gene_id"] == eid

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_unknown_symbol_falls_back(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        df = convert2gene_id(["UNKNOWN_GENE"], species="human")
        assert df.loc["UNKNOWN_GENE", "gene_id"] == "UNKNOWN_GENE"

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_multi_first(self, MockER):
        # Gene with two IDs
        dup_map = {
            "ENSG00000141510": "TP53",
            "ENSG00000141511": "TP53",   # duplicate symbol
        }
        MockER.return_value = _mock_ensembl_release(dup_map)
        df = convert2gene_id(["TP53"], species="human", multi="first")
        result = df.loc["TP53", "gene_id"]
        assert isinstance(result, str)
        assert result in dup_map

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_multi_all(self, MockER):
        dup_map = {
            "ENSG00000141510": "TP53",
            "ENSG00000141511": "TP53",
        }
        MockER.return_value = _mock_ensembl_release(dup_map)
        df = convert2gene_id(["TP53"], species="human", multi="all")
        result = df.loc["TP53", "gene_id"]
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_multi_join(self, MockER):
        dup_map = {
            "ENSG00000141510": "TP53",
            "ENSG00000141511": "TP53",
        }
        MockER.return_value = _mock_ensembl_release(dup_map)
        df = convert2gene_id(["TP53"], species="human", multi="join")
        result = df.loc["TP53", "gene_id"]
        assert isinstance(result, str)
        assert "|" in result

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_default_species_is_human(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        convert2gene_id(["TP53"])
        MockER.assert_called_once_with(77, species="human")

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_default_release_is_77(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        convert2gene_id(["TP53"], species="human")
        MockER.assert_called_once_with(77, species="human")


# ---------------------------------------------------------------------------
# symbol2id  (AnnData wrapper)
# ---------------------------------------------------------------------------

class TestSymbol2Id:
    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_replaces_var_names(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        adata = _make_adata(list(SYMBOL_MAP.keys()))
        out = symbol2id(adata, species="human")
        assert set(out.var_names) == set(SYMBOL_MAP.values())

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_symbol_column_preserved(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        adata = _make_adata(list(SYMBOL_MAP.keys()))
        out = symbol2id(adata, species="human")
        assert "symbol" in out.var.columns

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_subset_false_keeps_all(self, MockER):
        partial_map = {"ENSG00000141510": "TP53"}
        MockER.return_value = _mock_ensembl_release(partial_map)
        adata = _make_adata(["TP53", "UNKNOWN_GENE"])
        out = symbol2id(adata, species="human", subset=False)
        assert out.n_vars == 2

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_subset_true_drops_unconverted(self, MockER):
        partial_map = {"ENSG00000141510": "TP53"}
        MockER.return_value = _mock_ensembl_release(partial_map)
        adata = _make_adata(["TP53", "UNKNOWN_GENE"])
        out = symbol2id(adata, species="human", subset=True)
        assert "ENSG00000141510" in out.var_names
        assert "UNKNOWN_GENE" not in out.var_names

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_obs_count_unchanged(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        adata = _make_adata(list(SYMBOL_MAP.keys()))
        n_obs = adata.n_obs
        out = symbol2id(adata, species="human")
        assert out.n_obs == n_obs


# ---------------------------------------------------------------------------
# Round-trip: symbol → ID → symbol
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_symbol_id_symbol_roundtrip(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        symbols = list(SYMBOL_MAP.keys())

        id_df = convert2gene_id(symbols, species="human")
        ensembl_ids = id_df["gene_id"].tolist()

        sym_df = convert2gene_symbol(ensembl_ids, species="human")
        recovered = sym_df["symbol"].tolist()

        assert set(recovered) == set(symbols)

    @patch("omicverse.utils._gene_id_conversion.EnsemblRelease")
    def test_id_symbol_id_roundtrip(self, MockER):
        MockER.return_value = _mock_ensembl_release(HUMAN_MAP)
        ensembl_ids = list(HUMAN_MAP.keys())

        sym_df = convert2gene_symbol(ensembl_ids, species="human")
        symbols = sym_df["symbol"].tolist()

        id_df = convert2gene_id(symbols, species="human")
        recovered = id_df["gene_id"].tolist()

        assert set(recovered) == set(ensembl_ids)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_functions_are_registered(self):
        import omicverse.utils  # ensure decorators have run
        from omicverse._registry import _global_registry
        registered_names = list(_global_registry._registry.keys())
        for fn_name in ["convert2gene_symbol", "convert2symbol", "convert2gene_id", "symbol2id"]:
            assert any(fn_name in name for name in registered_names), \
                f"{fn_name} not found in registry"


# ---------------------------------------------------------------------------
# Integration tests (require pre-downloaded Ensembl database)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """
    These tests hit the real pyensembl database.
    Run with: pytest -m integration tests/utils/test_gene_id_conversion.py

    Requires:
        pyensembl install --release 77 --species human
        pyensembl install --release 77 --species mouse
    """

    def test_human_tp53_to_symbol(self):
        df = convert2gene_symbol(["ENSG00000141510"], species="human", ensembl_release=77)
        assert df.loc["ENSG00000141510", "symbol"] == "TP53"

    def test_mouse_trp53_to_symbol(self):
        df = convert2gene_symbol(["ENSMUSG00000059552"], species="mouse", ensembl_release=77)
        assert df.loc["ENSMUSG00000059552", "symbol"] == "Trp53"

    def test_tp53_symbol_to_id(self):
        df = convert2gene_id(["TP53"], species="human", ensembl_release=77)
        assert "ENSG00000141510" in df.loc["TP53", "gene_id"]

    def test_auto_species_detection(self):
        df = convert2gene_symbol(["ENSMUSG00000059552"], ensembl_release=77)
        assert df.loc["ENSMUSG00000059552", "symbol"] == "Trp53"

    def test_adata_convert2symbol(self):
        adata = _make_adata(["ENSG00000141510", "ENSG00000012048"])
        out = convert2symbol(adata)
        assert "TP53" in out.var_names
        assert "BRCA1" in out.var_names

    def test_adata_symbol2id(self):
        adata = _make_adata(["TP53", "BRCA1"])
        out = symbol2id(adata, species="human", ensembl_release=77)
        assert "ENSG00000141510" in out.var_names
        assert "ENSG00000012048" in out.var_names
