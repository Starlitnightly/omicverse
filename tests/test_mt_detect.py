"""Tests for mitochondrial gene prefix auto-detection in pp.qc."""

import numpy as np
import pandas as pd
import pytest

from omicverse.pp._qc import _detect_mt_prefix, _mt_mask


class TestDetectMtPrefix:
    """Test _detect_mt_prefix with various species and edge cases."""

    def test_human_MT(self):
        """Human genes use 'MT-' prefix."""
        genes = pd.Index([
            "GAPDH", "ACTB", "TP53", "MT-CO1", "MT-CO2", "MT-CO3",
            "MT-ND1", "MT-ND2", "MT-ATP6", "MT-CYB", "BRCA1",
        ])
        assert _detect_mt_prefix(genes) == "MT-"

    def test_mouse_mt(self):
        """Mouse genes use 'mt-' prefix."""
        genes = pd.Index([
            "Gapdh", "Actb", "Trp53", "mt-Co1", "mt-Co2", "mt-Co3",
            "mt-Nd1", "mt-Nd2", "mt-Atp6", "mt-Cytb", "Brca1",
        ])
        assert _detect_mt_prefix(genes) == "mt-"

    def test_mixed_case_Mt(self):
        """Some annotations use 'Mt-' prefix."""
        genes = pd.Index(["Mt-Co1", "Mt-Nd1", "Gapdh", "Actb"])
        assert _detect_mt_prefix(genes) == "Mt-"

    def test_drosophila_mt_colon(self):
        """Drosophila genes use 'mt:' prefix."""
        genes = pd.Index([
            "Act5C", "RpL32", "mt:CoI", "mt:CoII", "mt:CoIII",
            "mt:ND1", "mt:Cyt-b", "Gapdh1",
        ])
        assert _detect_mt_prefix(genes) == "mt:"

    def test_arabidopsis_ATMG(self):
        """Arabidopsis thaliana mitochondrial genes use 'ATMG' prefix."""
        genes = pd.Index([
            "AT1G01010", "AT3G18780", "ATMG00010", "ATMG00020",
            "ATMG00030", "ATMG00040", "AT5G08670",
        ])
        assert _detect_mt_prefix(genes) == "ATMG"

    def test_c_elegans_heterogeneous(self):
        """C. elegans has heterogeneous mt gene names (ctc-, nduo-, ctb-)."""
        genes = pd.Index([
            "unc-54", "dpy-7", "ctc-1", "ctc-2", "ctc-3",
            "nduo-1", "nduo-2", "nduo-6", "ctb-1", "act-1",
        ])
        assert _detect_mt_prefix(genes) == "ctc-"

    def test_no_mt_genes_fallback(self):
        """When no MT genes found, fallback to 'MT-'."""
        genes = pd.Index(["GAPDH", "ACTB", "TP53", "BRCA1"])
        assert _detect_mt_prefix(genes) == "MT-"

    def test_human_dominates_mouse(self):
        """When both 'MT-' and 'mt-' present, the one with more matches wins."""
        genes = pd.Index([
            "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND1", "MT-ND2",
            "mt-Co1",  # only one mouse-style
            "GAPDH", "ACTB",
        ])
        assert _detect_mt_prefix(genes) == "MT-"

    def test_mouse_dominates_human(self):
        """When 'mt-' has more matches than 'MT-', select 'mt-'."""
        genes = pd.Index([
            "mt-Co1", "mt-Co2", "mt-Co3", "mt-Nd1", "mt-Nd2",
            "MT-CO1",  # only one human-style
            "Gapdh", "Actb",
        ])
        assert _detect_mt_prefix(genes) == "mt-"

    def test_list_input(self):
        """Works with plain list input (not pd.Index)."""
        genes = ["mt-Co1", "mt-Nd1", "Gapdh"]
        assert _detect_mt_prefix(genes) == "mt-"

    def test_empty_index(self):
        """Empty gene list falls back to 'MT-'."""
        genes = pd.Index([])
        assert _detect_mt_prefix(genes) == "MT-"

    def test_case_insensitive_fallback(self):
        """Case-insensitive fallback detects unusual capitalisation."""
        genes = pd.Index(["mT-CO1", "mT-ND1", "GAPDH"])
        result = _detect_mt_prefix(genes)
        assert result == "mT-"

    def test_ensembl_ids_fallback(self):
        """Ensembl gene IDs have no MT prefix, should fallback."""
        genes = pd.Index([
            "ENSG00000198888", "ENSG00000198763", "ENSG00000141510",
        ])
        assert _detect_mt_prefix(genes) == "MT-"


class TestMtMask:
    """Test _mt_mask helper for multi-prefix species."""

    def test_human_single_prefix(self):
        genes = pd.Index(["MT-CO1", "MT-ND1", "GAPDH", "ACTB"])
        mask = _mt_mask(genes, "MT-")
        assert list(mask) == [True, True, False, False]

    def test_c_elegans_multi_prefix(self):
        """ctc- prefix should also match nduo- and ctb-."""
        genes = pd.Index(["ctc-1", "nduo-2", "ctb-1", "unc-54", "act-1"])
        mask = _mt_mask(genes, "ctc-")
        assert list(mask) == [True, True, True, False, False]

    def test_drosophila(self):
        genes = pd.Index(["mt:CoI", "mt:ND1", "Act5C"])
        mask = _mt_mask(genes, "mt:")
        assert list(mask) == [True, True, False]

    def test_list_input(self):
        genes = ["MT-CO1", "GAPDH"]
        mask = _mt_mask(genes, "MT-")
        assert list(mask) == [True, False]

    def test_explicit_override(self):
        """Explicit mt_startswith bypasses auto-detection."""
        genes = pd.Index(["Gapdh", "mt-Co1", "mt-Co2", "MT-CO1"])
        mask = _mt_mask(genes, "MT-")
        assert list(mask) == [False, False, False, True]

    def test_auto_raises(self):
        """Unresolved 'auto' raises ValueError."""
        with pytest.raises(ValueError, match="auto"):
            _mt_mask(pd.Index(["MT-CO1"]), "auto")

    def test_nduo_expands_to_all_ce(self):
        """User passing 'nduo-' also matches ctc- and ctb-."""
        genes = pd.Index(["ctc-1", "nduo-2", "ctb-1", "unc-54"])
        mask = _mt_mask(genes, "nduo-")
        assert list(mask) == [True, True, True, False]


class TestQcAutoMt:
    """Integration test: verify auto-detection works end-to-end."""

    def test_qc_auto_mouse(self):
        """Detects 'mt-' for mouse data."""
        prefix = _detect_mt_prefix(pd.Index([
            "Gapdh", "mt-Co1", "mt-Co2", "mt-Nd1", "Actb",
        ]))
        assert prefix == "mt-"
        mask = _mt_mask(pd.Index(["mt-Co1", "Gapdh", "mt-Nd1"]), prefix)
        assert sum(mask) == 2

    def test_qc_auto_drosophila(self):
        """Detects 'mt:' for Drosophila data."""
        prefix = _detect_mt_prefix(pd.Index([
            "Act5C", "mt:CoI", "mt:ND1", "RpL32",
        ]))
        assert prefix == "mt:"
        mask = _mt_mask(pd.Index(["mt:CoI", "Act5C", "mt:ND1"]), prefix)
        assert sum(mask) == 2

    def test_qc_auto_c_elegans(self):
        """Detects C. elegans mt genes across multiple prefixes."""
        prefix = _detect_mt_prefix(pd.Index([
            "unc-54", "ctc-1", "ctc-2", "nduo-1", "ctb-1", "act-1",
        ]))
        assert prefix == "ctc-"
        mask = _mt_mask(pd.Index([
            "ctc-1", "nduo-1", "ctb-1", "unc-54",
        ]), prefix)
        assert sum(mask) == 3
